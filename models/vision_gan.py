"""
Urban-GenX | Vision Node — cGAN Architecture
=============================================
Classes:
  Generator       - UNet cGAN, GroupNorm (Opacus + batch-size-1 safe)
  Discriminator   - PatchGAN with spectral_norm, Phase 1 only
  DiscriminatorDP - PatchGAN plain Conv2d, Opacus-compatible, Phase 2 only
  discriminator_to_dp() - transfers weights Discriminator -> DiscriminatorDP
"""

import torch
import torch.nn as nn


def _make_groupnorm(out_c: int) -> nn.GroupNorm:
    """
    Safe GroupNorm: finds largest divisor of out_c that is <= 32.
    Works for any channel count including 32, 64, 128, 256, 512.
    """
    num_groups = min(32, out_c)
    while out_c % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, out_c)


# ─── Generator ────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_norm=True, activation="relu"):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_norm)
        else:
            self.conv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=not use_norm)

        # GroupNorm instead of InstanceNorm2d:
        #   - Works at batch size 1 (Opacus Poisson sampling can emit size-1 batches)
        #   - InstanceNorm2d crashes with IndexError when batch size == 1
        #   - GroupNorm has no running_mean/running_var, so strict=False
        #     checkpoint loading skips the mismatched keys cleanly
        self.norm = _make_groupnorm(out_c) if use_norm else nn.Identity()

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Generator(nn.Module):
    """
    UNet-style conditional generator.
    Input:  [B, num_classes, 64, 64]  semantic label map
    Output: [B, 3, 64, 64]            synthetic RGB in [-1, 1]

    Norm: GroupNorm throughout (safe for Opacus Poisson batches of size 1).
    """
    def __init__(self, noise_dim=100, num_classes=35, img_channels=3):
        super().__init__()
        self.noise_dim = noise_dim

        self.enc1 = ConvBlock(num_classes, 64,  down=True,  use_norm=False, activation="leaky")
        self.enc2 = ConvBlock(64,         128,  down=True,  use_norm=True,  activation="leaky")
        self.enc3 = ConvBlock(128,        256,  down=True,  use_norm=True,  activation="leaky")
        self.enc4 = ConvBlock(256,        512,  down=True,  use_norm=True,  activation="leaky")

        self.bottleneck_conv = nn.Conv2d(512 + noise_dim, 512, 3, 1, 1)
        self.bottleneck_norm = nn.GroupNorm(32, 512)   # was InstanceNorm2d(512)
        self.bottleneck_act  = nn.ReLU(inplace=True)

        self.dec4 = ConvBlock(512,       256, down=False, use_norm=True, activation="relu")
        self.dec3 = ConvBlock(256 + 256, 128, down=False, use_norm=True, activation="relu")
        self.dec2 = ConvBlock(128 + 128,  64, down=False, use_norm=True, activation="relu")
        self.dec1 = ConvBlock(64 + 64,    32, down=False, use_norm=True, activation="relu")

        self.final = nn.Sequential(
            nn.Conv2d(32, img_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, condition, noise=None):
        B = condition.size(0)
        if noise is None:
            noise = torch.randn(B, self.noise_dim, 1, 1, device=condition.device)

        e1 = self.enc1(condition)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        noise_tiled = noise.expand(-1, -1, e4.size(2), e4.size(3))
        b = torch.cat([e4, noise_tiled], dim=1)
        b = self.bottleneck_act(self.bottleneck_norm(self.bottleneck_conv(b)))

        d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.final(d1)


# ─── Discriminator: spectral_norm version (Phase 1 only) ─────────────────────
class Discriminator(nn.Module):
    """
    PatchGAN with spectral normalisation.
    Used ONLY in Phase 1 (standard GAN training, no Opacus).
    Spectral norm is NOT compatible with Opacus PrivacyEngine.
    For Phase 2, use DiscriminatorDP via discriminator_to_dp().
    """
    def __init__(self, num_classes=35, img_channels=3):
        super().__init__()
        in_c = num_classes + img_channels
        sn = nn.utils.spectral_norm

        self.model = nn.Sequential(
            sn(nn.Conv2d(in_c, 64,  4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(64,  128, 4, 2, 1)),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(256,   1, 4, 1, 1)),
        )

    def forward(self, condition, image):
        x = torch.cat([condition, image], dim=1)
        return self.model(x)


# ─── DiscriminatorDP: plain Conv2d version (Phase 2, Opacus) ─────────────────
class DiscriminatorDP(nn.Module):
    """
    Same architecture as Discriminator but NO spectral_norm.

    Why this exists:
      spectral_norm wraps Conv2d and stores weights as weight_orig + weight_u +
      weight_v instead of weight. Opacus per-sample gradient hooks cannot
      attach to spectral_norm layers. ModuleValidator.fix() does NOT help —
      it only converts BatchNorm, not spectral_norm.

    This class is only used in Phase 2 (DP fine-tuning).
    Weights are transferred from the trained Discriminator via discriminator_to_dp().
    """
    def __init__(self, num_classes=35, img_channels=3):
        super().__init__()
        in_c = num_classes + img_channels

        self.model = nn.Sequential(
            nn.Conv2d(in_c, 64,  4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64,  128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256,   1, 4, 1, 1),
        )

    def forward(self, condition, image):
        x = torch.cat([condition, image], dim=1)
        return self.model(x)


def discriminator_to_dp(D_spectral: Discriminator) -> DiscriminatorDP:
    """
    Transfer trained weights from Discriminator (spectral_norm)
    into DiscriminatorDP (plain Conv2d) for Opacus wrapping.

    spectral_norm renames 'weight' -> 'weight_orig'.
    This function maps weight_orig -> weight for each Conv2d layer.
    GroupNorm weight/bias keys are identical and copy directly.
    """
    D_dp = DiscriminatorDP()
    src = D_spectral.state_dict()
    dst = D_dp.state_dict()

    result = {}
    for dst_key in dst.keys():
        if dst_key in src:
            result[dst_key] = src[dst_key]
        else:
            orig_key = dst_key + "_orig"
            if orig_key in src:
                result[dst_key] = src[orig_key]
            else:
                result[dst_key] = dst[dst_key]

    D_dp.load_state_dict(result, strict=True)
    return D_dp