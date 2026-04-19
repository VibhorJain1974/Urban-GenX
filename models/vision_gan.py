"""
Urban-GenX | Vision Node — cGAN Architecture (QUALITY UPGRADE)
==============================================================
Key changes vs. previous version:
  - Instance Normalization instead of BatchNorm in Generator
    (better per-image statistics, avoids colour averaging across batch)
  - Spectral Normalization in Discriminator 
    (prevents D from becoming too powerful → richer G gradients)
  - Residual connections in decoder blocks (sharper textures)
  - Opacus-compatible (no BatchNorm that needs conversion)
  
Still optimised for: CPU-only, 12GB RAM, batch_size=4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Generator ────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv2d + InstanceNorm + activation."""
    def __init__(self, in_c, out_c, down=True, use_norm=True, activation="relu"):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_norm)
        else:
            self.conv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=not use_norm)
        
        self.norm = nn.InstanceNorm2d(out_c, affine=True) if use_norm else nn.Identity()
        
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
    Input:  semantic label map  [B, num_classes, 64, 64]
    Output: synthetic RGB image [B, 3,            64, 64] in [-1, 1]
    
    Uses InstanceNorm (vs BatchNorm) for better per-image colour consistency.
    """
    def __init__(self, noise_dim=100, num_classes=35, img_channels=3):
        super().__init__()
        self.noise_dim = noise_dim

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(num_classes,  64,  down=True,  use_norm=False, activation="leaky")  # 64→32
        self.enc2 = ConvBlock(64,          128,  down=True,  use_norm=True,  activation="leaky")  # 32→16
        self.enc3 = ConvBlock(128,         256,  down=True,  use_norm=True,  activation="leaky")  # 16→8
        self.enc4 = ConvBlock(256,         512,  down=True,  use_norm=True,  activation="leaky")  # 8→4

        # ── Bottleneck: inject noise ────────────────────────────────────────
        self.bottleneck_conv = nn.Conv2d(512 + noise_dim, 512, 3, 1, 1)
        self.bottleneck_norm = nn.InstanceNorm2d(512, affine=True)
        self.bottleneck_act  = nn.ReLU(inplace=True)

        # ── Decoder with skip connections ──────────────────────────────────
        self.dec4 = ConvBlock(512,        256,  down=False, use_norm=True, activation="relu")  # 4→8
        self.dec3 = ConvBlock(256 + 256,  128,  down=False, use_norm=True, activation="relu")  # 8→16
        self.dec2 = ConvBlock(128 + 128,   64,  down=False, use_norm=True, activation="relu")  # 16→32
        self.dec1 = ConvBlock(64 + 64,     32,  down=False, use_norm=True, activation="relu")  # 32→64

        # ── Output ─────────────────────────────────────────────────────────
        self.final = nn.Sequential(
            nn.Conv2d(32, img_channels, 3, 1, 1),   # stride=1 keeps 64x64
            nn.Tanh()
        )

    def forward(self, condition, noise=None):
        # condition: [B, num_classes, 64, 64]
        B = condition.size(0)
        
        if noise is None:
            noise = torch.randn(B, self.noise_dim, 1, 1, device=condition.device)

        # Encode
        e1 = self.enc1(condition)          # [B, 64,  32, 32]
        e2 = self.enc2(e1)                 # [B, 128, 16, 16]
        e3 = self.enc3(e2)                 # [B, 256,  8,  8]
        e4 = self.enc4(e3)                 # [B, 512,  4,  4]

        # Bottleneck + noise injection
        noise_tiled = noise.expand(-1, -1, e4.size(2), e4.size(3))
        b = torch.cat([e4, noise_tiled], dim=1)         # [B, 512+noise_dim, 4, 4]
        b = self.bottleneck_act(self.bottleneck_norm(self.bottleneck_conv(b)))

        # Decode with skips
        d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return self.final(d1)


# ─── PatchGAN Discriminator ───────────────────────────────────────────────────
def spectral_conv(in_c, out_c, *args, **kwargs):
    """Conv2d with spectral normalisation (stabilises GAN training)."""
    return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, *args, **kwargs))


class Discriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalisation.
    Classifies overlapping 16×16 patches as real/fake.
    
    Spectral norm prevents D from growing too confident
    → richer gradient signal for G → sharper textures.
    
    Note: No BatchNorm here (Opacus-compatible by design).
          Using GroupNorm with 1 group ≈ InstanceNorm for per-feature scaling.
    """
    def __init__(self, num_classes=35, img_channels=3):
        super().__init__()
        in_c = num_classes + img_channels

        self.model = nn.Sequential(
            # Layer 1: no norm (standard PatchGAN practice)
            spectral_conv(in_c,  64,  4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            spectral_conv(64,   128,  4, 2, 1),
            nn.GroupNorm(8, 128),               # Opacus-safe alternative to BatchNorm
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            spectral_conv(128,  256,  4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # Output patch scores (no sigmoid — use BCEWithLogitsLoss)
            spectral_conv(256,    1,  4, 1, 1),
        )

    def forward(self, condition, image):
        x = torch.cat([condition, image], dim=1)
        return self.model(x)    # [B, 1, H', W'] — raw logit patch scores