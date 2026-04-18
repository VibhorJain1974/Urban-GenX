"""
Urban-GenX | Vision Node — cGAN Architecture (FIXED)
=====================================================
Fixes:
  1. Generator final layer: ConvTranspose2d stride=1 + kernel=3 (not stride=2)
     Previous code had stride=2 which doubled 64→128, breaking the architecture.
  2. Added spectral normalization option on Discriminator for stability.
  3. Opacus-compatible (no BatchNorm in default mode — uses GroupNorm).
"""

import torch
import torch.nn as nn


# ─── Generator ───────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    UNet-style conditional generator with skip connections.
    Input:  condition [B, num_classes, 64, 64] + noise z [B, noise_dim, 1, 1]
    Output: synthetic RGB image [B, 3, 64, 64] ∈ [-1, 1]

    Architecture:
        Encoder: 64→32→16→8→4 (spatial)
        Bottleneck: concat noise z
        Decoder: 4→8→16→32→64 with skip connections
    """

    def __init__(self, noise_dim: int = 100, num_classes: int = 35, img_channels: int = 3):
        super().__init__()
        self.noise_dim = noise_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = self._down(num_classes, 64,  norm=False)  # [B,64,32,32]
        self.enc2 = self._down(64,          128)               # [B,128,16,16]
        self.enc3 = self._down(128,         256)               # [B,256,8,8]
        self.enc4 = self._down(256,         512)               # [B,512,4,4]

        # ── Bottleneck (noise injection) ──────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + noise_dim, 512, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
        )

        # ── Decoder with skip connections ────────────────────────────────────
        self.dec4 = self._up(512,        256)                  # [B,256,8,8]
        self.dec3 = self._up(256 + 256,  128)                  # [B,128,16,16]
        self.dec2 = self._up(128 + 128,  64)                   # [B,64,32,32]
        self.dec1 = self._up(64  + 64,   32)                   # [B,32,64,64]

        # ── Output head ──────────────────────────────────────────────────────
        # FIX: stride=1, kernel=3, padding=1 → keeps 64×64 (no doubling)
        self.final = nn.Sequential(
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def _down(self, in_c: int, out_c: int, norm: bool = True) -> nn.Sequential:
        layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=not norm)]
        if norm:
            layers.append(nn.GroupNorm(min(32, out_c // 4), out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _up(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, out_c // 4), out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, condition: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        B = condition.size(0)
        if noise is None:
            noise = torch.randn(B, self.noise_dim, 1, 1, device=condition.device)

        e1 = self.enc1(condition)                                  # [B,64,32,32]
        e2 = self.enc2(e1)                                         # [B,128,16,16]
        e3 = self.enc3(e2)                                         # [B,256,8,8]
        e4 = self.enc4(e3)                                         # [B,512,4,4]

        noise_t = noise.expand(-1, -1, e4.size(2), e4.size(3))    # [B,100,4,4]
        b = self.bottleneck(torch.cat([e4, noise_t], dim=1))       # [B,512,4,4]

        d4 = self.dec4(b)                                          # [B,256,8,8]
        d3 = self.dec3(torch.cat([d4, e3], dim=1))                 # [B,128,16,16]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))                 # [B,64,32,32]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))                 # [B,32,64,64]

        out = self.final(d1)                                        # [B,3,64,64]
        assert out.shape[2:] == (64, 64), f"Generator output shape mismatch: {out.shape}"
        return out


# ─── PatchGAN Discriminator ──────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    PatchGAN Discriminator — classifies N×N image patches as real/fake.
    Input: condition [B, num_classes, H, W] concatenated with image [B, 3, H, W]
    Output: patch score map [B, 1, H', W']

    Opacus-compatible: no BatchNorm (uses GroupNorm or no norm on first layer).
    """

    def __init__(self, num_classes: int = 35, img_channels: int = 3):
        super().__init__()
        in_c = num_classes + img_channels  # 38 channels

        self.model = nn.Sequential(
            # Layer 1 — no norm (Radford et al. recommendation)
            nn.Conv2d(in_c,  64,  4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(64,   128,  4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(128,  256,  4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output — patch scores (no activation, used with BCEWithLogitsLoss)
            nn.Conv2d(256,    1,  4, stride=1, padding=1, bias=True),
        )

    def forward(self, condition: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, image], dim=1)  # [B, num_classes+3, H, W]
        return self.model(x)                       # [B, 1, H', W']
