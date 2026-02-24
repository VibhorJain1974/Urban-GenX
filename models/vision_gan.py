"""
Urban-GenX | Vision Node
Convolutional cGAN: maps Cityscapes semantic label maps → RGB street views
Input: 64x64 semantic map (condition)
Output: 64x64 synthetic RGB image
Optimized for: CPU-only, 12GB RAM, batch_size=4
"""

import torch
import torch.nn as nn

# ─── Generator ────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    UNet-style generator with skip connections.
    Condition: semantic label map (num_classes channels).
    Noise: latent z vector concatenated at bottleneck.
    """
    def __init__(self, noise_dim=100, num_classes=35, img_channels=3):
        super().__init__()
        self.noise_dim = noise_dim

        # Encoder (condition stream)
        self.enc1 = self._down_block(num_classes,    64,  norm=False)  # 64→32
        self.enc2 = self._down_block(64,            128)               # 32→16
        self.enc3 = self._down_block(128,           256)               # 16→8
        self.enc4 = self._down_block(256,           512)               # 8→4

        # Bottleneck: inject noise
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + noise_dim, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # Decoder with skip connections
        self.dec4 = self._up_block(512,        256)   # 4→8
        self.dec3 = self._up_block(256 + 256,  128)   # 8→16  (skip from enc3)
        self.dec2 = self._up_block(128 + 128,   64)   # 16→32 (skip from enc2)
        self.dec1 = self._up_block(64 + 64,     32)   # 32→64 (skip from enc1)


        self.final = nn.Sequential(
            # CHANGE stride from 2 to 1 to keep spatial size at 64x64
            nn.ConvTranspose2d(32, img_channels, 3, 1, 1), 
            nn.Tanh()  # Output in [-1, 1]
        )

    def _down_block(self, in_c, out_c, norm=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, condition, noise=None):
        # condition: [B, num_classes, 64, 64]
        if noise is None:
            noise = torch.randn(condition.size(0), self.noise_dim, 1, 1)
        
        e1 = self.enc1(condition)   # [B, 64,  32, 32]
        e2 = self.enc2(e1)          # [B, 128, 16, 16]
        e3 = self.enc3(e2)          # [B, 256, 8,  8 ]
        e4 = self.enc4(e3)          # [B, 512, 4,  4 ]

        # Tile noise to spatial dims of bottleneck
        noise_tiled = noise.expand(-1, -1, e4.size(2), e4.size(3))
        b = self.bottleneck(torch.cat([e4, noise_tiled], dim=1))

        d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.final(d1)


# ─── PatchGAN Discriminator ───────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    PatchGAN discriminator: classifies 16x16 image patches as real/fake.
    Input: concatenation of condition + image.
    Memory-efficient alternative to full-image discriminator.
    """
    def __init__(self, num_classes=35, img_channels=3):
        super().__init__()
        in_c = num_classes + img_channels

        self.model = nn.Sequential(
            # No BN on first layer (Radford et al.)
            nn.Conv2d(in_c,  64,  4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,   128,  4, 2, 1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,  256,  4, 2, 1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,    1,  4, 1, 1)  # Output: patch score map
        )

    def forward(self, condition, image):
        x = torch.cat([condition, image], dim=1)
        return self.model(x)  # [B, 1, H', W'] — patch scores
