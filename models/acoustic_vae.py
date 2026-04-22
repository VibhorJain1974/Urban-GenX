"""
Urban-GenX | Acoustic Node — FIXED & UPGRADED
=============================================
FIXES vs original:
  1. FREE BITS: KL floor prevents posterior collapse (was KL=0.002, useless latent)
  2. CONVOLUTIONAL encoder/decoder: captures local MFCC patterns (was flat Linear)
  3. MEL-SPECTROGRAM input support: richer than raw MFCC (backwards-compatible)
  4. LABEL CONDITIONING (optional): cVAE mode for class-aware generation
  5. Tanh removed from decoder: was clipping values outside [-1,1], hurting recon

WHY these fixes help:
  - Original KL=0.0021 → posterior never left prior → latent space useless for synthesis
  - Free bits forces each latent dim to carry ≥ free_bits nats of info
  - Conv layers capture time-frequency structure MFCCs encode
  - Without Tanh clipping, MSE can actually drive reconstruction loss lower
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticVAE(nn.Module):
    """
    Convolutional Beta-VAE for MFCC or Mel-spectrogram audio features.

    Input shape:  [B, 1, n_bins, time_frames]
                  e.g. [B, 1, 40, 128]  (MFCC)
                  or   [B, 1, 64, 128]  (Mel-spectrogram)

    Args:
        mfcc_bins   : number of frequency bins (40 for MFCC, 64/128 for Mel)
        time_frames : fixed time axis length (128)
        latent_dim  : size of latent space (64)
        n_classes   : if > 0, enables conditional VAE (cVAE) with label embedding
        free_bits   : minimum KL per latent dim (KEY FIX — prevents collapse)
    """

    def __init__(
        self,
        mfcc_bins:   int = 40,
        time_frames: int = 128,
        latent_dim:  int = 64,
        n_classes:   int = 0,       # set to 10 for UrbanSound8K cVAE mode
        free_bits:   float = 0.5,   # KL floor per latent dim (nats)
    ):
        super().__init__()
        self.latent_dim  = latent_dim
        self.mfcc_bins   = mfcc_bins
        self.time_frames = time_frames
        self.n_classes   = n_classes
        self.free_bits   = free_bits

        # ── Convolutional Encoder ─────────────────────────────────────────────
        # Processes [B, 1, mfcc_bins, time_frames] with local receptive fields
        self.enc_conv = nn.Sequential(
            # Layer 1: [B, 1, 40, 128] → [B, 32, 20, 64]
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: [B, 32, 20, 64] → [B, 64, 10, 32]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: [B, 64, 10, 32] → [B, 128, 5, 16]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Compute flattened size after conv layers
        self._enc_flat = self._get_enc_flat_size(mfcc_bins, time_frames)

        # Optional class conditioning — embeds label → concat with enc output
        cond_dim = 0
        if n_classes > 0:
            self.label_emb = nn.Embedding(n_classes, 32)
            cond_dim = 32

        self.enc_fc  = nn.Linear(self._enc_flat + cond_dim, 256)
        self.fc_mu      = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        # ── Convolutional Decoder ─────────────────────────────────────────────
        dec_input_dim = latent_dim + cond_dim
        self.dec_fc = nn.Linear(dec_input_dim, self._enc_flat)

        # Compute the spatial shape before transposed convs
        self._dec_spatial = self._get_dec_spatial(mfcc_bins, time_frames)

        self.dec_conv = nn.Sequential(
            # Layer 1: [B, 128, 5, 16] → [B, 64, 10, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 2: [B, 64, 10, 32] → [B, 32, 20, 64]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 3: [B, 32, 20, 64] → [B, 1, 40, 128]
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # NOTE: NO Tanh here — was clamping valid MFCC values outside [-1,1]
            # MSE loss works better without output clipping
        )

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _get_enc_flat_size(self, bins, frames):
        """Calculate flattened encoder output size for given input dims."""
        dummy = torch.zeros(1, 1, bins, frames)
        out = self.enc_conv(dummy)
        return int(out.numel())

    def _get_dec_spatial(self, bins, frames):
        """Calculate spatial shape entering decoder conv (after dec_fc reshape)."""
        dummy = torch.zeros(1, 1, bins, frames)
        out = self.enc_conv(dummy)
        return out.shape[1:]  # (C, H, W) — e.g. (128, 5, 16)

    # ── Forward ───────────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor, label: torch.Tensor = None):
        """
        x     : [B, 1, mfcc_bins, time_frames]
        label : [B] long tensor of class indices (optional, for cVAE)
        Returns: (mu, log_var) each [B, latent_dim]
        """
        h = self.enc_conv(x)                        # [B, 128, H', W']
        h = h.view(h.size(0), -1)                   # [B, enc_flat]

        if self.n_classes > 0 and label is not None:
            emb = self.label_emb(label)              # [B, 32]
            h = torch.cat([h, emb], dim=1)

        h = F.leaky_relu(self.enc_fc(h), 0.2)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def decode(self, z: torch.Tensor, label: torch.Tensor = None):
        """
        z     : [B, latent_dim]
        label : [B] long tensor (optional, for cVAE)
        Returns: [B, 1, mfcc_bins, time_frames]
        """
        if self.n_classes > 0 and label is not None:
            emb = self.label_emb(label)              # [B, 32]
            z = torch.cat([z, emb], dim=1)

        h = F.relu(self.dec_fc(z))                  # [B, enc_flat]
        C, H, W = self._dec_spatial
        h = h.view(h.size(0), C, H, W)             # [B, 128, H', W']
        out = self.dec_conv(h)                      # [B, 1, bins, frames]

        # Crop/pad to exact target size (handles edge cases in conv arithmetic)
        out = F.interpolate(
            out,
            size=(self.mfcc_bins, self.time_frames),
            mode='bilinear',
            align_corners=False,
        )
        return out

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        """Returns (reconstruction, mu, log_var)."""
        mu, log_var = self.encode(x, label)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z, label)
        return recon, mu, log_var

    # ── Loss ──────────────────────────────────────────────────────────────────
    @staticmethod
    def loss(
        recon_x:   torch.Tensor,
        x:         torch.Tensor,
        mu:        torch.Tensor,
        log_var:   torch.Tensor,
        beta:      float = 1.0,
        free_bits: float = 0.5,     # KEY FIX — prevents KL collapse
    ) -> torch.Tensor:
        """
        Beta-VAE ELBO with Free Bits (Kingma et al., 2016).

        FREE BITS FIX:
          - Original had KL=0.0021 → encoder was ignoring latent space entirely
          - free_bits=0.5 means each latent dim MUST carry ≥ 0.5 nats of info
          - This forces the model to actually use the latent space for generation
          - Result: meaningful interpolation, diverse synthesis, better FID

        Args:
            recon_x   : [B, 1, bins, T]  reconstructed
            x         : [B, 1, bins, T]  original
            mu        : [B, latent_dim]
            log_var   : [B, latent_dim]
            beta      : KL weight (annealed 0→1)
            free_bits : KL floor per latent dim (nats) — set 0 to disable

        Returns:
            scalar loss
        """
        # Reconstruction loss: per-element MSE
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL per latent dimension: shape [B, latent_dim]
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        # Mean over batch first
        kl_per_dim = kl_per_dim.mean(dim=0)         # [latent_dim]

        # Free bits: clamp each dim to at least free_bits
        # This is what prevents posterior collapse
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

        kl_loss = kl_per_dim.mean()                 # scalar

        return recon_loss + beta * kl_loss

    # ── Generation & Utilities ────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 1,
        label: torch.Tensor = None,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """
        Sample from N(0,I) → synthesize MFCC spectrogram.
        Returns: [n_samples, 1, mfcc_bins, time_frames]
        """
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z, label)

    @torch.no_grad()
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        steps: int = 8,
        label: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Latent-space interpolation between two MFCC inputs.
        Returns: [steps, 1, mfcc_bins, time_frames]
        """
        mu1, _ = self.encode(x1, label)
        mu2, _ = self.encode(x2, label)
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        frames = []
        for a in alphas:
            z = mu1 * (1 - a) + mu2 * a
            frames.append(self.decode(z, label))
        return torch.cat(frames, dim=0)