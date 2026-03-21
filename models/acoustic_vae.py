"""
Urban-GenX | Acoustic Node
VAE over MFCC spectrograms from UrbanSound8K.

Input:  [B, 1, mfcc_bins, time_frames]  e.g. [B, 1, 40, 128]
Output: Reconstructed MFCC + latent (mu, log_var) for generation

FIX (vs original): loss() uses reduction='mean' instead of 'sum'
  - 'sum' was producing ~50,000 loss values (5120-element sum per sample)
  - 'mean' produces ~0.3–1.5 range, correct for normalized MFCC input
  - This makes convergence visible and KL/recon balance meaningful
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticVAE(nn.Module):
    def __init__(self, mfcc_bins: int = 40, time_frames: int = 128, latent_dim: int = 64):
        super().__init__()
        self.latent_dim  = latent_dim
        self.mfcc_bins   = mfcc_bins
        self.time_frames = time_frames
        flat_size        = mfcc_bins * time_frames   # 5120 for default params

        # ── Encoder ──────────────────────────────────────────────────────────
        # 5120 → 1024 → 256 → (mu, log_var) ∈ R^64
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        # 64 → 256 → 1024 → 5120
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, flat_size),
            nn.Tanh(),   # output in [-1, 1] matching normalized MFCC input
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * sigma  (differentiable sampling)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor):
        """x: [B, 1, mfcc_bins, time_frames] → (mu, log_var) each [B, latent_dim]"""
        h       = self.encoder(x)      # flattens internally
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] → [B, 1, mfcc_bins, time_frames]"""
        out = self.decoder(z)
        return out.view(-1, 1, self.mfcc_bins, self.time_frames)

    def forward(self, x: torch.Tensor):
        """Full forward pass. Returns (reconstruction, mu, log_var)."""
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z)
        return recon, mu, log_var

    # ── Loss ─────────────────────────────────────────────────────────────────
    @staticmethod
    def loss(recon_x: torch.Tensor,
             x: torch.Tensor,
             mu: torch.Tensor,
             log_var: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
        """
        Beta-VAE ELBO loss.

        FIX: reduction='mean' (not 'sum').
          - 'mean' divides by (B * mfcc_bins * time_frames) so the loss is
            per-element, making it ~0.3–1.5 for well-normalized MFCCs.
          - KL is also mean-reduced (per latent dim per sample) so beta
            stays interpretable across different latent_dim choices.

        Args:
            recon_x : reconstructed MFCC  [B, 1, mfcc_bins, T]
            x       : original MFCC       [B, 1, mfcc_bins, T]
            mu      : latent mean          [B, latent_dim]
            log_var : latent log-variance  [B, latent_dim]
            beta    : KL weight (annealed 0→1 during training)

        Returns:
            scalar loss tensor
        """
        # Reconstruction: mean squared error per element
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # KL divergence: mean over batch AND latent dims
        # = -0.5 * mean_over_batch(sum_over_latent(1 + lv - mu^2 - exp(lv)))
        # Dividing by latent_dim normalises KL independently of latent_dim size
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        ) / mu.size(1)

        return recon_loss + beta * kl_loss

    # ── Generation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, n_samples: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from N(0,I) latent space → synthesize MFCC spectrogram.
        Returns: [n_samples, 1, mfcc_bins, time_frames]
        """
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    steps: int = 8) -> torch.Tensor:
        """
        Linearly interpolate in latent space between two MFCC inputs.
        Useful for smooth soundscape transitions in the dashboard.
        Returns: [steps, 1, mfcc_bins, time_frames]
        """
        mu1, lv1 = self.encode(x1)
        mu2, lv2 = self.encode(x2)
        alphas   = torch.linspace(0, 1, steps, device=x1.device)
        frames   = []
        for a in alphas:
            z = mu1 * (1 - a) + mu2 * a   # linear interp (no noise for clean path)
            frames.append(self.decode(z))
        return torch.cat(frames, dim=0)    # [steps, 1, bins, T]
