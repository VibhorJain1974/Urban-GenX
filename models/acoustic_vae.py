"""
Urban-GenX | Acoustic Node
VAE over MFCC spectrograms from UrbanSound8K.
Input:  [B, 1, 40, T] MFCC frames
Output: Reconstructed MFCC + latent sample for generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AcousticVAE(nn.Module):
    def __init__(self, mfcc_bins=40, time_frames=128, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        flat_size = mfcc_bins * time_frames  # 5120

        # ── Encoder ──────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024,       256), nn.BatchNorm1d(256),  nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256,       1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024,  flat_size), nn.Tanh()
        )
        self.mfcc_bins   = mfcc_bins
        self.time_frames = time_frames

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + eps * sigma"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: [B, 1, mfcc_bins, time_frames]
        B = x.size(0)
        x_flat = x.view(B, -1)
        h   = self.encoder(x_flat)
        mu  = self.fc_mu(h)
        lv  = self.fc_log_var(h)
        z   = self.reparameterize(mu, lv)
        out = self.decoder(z).view(B, 1, self.mfcc_bins, self.time_frames)
        return out, mu, lv

    @staticmethod
    def loss(recon_x, x, mu, log_var, beta=1.0):
        """Beta-VAE loss: reconstruction + KL divergence"""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss

    @torch.no_grad()
    def generate(self, n_samples=1, device='cpu'):
        """Sample from latent space → synthesize MFCC"""
        z = torch.randn(n_samples, self.latent_dim).to(device)
        return self.decoder(z).view(n_samples, 1, self.mfcc_bins, self.time_frames)
