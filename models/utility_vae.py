"""
Urban-GenX | Utility/Environmental Node
VAE for synthesizing:
  - Traffic speed time-series (METR-LA, 207 sensors)
  - Water quality parameters (USGS: DO, pH, Turbidity, Temp)

Both use the same architecture with different input dimensions.
CPU/12GB safe: fully-connected VAE, no convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityVAE(nn.Module):
    """
    Generic fully-connected VAE for utility time-series.

    For METR-LA traffic:   input_dim = seq_len * n_sensors (e.g. 12 * 207 = 2484)
    For USGS water:        input_dim = seq_len * n_params  (e.g. 24 * 5  = 120)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list = None,
        name: str = "utility",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.name = name

        if hidden_dims is None:
            # Automatically scale hidden dims to input
            h = min(512, max(64, input_dim // 4))
            hidden_dims = [h * 2, h]

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """x: [B, input_dim] → (mu, log_var)"""
        B = x.size(0)
        h = self.encoder(x.view(B, -1))
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        B = x.size(0)
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        recon = self.decode(z).view(B, -1)
        return recon, mu, lv

    @staticmethod
    def loss(recon_x, x, mu, log_var, beta: float = 1.0):
        """Beta-VAE ELBO loss"""
        B = x.size(0)
        x_flat = x.view(B, -1)
        recon_loss = F.mse_loss(recon_x, x_flat, reduction="sum") / B
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss

    @torch.no_grad()
    def generate(self, n_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """Sample from latent space → synthesize utility vector"""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Latent space interpolation between two utility states (counterfactual)"""
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        results = []
        for t in torch.linspace(0, 1, steps):
            z_interp = mu1 * (1 - t) + mu2 * t
            results.append(self.decode(z_interp))
        return torch.stack(results, dim=0)


# ── Convenience constructors ──────────────────────────────────────────────────

def build_traffic_vae(seq_len: int = 12, n_sensors: int = 207, latent_dim: int = 64) -> UtilityVAE:
    """METR-LA: 207 sensors × 12 time steps"""
    return UtilityVAE(
        input_dim=seq_len * n_sensors,
        latent_dim=latent_dim,
        hidden_dims=[512, 128],
        name="traffic_metrla",
    )


def build_water_vae(seq_len: int = 24, n_params: int = 5, latent_dim: int = 16) -> UtilityVAE:
    """USGS Water: 5 parameters × 24 time steps"""
    return UtilityVAE(
        input_dim=seq_len * n_params,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        name="water_usgs",
    )
