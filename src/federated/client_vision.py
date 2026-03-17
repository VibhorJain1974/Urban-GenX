"""
Urban-GenX | Federated Vision Client (Flower)
Federated Learning node for cGAN discriminator aggregation.
Only D parameters are federated (FedAvg on discriminator weights).
G is trained locally — generator is NOT shared across federation
(privacy boundary: model trained on private data stays private until DP-clean).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import flwr as fl
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.vision_gan import Generator, Discriminator
from src.utils.data_loader import CityscapesDataset

DEVICE = torch.device("cpu")

# Each FL client uses a different city split of Cityscapes
# to simulate data heterogeneity (non-IID federated scenario)
CLIENT_CITY_SPLITS = {
    "0": ["aachen", "bochum", "bremen"],
    "1": ["cologne", "darmstadt", "dusseldorf"],
    "2": ["erfurt", "frankfurt", "hamburg"],
}


class VisionClient(fl.client.NumPyClient):
    """
    Flower client for Vision GAN.
    Aggregated: Discriminator weights (D)
    Local only:  Generator weights (G)
    """

    def __init__(self, client_id: str = "0", data_root: str = "data/raw/cityscapes"):
        self.client_id = client_id
        self.data_root = data_root

        # Models
        self.G = Generator(noise_dim=100, num_classes=35).to(DEVICE)
        self.D = Discriminator(num_classes=35).to(DEVICE)

        self.opt_g = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.criterion = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        # Load checkpoint if exists
        ckpt_path = f"checkpoints/vision_client{client_id}_checkpoint.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.G.load_state_dict(ckpt.get("generator", self.G.state_dict()))
            self.D.load_state_dict(ckpt.get("discriminator", self.D.state_dict()))
            print(f"[VisionClient {client_id}] Loaded checkpoint from {ckpt_path}")

        # Data
        self._setup_data()

    def _setup_data(self):
        """Load a partition of Cityscapes for this client."""
        try:
            dataset = CityscapesDataset(
                self.data_root, split="train", img_size=64
            )
            # Simple partition: each client gets ~1/3 of the data
            total = len(dataset)
            client_idx = int(self.client_id)
            n_clients = max(len(CLIENT_CITY_SPLITS), 1)
            part_size = total // n_clients
            start = client_idx * part_size
            end = start + part_size if client_idx < n_clients - 1 else total

            from torch.utils.data import Subset
            subset = Subset(dataset, list(range(start, end)))
            self.dataloader = DataLoader(
                subset, batch_size=2, shuffle=True, num_workers=0
            )
            print(f"[VisionClient {self.client_id}] Data partition: {len(subset)} samples")
        except Exception as e:
            print(f"[VisionClient {self.client_id}] Data load failed: {e} — using empty loader")
            self.dataloader = []

    # ── Flower interface ──────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return D parameters as numpy arrays for FedAvg aggregation."""
        return [val.cpu().numpy() for val in self.D.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set D parameters received from server (FedAvg result)."""
        keys = list(self.D.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.D.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Local training round: update D and G for N batches, return updated D params."""
        self.set_parameters(parameters)

        self.G.train()
        self.D.train()

        n_local_steps = int(config.get("local_steps", 5))
        d_losses, g_losses = [], []

        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx >= n_local_steps:
                break
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            real_img, cond = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # ── D step ─────────────────────────────────────────────────────
            for p in self.D.parameters():
                p.requires_grad_(True)
            self.opt_d.zero_grad(set_to_none=True)

            fake_img = self.G(cond).detach()
            real_pred = self.D(cond, real_img)
            fake_pred = self.D(cond, fake_img)

            d_loss = 0.5 * (
                self.criterion(real_pred, torch.ones_like(real_pred)) +
                self.criterion(fake_pred, torch.zeros_like(fake_pred))
            )
            d_loss.backward()
            self.opt_d.step()
            d_losses.append(float(d_loss.item()))

            # ── G step ─────────────────────────────────────────────────────
            for p in self.D.parameters():
                p.requires_grad_(False)
            self.opt_g.zero_grad(set_to_none=True)

            fake_img = self.G(cond)
            fake_pred = self.D(cond, fake_img)
            g_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
            g_loss.backward()
            self.opt_g.step()
            g_losses.append(float(g_loss.item()))

        avg_d = sum(d_losses) / max(1, len(d_losses))
        avg_g = sum(g_losses) / max(1, len(g_losses))
        print(f"[VisionClient {self.client_id}] local fit | D={avg_d:.4f} G={avg_g:.4f}")

        return self.get_parameters(config), len(self.dataloader), {
            "d_loss": avg_d,
            "g_loss": avg_g,
        }

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate D on local data."""
        self.set_parameters(parameters)
        self.D.eval()

        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= 3:
                    break
                if not isinstance(batch, (list, tuple)):
                    continue
                real_img, cond = batch[0].to(DEVICE), batch[1].to(DEVICE)
                fake_img = self.G(cond).detach()
                real_pred = self.D(cond, real_img)
                fake_pred = self.D(cond, fake_img)
                loss = 0.5 * (
                    self.criterion(real_pred, torch.ones_like(real_pred)) +
                    self.criterion(fake_pred, torch.zeros_like(fake_pred))
                )
                total_loss += float(loss.item())
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss, len(self.dataloader), {"d_eval_loss": avg_loss}
