"""
Urban-GenX | Federated Vision Client  (FINAL FIXED VERSION)
============================================================
Flower 1.5 API – Vision cGAN Discriminator federated node.

Fixes vs previous version:
  1. fit() returns len(dataset) not len(dataloader) — Flower FedAvg fix.
  2. evaluate() returns len(dataset) not len(dataloader).
  3. Added _safe_num_examples() helper.
  4. D-params freeze during G step is preserved (no Opacus in FL client
     since we use per-client local GAN training, not DP here).
  5. Graceful handling when Cityscapes data is absent (returns empty metrics).
  6. Explicit Subset import moved to top-level to avoid late ImportError.

Federated design:
  - Only Discriminator (D) weights are shared via FedAvg.
  - Generator (G) stays local on each client.
  - This implements "Private Generator" FL: D improves via federation,
    but G is never released to the server.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
import flwr as fl
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.vision_gan import Generator, Discriminator
from src.utils.data_loader import CityscapesDataset

DEVICE = torch.device("cpu")

# Non-IID partitioning: different city groups per client
CLIENT_CITY_SPLITS: Dict[str, List[str]] = {
    "0": ["aachen", "bochum", "bremen"],
    "1": ["cologne", "darmstadt", "dusseldorf"],
    "2": ["erfurt", "frankfurt", "hamburg"],
}


class VisionClient(fl.client.NumPyClient):
    """
    Flower 1.5 NumPyClient for Vision cGAN.
    Federated : Discriminator (D) weights
    Local only : Generator (G) weights
    """

    def __init__(
        self,
        client_id:  str = "0",
        data_root:  str = "data/raw/cityscapes",
        noise_dim:  int = 100,
        num_classes: int = 35,
    ):
        self.client_id   = client_id
        self.noise_dim   = noise_dim
        self.num_classes = num_classes

        # Models
        self.G = Generator(noise_dim=noise_dim,   num_classes=num_classes).to(DEVICE)
        self.D = Discriminator(num_classes=num_classes).to(DEVICE)

        # Optimizers
        self.opt_g = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Load per-client checkpoint if exists
        ckpt_path = f"checkpoints/vision_client{client_id}_checkpoint.pth"
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self.G.load_state_dict(ckpt.get("generator",     self.G.state_dict()))
                self.D.load_state_dict(ckpt.get("discriminator", self.D.state_dict()))
                print(f"[VisionClient {client_id}] Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"[VisionClient {client_id}] Checkpoint load failed: {e} — using random weights")

        # Data
        self._setup_data(data_root)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _setup_data(self, data_root: str) -> None:
        """Load a partition of Cityscapes for this client."""
        try:
            full_dataset = CityscapesDataset(data_root, split="train", img_size=64)
            total        = len(full_dataset)
            n_clients    = len(CLIENT_CITY_SPLITS)
            client_idx   = int(self.client_id) % n_clients
            part_size    = total // n_clients
            start        = client_idx * part_size
            end          = start + part_size if client_idx < n_clients - 1 else total

            subset = Subset(full_dataset, list(range(start, end)))
            self.dataloader = DataLoader(
                subset,
                batch_size=2,
                shuffle=True,
                num_workers=0,
            )
            print(
                f"[VisionClient {self.client_id}] "
                f"Partition [{start}:{end}] | {len(subset)} examples"
            )
        except Exception as e:
            print(
                f"[VisionClient {self.client_id}] "
                f"Data setup failed (Cityscapes not found?): {e}"
            )
            self.dataloader = None

    def _safe_num_examples(self) -> int:
        """
        Return number of EXAMPLES for Flower FedAvg weighting.
        FIX: use len(dataset), NOT len(dataloader).
        """
        if self.dataloader is None:
            return 0
        if hasattr(self.dataloader, "dataset"):
            return len(self.dataloader.dataset)
        return 0

    # ── Flower NumPyClient interface ─────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return Discriminator parameter tensors as numpy arrays."""
        return [val.cpu().numpy() for val in self.D.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Apply server-aggregated Discriminator parameters."""
        keys       = list(self.D.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.D.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Local training: run N GAN steps, return updated D params.

        Returns:
            (updated_D_params, num_examples, metrics_dict)
            FIX: num_examples uses len(dataset).
        """
        self.set_parameters(parameters)

        if self.dataloader is None:
            return self.get_parameters(config), 0, {"d_loss": 0.0, "g_loss": 0.0}

        self.G.train()
        self.D.train()

        n_local_steps = int(config.get("local_steps", 5))
        d_losses: List[float] = []
        g_losses: List[float] = []

        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx >= n_local_steps:
                break
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue

            real_img, cond = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # ── Discriminator step ───────────────────────────────────────
            for p in self.D.parameters():
                p.requires_grad_(True)
            self.opt_d.zero_grad(set_to_none=True)

            fake_img  = self.G(cond).detach()
            real_pred = self.D(cond, real_img)
            fake_pred = self.D(cond, fake_img)
            d_loss = 0.5 * (
                self.criterion(real_pred, torch.ones_like(real_pred)) +
                self.criterion(fake_pred, torch.zeros_like(fake_pred))
            )
            d_loss.backward()
            self.opt_d.step()
            d_losses.append(float(d_loss.item()))

            # ── Generator step (D params frozen, no grad accumulation) ───
            for p in self.D.parameters():
                p.requires_grad_(False)
            self.opt_g.zero_grad(set_to_none=True)

            fake_img  = self.G(cond)
            fake_pred = self.D(cond, fake_img)
            g_loss    = self.criterion(fake_pred, torch.ones_like(fake_pred))
            g_loss.backward()
            self.opt_g.step()
            g_losses.append(float(g_loss.item()))

        avg_d = sum(d_losses) / max(1, len(d_losses))
        avg_g = sum(g_losses) / max(1, len(g_losses))
        print(
            f"[VisionClient {self.client_id}] fit done | "
            f"D={avg_d:.4f} G={avg_g:.4f}"
        )

        # CRITICAL FIX: return num_examples (not num_batches)
        return self.get_parameters(config), self._safe_num_examples(), {
            "d_loss": avg_d,
            "g_loss": avg_g,
        }

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate Discriminator on local data.

        Returns:
            (d_loss, num_examples, metrics_dict)
            FIX: num_examples uses len(dataset).
        """
        self.set_parameters(parameters)

        if self.dataloader is None:
            return 0.0, 0, {"d_eval_loss": 0.0}

        self.D.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= 3:
                    break
                if not isinstance(batch, (list, tuple)):
                    continue
                real_img, cond = batch[0].to(DEVICE), batch[1].to(DEVICE)
                fake_img  = self.G(cond).detach()
                real_pred = self.D(cond, real_img)
                fake_pred = self.D(cond, fake_img)
                loss = 0.5 * (
                    self.criterion(real_pred, torch.ones_like(real_pred)) +
                    self.criterion(fake_pred, torch.zeros_like(fake_pred))
                )
                total_loss += float(loss.item())
                n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)

        # CRITICAL FIX: return num_examples (not num_batches)
        return avg_loss, self._safe_num_examples(), {
            "d_eval_loss": avg_loss,
        }
