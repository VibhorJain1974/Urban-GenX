"""
Urban-GenX | Federated Acoustic Client  (FINAL FIXED VERSION)
==============================================================
Flower 1.5 API – AcousticVAE federated node.

Fixes vs previous version:
  1. fit() returns len(self.dataloader.dataset) not len(self.dataloader)
     Flower expects number of EXAMPLES, not number of batches.
     Wrong count breaks FedAvg weighted aggregation.
  2. evaluate() returns len(self.dataloader.dataset) for same reason.
  3. Added _safe_num_examples() helper to avoid AttributeError on edge cases.
  4. AcousticVAE.loss() is called as static method (correct API).
  5. Graceful empty-loader handling returns 0 examples (not crash).
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import flwr as fl
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.acoustic_vae import AcousticVAE
from src.utils.data_loader import UrbanSound8KDataset

DEVICE = torch.device("cpu")


class AcousticClient(fl.client.NumPyClient):
    """
    Flower 1.5 NumPyClient for AcousticVAE.
    Federated: all VAE parameters (encoder + decoder).
    Partitioning: each client gets a non-IID subset of UrbanSound8K folds.
    """

    # Non-IID fold assignments (simulates different acoustic environments per city)
    FOLD_SPLITS: Dict[str, List[int]] = {
        "0": [1, 2, 3, 4],
        "1": [5, 6, 7],
        "2": [8, 9, 10],
    }

    def __init__(
        self,
        client_id: str = "0",
        data_root: str = "data/raw/urbansound8k",
        n_mfcc: int = 40,
        time_frames: int = 128,
        latent_dim: int = 64,
    ):
        self.client_id    = client_id
        self.n_mfcc       = n_mfcc
        self.time_frames  = time_frames
        self.latent_dim   = latent_dim

        # Model & optimizer
        self.model = AcousticVAE(
            mfcc_bins=n_mfcc,
            time_frames=time_frames,
            latent_dim=latent_dim,
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Load client-specific checkpoint (if exists)
        ckpt_path = f"checkpoints/acoustic_client{client_id}_checkpoint.pth"
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict(ckpt.get("model", {}))
                print(f"[AcousticClient {client_id}] Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"[AcousticClient {client_id}] Checkpoint load failed: {e} — using random weights")

        # Data setup
        self._setup_data(data_root)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _resolve_data_root(self, root: str) -> str:
        """Auto-detect correct UrbanSound8K folder layout (Kaggle vs soundata)."""
        if os.path.exists(os.path.join(root, "metadata", "UrbanSound8K.csv")):
            return root
        alt = os.path.join(root, "UrbanSound8K")
        if os.path.exists(os.path.join(alt, "metadata", "UrbanSound8K.csv")):
            return alt
        return root  # fallback; dataset loader will give clear error

    def _setup_data(self, data_root: str) -> None:
        data_root = self._resolve_data_root(data_root)
        folds = self.FOLD_SPLITS.get(self.client_id, [1, 2, 3])

        try:
            dataset = UrbanSound8KDataset(
                root=data_root,
                folds=folds,
                n_mfcc=self.n_mfcc,
                time_frames=self.time_frames,
            )
            self.dataloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=True,
                num_workers=0,
            )
            print(
                f"[AcousticClient {self.client_id}] "
                f"Folds={folds} | {len(dataset)} examples loaded"
            )
        except Exception as e:
            print(f"[AcousticClient {self.client_id}] Data setup failed: {e}")
            self.dataloader = None

    def _safe_num_examples(self) -> int:
        """
        Return number of examples for Flower.
        FIX: use len(dataset), NOT len(dataloader).
        Flower uses this for weighted FedAvg aggregation.
        """
        if self.dataloader is None:
            return 0
        if hasattr(self.dataloader, "dataset"):
            return len(self.dataloader.dataset)
        return 0

    # ── Flower NumPyClient interface ─────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return all VAE parameter tensors as numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set VAE parameters from server aggregation result."""
        keys       = list(self.model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Local training round.

        Returns:
            (updated_parameters, num_examples, metrics_dict)
            FIX: num_examples = len(dataset) — Flower requires EXAMPLES, not batches.
        """
        self.set_parameters(parameters)

        if self.dataloader is None:
            return self.get_parameters(config), 0, {"vae_loss": 0.0}

        self.model.train()
        n_local_steps = int(config.get("local_steps", 10))
        total_loss    = 0.0
        n_batches     = 0

        for batch_idx, (x, _) in enumerate(self.dataloader):
            if batch_idx >= n_local_steps:
                break
            x = x.to(DEVICE)

            self.optimizer.zero_grad(set_to_none=True)
            recon, mu, lv = self.model(x)
            loss          = AcousticVAE.loss(recon, x, mu, lv, beta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)
        print(
            f"[AcousticClient {self.client_id}] fit done | "
            f"steps={n_batches} | avg_loss={avg_loss:.4f}"
        )

        # CRITICAL FIX: return num_examples (not num_batches)
        return self.get_parameters(config), self._safe_num_examples(), {
            "vae_loss": avg_loss,
        }

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Local evaluation round.

        Returns:
            (loss, num_examples, metrics_dict)
            FIX: num_examples = len(dataset), not len(dataloader).
        """
        self.set_parameters(parameters)

        if self.dataloader is None:
            return 0.0, 0, {"vae_eval_loss": 0.0}

        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(self.dataloader):
                if batch_idx >= 5:
                    break
                x         = x.to(DEVICE)
                recon, mu, lv = self.model(x)
                total_loss += float(
                    AcousticVAE.loss(recon, x, mu, lv, beta=1.0).item()
                )
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # CRITICAL FIX: return num_examples (not num_batches)
        return avg_loss, self._safe_num_examples(), {
            "vae_eval_loss": avg_loss,
        }
