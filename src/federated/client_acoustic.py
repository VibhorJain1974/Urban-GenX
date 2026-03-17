"""
Urban-GenX | Federated Acoustic Client (Flower)
Federated Learning node for AcousticVAE (encoder/decoder) aggregation.
All VAE parameters are federated since the VAE doesn't directly store private samples.
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
import flwr as fl
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.acoustic_vae import AcousticVAE
from src.utils.data_loader import UrbanSound8KDataset

DEVICE = torch.device("cpu")


class AcousticClient(fl.client.NumPyClient):
    """
    Flower client for Acoustic VAE.
    Aggregated: all VAE parameters (encoder + decoder)
    Partitioning: each client gets a subset of UrbanSound8K folds
    """

    # Each client covers different folds (non-IID: different sound environments)
    FOLD_SPLITS = {
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
        self.client_id = client_id
        self.n_mfcc = n_mfcc
        self.time_frames = time_frames

        # Model
        self.model = AcousticVAE(
            mfcc_bins=n_mfcc,
            time_frames=time_frames,
            latent_dim=latent_dim,
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Load checkpoint if exists
        ckpt_path = f"checkpoints/acoustic_client{client_id}_checkpoint.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt.get("model", self.model.state_dict()))
            print(f"[AcousticClient {client_id}] Loaded checkpoint.")

        # Data
        self._setup_data(data_root)

    def _setup_data(self, data_root: str):
        # Handle both folder structures
        if not os.path.exists(os.path.join(data_root, "metadata")):
            alt = os.path.join(data_root, "UrbanSound8K")
            if os.path.exists(os.path.join(alt, "metadata")):
                data_root = alt

        folds = self.FOLD_SPLITS.get(self.client_id, [1, 2, 3])
        try:
            dataset = UrbanSound8KDataset(
                root=data_root,
                folds=folds,
                n_mfcc=self.n_mfcc,
                time_frames=self.time_frames,
            )
            self.dataloader = DataLoader(
                dataset, batch_size=8, shuffle=True, num_workers=0
            )
            print(f"[AcousticClient {self.client_id}] Folds={folds} | {len(dataset)} samples")
        except Exception as e:
            print(f"[AcousticClient {self.client_id}] Data load error: {e}")
            self.dataloader = []

    # ── Flower interface ──────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        keys = list(self.model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.model.train()

        n_local_steps = int(config.get("local_steps", 10))
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x, _) in enumerate(self.dataloader):
            if batch_idx >= n_local_steps:
                break

            x = x.to(DEVICE)
            self.optimizer.zero_grad(set_to_none=True)

            recon, mu, lv = self.model(x)
            loss = AcousticVAE.loss(recon, x, mu, lv, beta=1.0)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[AcousticClient {self.client_id}] local fit | loss={avg_loss:.4f}")

        return self.get_parameters(config), len(self.dataloader), {"vae_loss": avg_loss}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(self.dataloader):
                if batch_idx >= 5:
                    break
                x = x.to(DEVICE)
                recon, mu, lv = self.model(x)
                total_loss += AcousticVAE.loss(recon, x, mu, lv, beta=1.0).item()
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss, len(self.dataloader), {"vae_eval_loss": avg_loss}
