"""
Urban-GenX | Acoustic Node Training
Trains AcousticVAE on UrbanSound8K MFCCs with:
  - Epoch checkpointing (crash recovery)
  - tqdm progress bars
  - ntfy.sh mobile alerts
  - Beta-VAE annealing for better disentanglement
"""

import os
import sys
import traceback
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.acoustic_vae import AcousticVAE
from src.utils.data_loader import UrbanSound8KDataset
from src.utils.notifier import (
    notify_epoch,
    notify_crash_save,
    notify_training_complete,
    notify_error,
)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = {
    # IMPORTANT: check your actual extracted path
    # Common variants:
    #   "data/raw/urbansound8k/UrbanSound8K"  (soundata)
    #   "data/raw/urbansound8k"               (kaggle direct)
    "data_root": "data/raw/urbansound8k",
    "checkpoint": "checkpoints/acoustic_checkpoint.pth",

    "batch_size": 16,
    "num_workers": 0,     # Windows safe
    "num_epochs": 50,
    "lr": 1e-3,
    "lr_decay": 0.95,     # LR scheduler factor per epoch
    "latent_dim": 64,
    "n_mfcc": 40,
    "time_frames": 128,

    # Beta-VAE: anneal from 0→1 over first 10 epochs for better disentanglement
    "beta_start": 0.0,
    "beta_end": 1.0,
    "beta_anneal_epochs": 10,
}

DEVICE = torch.device("cpu")


def get_beta(epoch: int) -> float:
    """Linear KL annealing schedule."""
    if epoch >= CFG["beta_anneal_epochs"]:
        return CFG["beta_end"]
    frac = epoch / max(1, CFG["beta_anneal_epochs"])
    return CFG["beta_start"] + frac * (CFG["beta_end"] - CFG["beta_start"])


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: AcousticVAE, optimizer):
    if not os.path.exists(path):
        print(f"[INFO] No acoustic checkpoint at {path}. Starting fresh.")
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    epoch = int(ckpt.get("epoch", 0))
    print(f"[INFO] Resumed acoustic training from epoch {epoch} | avg_loss={ckpt.get('avg_loss', '?')}")
    return epoch


def train():
    # ── Data ────────────────────────────────────────────────────────────────
    print("[DATA] Loading UrbanSound8K...")

    # Try both common folder structures
    data_root = CFG["data_root"]
    if not os.path.exists(os.path.join(data_root, "metadata")):
        alt = os.path.join(data_root, "UrbanSound8K")
        if os.path.exists(os.path.join(alt, "metadata")):
            data_root = alt
            print(f"[DATA] Found UrbanSound8K at {data_root}")
        else:
            print(f"[WARN] Cannot find UrbanSound8K metadata at {data_root}")
            print("       Place UrbanSound8K.csv in: <data_root>/metadata/UrbanSound8K.csv")

    dataset = UrbanSound8KDataset(
        root=data_root,
        folds=list(range(1, 10)),   # folds 1–9 for training
        n_mfcc=CFG["n_mfcc"],
        time_frames=CFG["time_frames"],
    )
    val_dataset = UrbanSound8KDataset(
        root=data_root,
        folds=[10],                  # fold 10 for validation
        n_mfcc=CFG["n_mfcc"],
        time_frames=CFG["time_frames"],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
    )

    print(f"[DATA] Train: {len(dataset)} | Val: {len(val_dataset)} | "
          f"Batches/epoch: {len(train_loader)}")

    # ── Model & Optimizer ───────────────────────────────────────────────────
    model = AcousticVAE(
        mfcc_bins=CFG["n_mfcc"],
        time_frames=CFG["time_frames"],
        latent_dim=CFG["latent_dim"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=CFG["lr_decay"])

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(CFG["checkpoint"], model, optimizer)

    # ── Training Loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        model.train()
        beta = get_beta(epoch)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Acoustic Epoch [{epoch+1}/{CFG['num_epochs']}] β={beta:.2f}",
            unit="batch",
        )

        for x, label in pbar:
            x = x.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            recon, mu, lv = model(x)
            loss = AcousticVAE.loss(recon, x, mu, lv, beta=beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Compute components for display (no grad needed)
            with torch.no_grad():
                recon_l = torch.nn.functional.mse_loss(recon, x, reduction="sum").item()
                kl_l    = (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())).item()

            total_loss  += float(loss.item())
            total_recon += recon_l
            total_kl    += kl_l

            pbar.set_postfix(
                loss=f"{loss.item():.1f}",
                recon=f"{recon_l:.1f}",
                kl=f"{kl_l:.1f}",
            )

        scheduler.step()

        avg_loss  = total_loss  / max(1, len(train_loader))
        avg_recon = total_recon / max(1, len(train_loader))
        avg_kl    = total_kl    / max(1, len(train_loader))

        # ── Validation loss ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                recon, mu, lv = model(x)
                val_loss += AcousticVAE.loss(recon, x, mu, lv, beta=1.0).item()
        avg_val = val_loss / max(1, len(val_loader))

        # ── Checkpoint ───────────────────────────────────────────────────────
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt":   optimizer.state_dict(),
            "avg_loss": avg_loss,
            "avg_val_loss": avg_val,
            "beta": beta,
            "latent_dim": CFG["latent_dim"],
            "n_mfcc": CFG["n_mfcc"],
            "time_frames": CFG["time_frames"],
        }
        save_checkpoint(state, CFG["checkpoint"])
        notify_crash_save(epoch + 1, CFG["checkpoint"])

        # ── Notifications & logs ─────────────────────────────────────────────
        notify_epoch(epoch + 1, CFG["num_epochs"], d_loss=avg_recon / 1e4, g_loss=avg_loss)
        print(
            f"[ACOUSTIC] Epoch {epoch+1}/{CFG['num_epochs']} | "
            f"Train={avg_loss:.2f} | Val={avg_val:.2f} | "
            f"Recon={avg_recon:.2f} | KL={avg_kl:.2f} | β={beta:.2f}"
        )

    notify_training_complete(CFG["num_epochs"], avg_loss)
    print("[DONE] Acoustic training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
