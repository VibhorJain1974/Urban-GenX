"""
Urban-GenX | Utility Node Training
Trains UtilityVAE on METR-LA traffic data (and optionally USGS water).
With checkpointing, tqdm, ntfy.
"""

import os
import sys
import traceback
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.utility_vae import build_traffic_vae, build_water_vae, UtilityVAE
from src.utils.data_loader import METRLADataset
from src.utils.notifier import (
    notify_epoch,
    notify_crash_save,
    notify_training_complete,
    notify_error,
)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = {
    "mode": "traffic",   # "traffic" | "water"

    # METR-LA
    "traffic_h5":  "data/raw/metr-la/metr-la.h5",
    "seq_len":     12,
    "pred_len":    12,
    "n_sensors":   207,

    "checkpoint":  "checkpoints/utility_traffic_checkpoint.pth",
    "batch_size":  32,
    "num_workers": 0,
    "num_epochs":  50,
    "lr":          1e-3,
    "lr_decay":    0.97,
    "latent_dim":  64,
    "beta_start":  0.0,
    "beta_end":    1.0,
    "beta_anneal": 10,
    "val_split":   0.1,
}

DEVICE = torch.device("cpu")


def get_beta(epoch: int) -> float:
    if epoch >= CFG["beta_anneal"]:
        return CFG["beta_end"]
    return CFG["beta_start"] + (epoch / CFG["beta_anneal"]) * (CFG["beta_end"] - CFG["beta_start"])


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, opt):
    if not os.path.exists(path):
        print(f"[INFO] No utility checkpoint found at {path}. Starting fresh.")
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    epoch = int(ckpt.get("epoch", 0))
    print(f"[INFO] Resumed utility training from epoch {epoch}")
    return epoch


def train():
    # ── Data ────────────────────────────────────────────────────────────────
    if CFG["mode"] == "traffic":
        print("[DATA] Loading METR-LA traffic dataset...")
        h5_path = CFG["traffic_h5"]
        if not os.path.exists(h5_path):
            # Try common alternative paths
            for alt in [
                "data/raw/metr-la/metr-la.h5",
                "data/raw/metr-la/METR-LA.h5",
                "data/raw/metr-la/data/metr-la.h5",
            ]:
                if os.path.exists(alt):
                    h5_path = alt
                    print(f"[DATA] Found METR-LA at: {h5_path}")
                    break
            else:
                raise FileNotFoundError(
                    f"METR-LA h5 not found at {CFG['traffic_h5']}.\n"
                    "Download: kaggle datasets download -d annnnguyen/metr-la-dataset "
                    "-p data/raw/metr-la --unzip"
                )

        full_dataset = METRLADataset(
            h5_path, seq_len=CFG["seq_len"], pred_len=CFG["pred_len"]
        )
        input_dim = CFG["seq_len"] * CFG["n_sensors"]
        model = build_traffic_vae(
            seq_len=CFG["seq_len"],
            n_sensors=CFG["n_sensors"],
            latent_dim=CFG["latent_dim"],
        ).to(DEVICE)

    else:
        raise NotImplementedError(
            "Water mode: ensure USGS CSV is at data/raw/usgs_water/water_quality.csv"
        )

    # Train/Val split
    val_size = max(1, int(len(full_dataset) * CFG["val_split"]))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"],
    )
    print(f"[DATA] Train={train_size} | Val={val_size} | Batches/epoch={len(train_loader)}")

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=CFG["lr_decay"])

    start_epoch = load_checkpoint(CFG["checkpoint"], model, optimizer)

    # ── Training Loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        model.train()
        beta = get_beta(epoch)
        total_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Utility Epoch [{epoch+1}/{CFG['num_epochs']}] β={beta:.2f}",
            unit="batch",
        )

        for x, _ in pbar:
            # x: [B, seq_len, n_sensors] → flatten for VAE
            x_flat = x.view(x.size(0), -1).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            recon, mu, lv = model(x_flat)
            loss = UtilityVAE.loss(recon, x_flat, mu, lv, beta=beta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", beta=f"{beta:.2f}")

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x_flat = x.view(x.size(0), -1).to(DEVICE)
                recon, mu, lv = model(x_flat)
                val_loss += UtilityVAE.loss(recon, x_flat, mu, lv, beta=1.0).item()
        avg_val = val_loss / max(1, len(val_loader))

        # Checkpoint
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "avg_val": avg_val,
            "mode": CFG["mode"],
        }
        save_checkpoint(state, CFG["checkpoint"])
        notify_crash_save(epoch + 1, CFG["checkpoint"])
        notify_epoch(epoch + 1, CFG["num_epochs"], d_loss=avg_val, g_loss=avg_loss)

        print(
            f"[UTILITY] Epoch {epoch+1}/{CFG['num_epochs']} | "
            f"Train={avg_loss:.4f} | Val={avg_val:.4f} | Mode={CFG['mode']}"
        )

        # Only notify if we actually trained (loop executed)
    if start_epoch < CFG["num_epochs"]:
        notify_training_complete(CFG["num_epochs"], avg_loss)
        print(f"[DONE] Utility training complete. Final loss: {avg_loss:.4f}")
    else:
        print(f"[INFO] Training already complete at epoch {start_epoch}. No new training performed.")

if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
