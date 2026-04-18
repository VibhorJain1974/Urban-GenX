"""
Urban-GenX | Water Quality VAE Training — FIXED
================================================
Critical fix: saves actual n_params and seq_len in checkpoint so the
dashboard can auto-detect architecture and avoid size mismatch errors.

Run: python scripts/train_water.py
"""

import os
import sys
import traceback
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.utility_vae import UtilityVAE
from src.utils.data_loader import WaterQualityDataset
from src.utils.notifier import notify_epoch, notify_crash_save, notify_training_complete, notify_error

CFG = {
    "water_csv":   "data/raw/usgs_water/water_quality.csv",
    "checkpoint":  "checkpoints/utility_water_checkpoint.pth",
    "batch_size":  32,
    "num_workers": 0,
    "num_epochs":  50,
    "lr":          1e-3,
    "lr_decay":    0.97,
    "latent_dim":  16,
    "seq_len":     24,
    "n_params":    5,
    "stride":      1,
    "beta_anneal": 10,
    "val_split":   0.1,
}


def get_beta(epoch):
    if epoch >= CFG["beta_anneal"]:
        return 1.0
    return epoch / CFG["beta_anneal"]


def train():
    print("=" * 60)
    print("  Urban-GenX | Water Quality VAE Training")
    print("=" * 60)

    # ── Download water data if missing ──────────────────────────────────────
    if not os.path.exists(CFG["water_csv"]):
        print("[INFO] Water CSV not found. Downloading/generating...")
        try:
            from src.utils.download_water_data import main as download_water
            download_water()
        except Exception as e:
            print(f"[WARN] Download failed: {e}. Generating synthetic water data...")
            _generate_synthetic_water(CFG["water_csv"])

    # ── Dataset ─────────────────────────────────────────────────────────────
    ds = WaterQualityDataset(
        csv_path=CFG["water_csv"],
        seq_len=CFG["seq_len"],
        n_params=CFG["n_params"],
        stride=CFG["stride"],
    )
    actual_n = ds.actual_n_params
    print(f"[DATA] {len(ds)} samples | {actual_n} params × {CFG['seq_len']} steps")

    val_size   = max(1, int(len(ds) * CFG["val_split"]))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True, num_workers=CFG["num_workers"], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"])

    # ── Model ────────────────────────────────────────────────────────────────
    input_dim = CFG["seq_len"] * actual_n
    hidden    = min(64, max(32, input_dim // 2))
    model = UtilityVAE(
        input_dim=input_dim,
        latent_dim=CFG["latent_dim"],
        hidden_dims=[hidden * 2, hidden],
        name="water_usgs",
    )
    print(f"[MODEL] input_dim={input_dim} latent={CFG['latent_dim']} hidden={[hidden*2,hidden]}")

    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=CFG["lr_decay"])

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if os.path.exists(CFG["checkpoint"]):
        try:
            ckpt = torch.load(CFG["checkpoint"], map_location="cpu", weights_only=False)
            # Check if architecture matches
            saved_n = ckpt.get("actual_n_params", actual_n)
            if saved_n == actual_n:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["opt"])
                start_epoch = int(ckpt.get("epoch", 0))
                print(f"[INFO] Resumed from epoch {start_epoch}")
            else:
                print(f"[WARN] Architecture mismatch (saved: {saved_n}, now: {actual_n}). Starting fresh.")
        except Exception as e:
            print(f"[WARN] Checkpoint load failed: {e}. Starting fresh.")

    avg_loss = 0.0
    for epoch in range(start_epoch, CFG["num_epochs"]):
        model.train()
        beta = get_beta(epoch)
        total = 0.0

        pbar = tqdm(train_loader, desc=f"Water [{epoch+1}/{CFG['num_epochs']}] β={beta:.2f}")
        for x, _ in pbar:
            x_flat = x.view(x.size(0), -1)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, lv = model(x_flat)
            loss = UtilityVAE.loss(recon, x_flat, mu, lv, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total / max(1, len(train_loader))

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x_flat = x.view(x.size(0), -1)
                recon, mu, lv = model(x_flat)
                val_total += UtilityVAE.loss(recon, x_flat, mu, lv, beta=1.0).item()
        avg_val = val_total / max(1, len(val_loader))

        # Save checkpoint WITH architecture metadata
        state = {
            "epoch":          epoch + 1,
            "model":          model.state_dict(),
            "opt":            optimizer.state_dict(),
            "avg_loss":       avg_loss,
            "avg_val":        avg_val,
            "actual_n_params": actual_n,  # ← CRITICAL: save for dashboard auto-detection
            "seq_len":        CFG["seq_len"],
            "latent_dim":     CFG["latent_dim"],
            "input_dim":      input_dim,
        }
        os.makedirs(os.path.dirname(CFG["checkpoint"]), exist_ok=True)
        torch.save(state, CFG["checkpoint"])
        notify_crash_save(epoch + 1, CFG["checkpoint"])
        notify_epoch(epoch + 1, CFG["num_epochs"], avg_val, avg_loss)
        print(f"[WATER] {epoch+1}/{CFG['num_epochs']} | Train={avg_loss:.4f} | Val={avg_val:.4f}")

    notify_training_complete(CFG["num_epochs"], avg_loss)
    print(f"[DONE] Water VAE training complete.")


def _generate_synthetic_water(csv_path):
    """Fallback synthetic water data generator."""
    import numpy as np
    import pandas as pd
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

    np.random.seed(42)
    n = 5000
    t = np.arange(n)
    seasonal = np.sin(2 * np.pi * t / 365)

    df = pd.DataFrame({
        "datetime":             pd.date_range("2020-01-01", periods=n, freq="D"),
        "dissolved_oxygen_mgL": 9.0 + 2.5 * seasonal + np.random.normal(0, 0.8, n),
        "ph":                   7.4 + 0.3 * seasonal + np.random.normal(0, 0.2, n),
        "temperature_celsius":  15.0 + 8.0 * seasonal + np.random.normal(0, 1.5, n),
        "turbidity_FNU":        np.clip(np.random.lognormal(1.5, 0.8, n), 0.1, 200),
        "streamflow_cfs":       np.clip(np.random.lognormal(5.5, 0.8, n), 1, 50000),
    })
    df.to_csv(csv_path, index=False)
    print(f"[SYNTH] Generated synthetic water data → {csv_path}")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
