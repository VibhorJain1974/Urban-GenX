"""
Urban-GenX | Utility VAE Training WITH DP-SGD (Phase 2)
========================================================
Adds Opacus DP-SGD to Traffic and Water VAE training.

Run:
  python src/training/train_utility_dp.py --mode traffic
  python src/training/train_utility_dp.py --mode water

Target: ε ≤ 10.0, δ = 1e-5
"""

import os
import sys
import argparse
import traceback
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.utility_vae import build_traffic_vae, build_water_vae, UtilityVAE
from src.utils.data_loader import METRLADataset, WaterQualityDataset
from src.utils.notifier import (
    notify_epoch, notify_crash_save,
    notify_training_complete, notify_error,
)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = {
    "mode":            "traffic",
    "traffic_h5":      "data/raw/metr-la/metr-la.h5",
    "water_csv":       "data/raw/usgs_water/water_quality.csv",
    "batch_size":      32,
    "num_workers":     0,
    "num_epochs":      50,
    "lr":              1e-3,
    "latent_dim":      64,
    "beta_start":      0.0,
    "beta_end":        1.0,
    "beta_anneal":     10,
    "val_split":       0.1,
    # DP
    "dp_enabled":      True,
    "target_epsilon":  10.0,
    "target_delta":    1e-5,
    "max_grad_norm":   1.0,
}

DEVICE = torch.device("cpu")


def get_beta(epoch):
    if epoch >= CFG["beta_anneal"]:
        return CFG["beta_end"]
    return CFG["beta_start"] + (epoch / CFG["beta_anneal"]) * (CFG["beta_end"] - CFG["beta_start"])


def train():
    mode = CFG["mode"]
    ckpt_path = f"checkpoints/utility_{mode}_dp_checkpoint.pth"
    print("=" * 60)
    print(f"  Urban-GenX | Utility VAE [{mode}] Training WITH DP-SGD")
    print("=" * 60)

    # ── Data & Model ────────────────────────────────────────────────
    if mode == "traffic":
        full_ds = METRLADataset(CFG["traffic_h5"], seq_len=12, pred_len=12)
        model = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=CFG["latent_dim"]).to(DEVICE)
        def flatten_fn(x):
            return x.view(x.size(0), -1)
    elif mode == "water":
        full_ds = WaterQualityDataset(csv_path=CFG["water_csv"], seq_len=24, n_params=5, stride=1)
        actual_n = full_ds.actual_n_params
        model = build_water_vae(seq_len=24, n_params=actual_n,
                                latent_dim=min(16, CFG["latent_dim"])).to(DEVICE)
        def flatten_fn(x):
            return x.view(x.size(0), -1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Opacus fix
    if CFG["dp_enabled"]:
        model = ModuleValidator.fix(model)

    # Split
    val_size = max(1, int(len(full_ds) * CFG["val_split"]))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"])
    print(f"[DATA] Train={train_size} | Val={val_size} | Batches={len(train_loader)}")

    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])

    # ── Attach DP ───────────────────────────────────────────────────
    privacy_engine = None
    if CFG["dp_enabled"]:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=CFG["target_epsilon"],
            target_delta=CFG["target_delta"],
            max_grad_norm=CFG["max_grad_norm"],
            epochs=CFG["num_epochs"],
        )
        print(f"[DP] Attached | target ε={CFG['target_epsilon']} δ={CFG['target_delta']}")

    # ── Resume ──────────────────────────────────────────────────────
    start_epoch = 0
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["opt"])
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"[INFO] Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"[WARN] Checkpoint load failed: {e}. Starting fresh.")

    # ── Training Loop ───────────────────────────────────────────────
    avg_loss = 0.0
    for epoch in range(start_epoch, CFG["num_epochs"]):
        model.train()
        beta = get_beta(epoch)
        total_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"UtilityDP[{mode}] [{epoch+1}/{CFG['num_epochs']}]",
                    unit="batch")
        for x, _ in pbar:
            if x.size(0) == 0:
                continue
            x_flat = flatten_fn(x).to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, lv = model(x_flat)
            loss = UtilityVAE.loss(recon, x_flat, mu, lv, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            seen += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", beta=f"{beta:.2f}")

        avg_loss = total_loss / max(1, seen)

        # DP budget
        if privacy_engine:
            eps = privacy_engine.get_epsilon(CFG["target_delta"])
            print(f"[DP] ε spent: {eps:.2f}")

        # Validation (without DP hooks)
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x_flat = flatten_fn(x).to(DEVICE)
                recon, mu, lv = model(x_flat)
                val_loss += UtilityVAE.loss(recon, x_flat, mu, lv, beta=1.0).item()
                val_n += 1
        avg_val = val_loss / max(1, val_n)

        # Save
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "avg_val": avg_val,
            "mode": mode,
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(state, ckpt_path)

        notify_epoch(epoch + 1, CFG["num_epochs"], d_loss=avg_val, g_loss=avg_loss)
        print(f"[UTILITY-DP] {epoch+1}/{CFG['num_epochs']} | Train={avg_loss:.4f} | Val={avg_val:.4f}")

    notify_training_complete(CFG["num_epochs"], avg_loss)
    print(f"[DONE] Utility[{mode}] DP training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["traffic", "water"], default="traffic")
    args = parser.parse_args()
    CFG["mode"] = args.mode

    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
