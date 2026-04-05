"""
Urban-GenX | Acoustic VAE Training WITH DP-SGD (Phase 2)
=========================================================
This script adds Opacus DP-SGD to the Acoustic VAE training.
For VAEs, DP is simpler than GANs — just wrap the single optimizer.

Run: python src/training/train_acoustic_dp.py
     (use this INSTEAD of train_acoustic.py for DP-protected training)

Target: ε ≤ 10.0, δ = 1e-5 (same as Vision)
"""

import os
import sys
import traceback
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.acoustic_vae import AcousticVAE
from src.utils.data_loader import UrbanSound8KDataset
from src.utils.notifier import (
    notify_epoch, notify_crash_save,
    notify_training_complete, notify_error,
)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = {
    "data_root":       "data/raw/urbansound8k",
    "checkpoint":      "checkpoints/acoustic_dp_checkpoint.pth",
    "batch_size":      16,
    "num_workers":     0,
    "num_epochs":      50,
    "lr":              5e-4,
    "latent_dim":      64,
    "n_mfcc":          40,
    "time_frames":     128,
    "beta_start":      0.0,
    "beta_end":        1.0,
    "beta_anneal":     10,
    # DP parameters
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


def resolve_data_root(root):
    if os.path.exists(os.path.join(root, "metadata", "UrbanSound8K.csv")):
        return root
    alt = os.path.join(root, "UrbanSound8K")
    if os.path.exists(os.path.join(alt, "metadata", "UrbanSound8K.csv")):
        return alt
    return root


def train():
    print("=" * 60)
    print("  Urban-GenX | Acoustic VAE Training WITH DP-SGD")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────
    data_root = resolve_data_root(CFG["data_root"])
    train_ds = UrbanSound8KDataset(root=data_root, folds=list(range(1, 10)),
                                   n_mfcc=CFG["n_mfcc"], time_frames=CFG["time_frames"])
    loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                        num_workers=CFG["num_workers"], drop_last=False)
    print(f"[DATA] {len(train_ds)} training clips | {len(loader)} batches/epoch")

    # ── Model ───────────────────────────────────────────────────────
    model = AcousticVAE(mfcc_bins=CFG["n_mfcc"], time_frames=CFG["time_frames"],
                        latent_dim=CFG["latent_dim"]).to(DEVICE)

    # Opacus compatibility: fix BatchNorm → GroupNorm
    if CFG["dp_enabled"]:
        model = ModuleValidator.fix(model)

    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])

    # ── Attach DP ───────────────────────────────────────────────────
    privacy_engine = None
    if CFG["dp_enabled"]:
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=CFG["target_epsilon"],
            target_delta=CFG["target_delta"],
            max_grad_norm=CFG["max_grad_norm"],
            epochs=CFG["num_epochs"],
        )
        print(f"[DP] Attached | target ε={CFG['target_epsilon']} δ={CFG['target_delta']}")

    # ── Resume ──────────────────────────────────────────────────────
    start_epoch = 0
    if os.path.exists(CFG["checkpoint"]):
        try:
            ckpt = torch.load(CFG["checkpoint"], map_location="cpu")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["opt"])
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"[INFO] Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"[WARN] Checkpoint load failed: {e}. Starting fresh.")

    # ── Training Loop ───────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        model.train()
        beta = get_beta(epoch)
        total_loss = 0.0
        seen = 0

        pbar = tqdm(loader, desc=f"AcousticDP [{epoch+1}/{CFG['num_epochs']}] β={beta:.2f}",
                    unit="batch")
        for x, _ in pbar:
            if x.size(0) == 0:
                continue
            x = x.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            recon, mu, lv = model(x)
            loss = AcousticVAE.loss(recon, x, mu, lv, beta=beta)
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

        # Checkpoint
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "avg_loss": avg_loss,
        }
        os.makedirs(os.path.dirname(CFG["checkpoint"]), exist_ok=True)
        torch.save(state, CFG["checkpoint"])

        notify_epoch(epoch + 1, CFG["num_epochs"], d_loss=0.0, g_loss=avg_loss)
        print(f"[ACOUSTIC-DP] {epoch+1}/{CFG['num_epochs']} | loss={avg_loss:.4f}")

    notify_training_complete(CFG["num_epochs"], avg_loss)
    print("[DONE] Acoustic DP training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
