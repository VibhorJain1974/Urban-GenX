"""
Urban-GenX | Vision Node Training  (FINAL STABLE VERSION)
=========================================================
Key design decisions:
  1. NaN-TOLERANT: GANs produce occasional NaN in early batches.
     Instead of crashing, we SKIP bad batches and only abort if
     NaN persists for MAX_CONSECUTIVE_NAN batches in a row.
  2. Label smoothing: real=0.9, fake=0.1 to prevent D from becoming
     too confident (reduces log(0) risk in BCE).
  3. Generator gradient clipping (max_norm=1.0).
  4. Discriminator output clamping before loss computation.
  5. Only saves checkpoint when epoch losses are finite.
  6. Rejects corrupt checkpoints on resume.
  7. DP on Discriminator only; lambda_l1=0.0 for safe G release.
  8. Opacus hooks disabled during G step (prevents activation stack crash).
"""

import os
import sys
import traceback
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.vision_gan import Generator, Discriminator
from src.utils.data_loader import CityscapesDataset
from src.utils.notifier import (
    notify_epoch,
    notify_crash_save,
    notify_training_complete,
    notify_error,
)

# ─── Configuration ───────────────────────────────────────────────────────────
CFG = {
    "data_root": "data/raw/cityscapes",
    "checkpoint": "checkpoints/vision_checkpoint.pth",
    "best_checkpoint": "checkpoints/vision_best.pth",
    "img_size": 64,
    "batch_size": 4,
    "num_workers": 0,
    "num_epochs": 50,
    "lr_g": 1e-4,
    "lr_d": 1e-4,
    "betas": (0.5, 0.999),
    "noise_dim": 100,
    "num_classes": 35,
    # DP-safe: no L1 term means G never directly touches real images
    "lambda_l1": 0.0,
    # DP settings
    "dp_enabled": True,
    "max_grad_norm": 1.0,
    "target_epsilon": 10.0,
    "target_delta": 1e-5,
    "accountant": "rdp",
    "secure_mode": False,
    # NaN tolerance for GAN training
    "max_consecutive_nan": 20,
    # Label smoothing for D stability
    "real_label": 0.9,
    "fake_label": 0.1,
    # G gradient clipping
    "max_grad_norm_g": 1.0,
    # Seed
    "seed": 42,
}

DEVICE = torch.device("cpu")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def is_finite(x) -> bool:
    """Check if a scalar value is finite."""
    try:
        return math.isfinite(float(x))
    except (ValueError, TypeError):
        return False


def validate_checkpoint(ckpt: dict) -> bool:
    """Return True if checkpoint has finite losses and no NaN weights."""
    g_loss = ckpt.get("g_loss", None)
    d_loss = ckpt.get("d_loss", None)
    if g_loss is not None and not is_finite(g_loss):
        return False
    if d_loss is not None and not is_finite(d_loss):
        return False
    for key in ("generator", "discriminator"):
        if key in ckpt:
            for name, param in ckpt[key].items():
                if torch.is_tensor(param) and param.is_floating_point():
                    if not torch.isfinite(param).all():
                        return False
    return True


def save_checkpoint(state: dict, path: str):
    """Save checkpoint only if losses are finite."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ─── Main Training ──────────────────────────────────────────────────────────
def train():
    torch.manual_seed(CFG["seed"])

    # ── Data ────────────────────────────────────────────────────────────────
    print("[DATA] Loading Cityscapes...")
    dataset = CityscapesDataset(CFG["data_root"], split="train", img_size=CFG["img_size"])
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=False,
        drop_last=False,
    )
    print(f"[DATA] {len(dataset)} samples | ~{len(loader)} batches/epoch")

    # ── Models ──────────────────────────────────────────────────────────────
    G = Generator(CFG["noise_dim"], CFG["num_classes"]).to(DEVICE)
    D = Discriminator(CFG["num_classes"]).to(DEVICE)

    if CFG["dp_enabled"]:
        G = ModuleValidator.fix(G)
        D = ModuleValidator.fix(D)

    # ── Optimizers ──────────────────────────────────────────────────────────
    opt_g = optim.Adam(G.parameters(), lr=CFG["lr_g"], betas=CFG["betas"])
    opt_d = optim.Adam(D.parameters(), lr=CFG["lr_d"], betas=CFG["betas"])

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_g_loss = float("inf")

    ckpt_path = CFG["checkpoint"]
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if validate_checkpoint(ckpt):
            start_epoch = int(ckpt.get("epoch", 0))
            best_g_loss = float(ckpt.get("best_g_loss", float("inf")))
            print(f"[INFO] Valid checkpoint found at epoch {start_epoch}, will load after DP setup.")
        else:
            print(f"[WARN] Checkpoint has NaN values. Renaming and starting fresh.")
            os.rename(ckpt_path, ckpt_path + ".corrupt")
            ckpt = None
    else:
        ckpt = None
        print(f"[INFO] No checkpoint found. Starting fresh.")

    # ── Attach DP ───────────────────────────────────────────────────────────
    privacy_engine = None
    remaining_epochs = max(1, CFG["num_epochs"] - start_epoch)

    if CFG["dp_enabled"]:
        privacy_engine = PrivacyEngine(secure_mode=CFG["secure_mode"])
        D, opt_d, loader = privacy_engine.make_private_with_epsilon(
            module=D,
            optimizer=opt_d,
            data_loader=loader,
            target_epsilon=CFG["target_epsilon"],
            target_delta=CFG["target_delta"],
            max_grad_norm=CFG["max_grad_norm"],
            epochs=remaining_epochs,
            accountant=CFG["accountant"],
        )
        print(f"[DP] Attached | ε={CFG['target_epsilon']} δ={CFG['target_delta']} "
              f"remaining_epochs={remaining_epochs} accountant={CFG['accountant']}")

    # ── Load weights AFTER DP wrapping ──────────────────────────────────────
    if ckpt is not None and validate_checkpoint(ckpt):
        G.load_state_dict(ckpt["generator"])
        D.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        print(f"[INFO] Loaded weights from epoch {start_epoch}")

    # ── Losses ──────────────────────────────────────────────────────────────
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    # ── Training Loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        G.train()
        D.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        valid_batches = 0
        consecutive_nan = 0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{CFG['num_epochs']}]", unit="batch")

        for batch_idx, (real_img, cond) in enumerate(pbar):
            real_img = real_img.to(DEVICE)
            cond = cond.to(DEVICE)

            # ════════════════════════════════════════════════════════════
            # (1) Train Discriminator (DP-protected)
            # ════════════════════════════════════════════════════════════
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()
            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)

            fake_img = G(cond).detach()

            real_pred = D(cond, real_img)
            fake_pred = D(cond, fake_img)

            # Clamp predictions to prevent extreme values
            real_pred = torch.clamp(real_pred, -10.0, 10.0)
            fake_pred = torch.clamp(fake_pred, -10.0, 10.0)

            # Label smoothing for stability
            real_labels = torch.full_like(real_pred, CFG["real_label"])
            fake_labels = torch.full_like(fake_pred, CFG["fake_label"])

            d_loss_real = criterion_gan(real_pred, real_labels)
            d_loss_fake = criterion_gan(fake_pred, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # NaN check — skip batch if d_loss is bad
            if not torch.isfinite(d_loss):
                consecutive_nan += 1
                if consecutive_nan >= CFG["max_consecutive_nan"]:
                    raise RuntimeError(
                        f"Training unstable: {consecutive_nan} consecutive NaN batches. "
                        f"Try reducing lr_d/lr_g or check your data."
                    )
                pbar.set_postfix(D_Loss="NaN(skip)", G_Loss="---")
                continue

            d_loss.backward()
            opt_d.step()

            # ════════════════════════════════════════════════════════════
            # (2) Train Generator (no DP hooks)
            # ════════════════════════════════════════════════════════════
            if hasattr(D, "disable_hooks"):
                D.disable_hooks()
            set_requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)

            fake_img = G(cond)
            fake_pred = D(cond, fake_img)
            fake_pred = torch.clamp(fake_pred, -10.0, 10.0)

            g_loss_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            g_loss_l1 = criterion_l1(fake_img, real_img) * float(CFG["lambda_l1"])
            g_loss = g_loss_gan + g_loss_l1

            # NaN check for G
            if not torch.isfinite(g_loss):
                consecutive_nan += 1
                if consecutive_nan >= CFG["max_consecutive_nan"]:
                    raise RuntimeError(
                        f"Training unstable: {consecutive_nan} consecutive NaN batches in G."
                    )
                # Re-enable hooks for next D step
                if hasattr(D, "enable_hooks"):
                    D.enable_hooks()
                pbar.set_postfix(D_Loss=f"{d_loss.item():.4f}", G_Loss="NaN(skip)")
                continue

            g_loss.backward()
            # Clip G gradients
            torch.nn.utils.clip_grad_norm_(G.parameters(), CFG["max_grad_norm_g"])
            opt_g.step()

            # Re-enable hooks for next iteration
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()

            # If we get here, batch was successful
            consecutive_nan = 0
            epoch_d_loss += float(d_loss.item())
            epoch_g_loss += float(g_loss.item())
            valid_batches += 1

            pbar.set_postfix(
                D_Loss=f"{d_loss.item():.4f}",
                G_Loss=f"{g_loss.item():.4f}",
            )

        # ── Epoch Summary ───────────────────────────────────────────────
        if valid_batches == 0:
            print(f"[WARN] Epoch {epoch+1}: zero valid batches! Skipping checkpoint save.")
            continue

        avg_d = epoch_d_loss / valid_batches
        avg_g = epoch_g_loss / valid_batches

        # Only save if finite
        if is_finite(avg_d) and is_finite(avg_g):
            state = {
                "epoch": epoch + 1,
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "g_loss": avg_g,
                "d_loss": avg_d,
                "best_g_loss": min(best_g_loss, avg_g),
            }
            save_checkpoint(state, CFG["checkpoint"])
            notify_crash_save(epoch + 1, CFG["checkpoint"])

            # Save best
            if avg_g < best_g_loss:
                best_g_loss = avg_g
                save_checkpoint(state, CFG["best_checkpoint"])
                print(f"  [BEST] New best G_Loss: {avg_g:.4f}")
        else:
            print(f"  [WARN] Epoch {epoch+1} had non-finite average loss. Checkpoint NOT saved.")

        # DP budget
        if privacy_engine is not None:
            eps = privacy_engine.get_epsilon(CFG["target_delta"])
            print(f"  [DP] ε spent: {eps:.2f} / {CFG['target_epsilon']}")

        notify_epoch(epoch + 1, CFG["num_epochs"], avg_d, avg_g)
        print(f"  [EPOCH] {epoch+1}/{CFG['num_epochs']} | D={avg_d:.4f} | G={avg_g:.4f} "
              f"| valid_batches={valid_batches}")

    notify_training_complete(CFG["num_epochs"], avg_g if valid_batches > 0 else 0.0)
    print("[DONE] Vision training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
