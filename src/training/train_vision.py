"""
Urban-GenX | Vision Node Training (DP-GAN safe loop)

Fixes:
- Opacus Poisson sampling + GAN: avoid grad accumulation by stepping D each time
- Prevents GradSampleModule "activations.pop() empty" by disabling Opacus hooks during G step
  (we only need DP hooks during D update)

Refs:
- Opacus advanced features / Poisson sampling constraints:
  https://opacus.ai/tutorials/intro_to_advanced_features
- Temporarily disable hooks:
  https://discuss.pytorch.org/t/opacus-how-to-disable-backward-hook-temporally-for-multiple-backward-pass/141607
- GradSampleModule API:
  https://opacus.ai/api/grad_sample_module.html
"""

import os
import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# Project-specific imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.vision_gan import Generator, Discriminator
from src.utils.data_loader import CityscapesDataset
from src.utils.notifier import (
    notify_epoch,
    notify_crash_save,
    notify_training_complete,
    notify_error,
)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
CFG = {
    "data_root": "data/raw/cityscapes",
    "checkpoint": "checkpoints/vision_checkpoint.pth",
    "img_size": 64,

    "batch_size": 4,      # CPU / 12GB-safe
    "num_workers": 0,

    "num_epochs": 50,
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "betas": (0.5, 0.999),
    "noise_dim": 100,
    "num_classes": 35,

    # IMPORTANT for “release G weights under DP”:
    # If dp_enabled=True and you will publish G weights, set lambda_l1=0.0
    # Otherwise G directly learns from private images via L1.
    "lambda_l1": 0.0,

    "dp_enabled": True,
    "max_grad_norm": 1.0,
    "target_epsilon": 10.0,
    "target_delta": 1e-5,
    "secure_mode": False,  # True for final run (slower)
}

DEVICE = torch.device("cpu")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint_if_exists(path: str):
    if not os.path.exists(path):
        print(f"[INFO] No checkpoint at {path}, starting fresh.")
        return None
    ckpt = torch.load(path, map_location="cpu")
    print(f"[INFO] Found checkpoint: epoch={ckpt.get('epoch', '?')}")
    return ckpt


def attach_dp_to_discriminator(D, opt_d, loader):
    privacy_engine = PrivacyEngine(secure_mode=CFG["secure_mode"])
    D, opt_d, loader = privacy_engine.make_private_with_epsilon(
        module=D,
        optimizer=opt_d,
        data_loader=loader,
        target_epsilon=CFG["target_epsilon"],
        target_delta=CFG["target_delta"],
        max_grad_norm=CFG["max_grad_norm"],
        epochs=CFG["num_epochs"],
    )
    print(f"[DP] Attached to D | target ε={CFG['target_epsilon']} δ={CFG['target_delta']}")
    return privacy_engine, D, opt_d, loader


# ─── Main Training ────────────────────────────────────────────────────────────
def train():
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
    print(f"[DATA] {len(dataset)} samples | ~{len(loader)} batches/epoch (pre-DP)")

    # ── Models ───────────────────────────────────────────────────────────────
    G = Generator(CFG["noise_dim"], CFG["num_classes"]).to(DEVICE)
    D = Discriminator(CFG["num_classes"]).to(DEVICE)

    if CFG["dp_enabled"]:
        G = ModuleValidator.fix(G)
        D = ModuleValidator.fix(D)

    # ── Optimizers ───────────────────────────────────────────────────────────
    opt_g = optim.Adam(G.parameters(), lr=CFG["lr_g"], betas=CFG["betas"])
    opt_d = optim.Adam(D.parameters(), lr=CFG["lr_d"], betas=CFG["betas"])

    # ── Resume ───────────────────────────────────────────────────────────────
    ckpt = load_checkpoint_if_exists(CFG["checkpoint"])
    start_epoch = int(ckpt["epoch"]) if ckpt and "epoch" in ckpt else 0

    # ── Attach DP ────────────────────────────────────────────────────────────
    privacy_engine = None
    if CFG["dp_enabled"]:
        privacy_engine, D, opt_d, loader = attach_dp_to_discriminator(D, opt_d, loader)

    # Load weights AFTER DP wrapping
    if ckpt is not None:
        G.load_state_dict(ckpt["generator"])
        D.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])

        # Restore accountant state if present
        if privacy_engine is not None and "dp_accountant" in ckpt:
            try:
                privacy_engine.accountant.load_state_dict(ckpt["dp_accountant"])
                print("[DP] Restored accountant state.")
            except Exception as e:
                print(f"[DP] Accountant restore skipped: {e}")

        print(f"[INFO] Resumed from epoch {start_epoch}")

    # ── Losses ───────────────────────────────────────────────────────────────
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        G.train()
        D.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        seen_batches = 0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{CFG['num_epochs']}]", unit="batch")

        for batch_idx, (real_img, cond) in enumerate(pbar):
            real_img = real_img.to(DEVICE)
            cond = cond.to(DEVICE)
            seen_batches += 1

            # ============================================================
            # (1) Train D (DP)
            # ============================================================
            # Ensure hooks are ON for D step (needed for DP accounting)
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()

            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)

            fake_img = G(cond).detach()
            real_pred = D(cond, real_img)
            fake_pred = D(cond, fake_img)

            d_loss = 0.5 * (
                criterion_gan(real_pred, torch.ones_like(real_pred)) +
                criterion_gan(fake_pred, torch.zeros_like(fake_pred))
            )

            d_loss.backward()
            opt_d.step()  # must step immediately under Poisson sampling

            # ============================================================
            # (2) Train G (NO DP hooks on D)
            # ============================================================
            # Critical: disable Opacus hooks during G step to avoid
            # GradSampleModule activation-stack mismatch (activations.pop empty).
            if hasattr(D, "disable_hooks"):
                D.disable_hooks()

            set_requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)

            fake_img = G(cond)
            fake_pred = D(cond, fake_img)

            g_loss_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            g_loss_l1 = criterion_l1(fake_img, real_img) * float(CFG["lambda_l1"])
            g_loss = g_loss_gan + g_loss_l1

            g_loss.backward()
            opt_g.step()

            # Re-enable hooks for next iteration’s D step (safe)
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()

            # Logging
            epoch_d_loss += float(d_loss.item())
            epoch_g_loss += float(g_loss.item())

            pbar.set_postfix(D_Loss=f"{d_loss.item():.4f}", G_Loss=f"{g_loss.item():.4f}")

        avg_d = epoch_d_loss / max(1, seen_batches)
        avg_g = epoch_g_loss / max(1, seen_batches)

        # DP budget
        if privacy_engine is not None:
            eps = privacy_engine.get_epsilon(CFG["target_delta"])
            print(f"[DP] ε spent so far: {eps:.2f} (δ={CFG['target_delta']})")

        # Checkpoint every epoch
        state = {
            "epoch": epoch + 1,
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "g_loss": avg_g,
            "d_loss": avg_d,
        }
        if privacy_engine is not None:
            try:
                state["dp_accountant"] = privacy_engine.accountant.state_dict()
            except Exception:
                pass

        save_checkpoint(state, CFG["checkpoint"])
        notify_crash_save(epoch + 1, CFG["checkpoint"])
        notify_epoch(epoch + 1, CFG["num_epochs"], avg_d, avg_g)

        print(f"[EPOCH] {epoch+1}/{CFG['num_epochs']} | D={avg_d:.4f} | G={avg_g:.4f}")

    notify_training_complete(CFG["num_epochs"], avg_g)
    print("[DONE] Training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
