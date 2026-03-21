"""
Urban-GenX | Vision Node Training (FIXED for resume + stable DP)
Fixes:
  1. PRV accountant → RDP accountant (stable, no numerical crashes on resume)
  2. epochs = remaining epochs on resume (prevents budget miscalculation)
  3. load_state_dict AFTER DP wrapping (correct order)
  4. GAN-safe Opacus: disable_hooks() during G step
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
    "data_root":      "data/raw/cityscapes",
    "checkpoint":     "checkpoints/vision_checkpoint.pth",
    "img_size":       64,
    "batch_size":     4,          # CPU / 12GB-safe
    "num_workers":    0,          # Windows: must be 0
    "num_epochs":     50,
    "lr_g":           2e-4,
    "lr_d":           2e-4,
    "betas":          (0.5, 0.999),
    "noise_dim":      100,
    "num_classes":    35,
    # DP: lambda_l1=0 → formally safe to release G weights
    "lambda_l1":      0.0,
    "dp_enabled":     True,
    "max_grad_norm":  1.0,
    "target_epsilon": 10.0,
    "target_delta":   1e-5,
    "secure_mode":    False,      # set True for final official run
    # FIX: use RDP accountant (stable on Windows CPU, no PRV numerical crashes)
    "accountant":     "rdp",
}

DEVICE = torch.device("cpu")


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint_if_exists(path):
    if not os.path.exists(path):
        print(f"[INFO] No checkpoint at {path}, starting fresh.")
        return None
    ckpt = torch.load(path, map_location="cpu")
    print(f"[INFO] Found checkpoint: epoch={ckpt.get('epoch', '?')}")
    return ckpt


def attach_dp_to_discriminator(D, opt_d, loader, remaining_epochs):
    """
    FIX: pass remaining_epochs (not total) to avoid PRV budget miscalculation on resume.
    FIX: use accountant='rdp' (stable) instead of default 'prv' (numerically unstable on CPU).
    """
    privacy_engine = PrivacyEngine(
        secure_mode=CFG["secure_mode"],
        accountant=CFG["accountant"],   # 'rdp' is stable and well-tested
    )
    D, opt_d, loader = privacy_engine.make_private_with_epsilon(
        module=D,
        optimizer=opt_d,
        data_loader=loader,
        target_epsilon=CFG["target_epsilon"],
        target_delta=CFG["target_delta"],
        max_grad_norm=CFG["max_grad_norm"],
        epochs=remaining_epochs,          # FIX: only remaining epochs
    )
    print(f"[DP] Attached | target ε={CFG['target_epsilon']} δ={CFG['target_delta']} | remaining_epochs={remaining_epochs} | accountant=rdp")
    return privacy_engine, D, opt_d, loader


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
    print(f"[DATA] {len(dataset)} samples | ~{len(loader)} batches/epoch")

    # ── Models ───────────────────────────────────────────────────────────────
    G = Generator(CFG["noise_dim"], CFG["num_classes"]).to(DEVICE)
    D = Discriminator(CFG["num_classes"]).to(DEVICE)

    if CFG["dp_enabled"]:
        G = ModuleValidator.fix(G)
        D = ModuleValidator.fix(D)

    # ── Optimizers ───────────────────────────────────────────────────────────
    opt_g = optim.Adam(G.parameters(), lr=CFG["lr_g"], betas=CFG["betas"])
    opt_d = optim.Adam(D.parameters(), lr=CFG["lr_d"], betas=CFG["betas"])

    # ── Read start_epoch from checkpoint BEFORE DP attachment ────────────────
    ckpt = load_checkpoint_if_exists(CFG["checkpoint"])
    start_epoch = int(ckpt["epoch"]) if ckpt and "epoch" in ckpt else 0

    # Compute remaining epochs (FIX: key change)
    remaining_epochs = max(1, CFG["num_epochs"] - start_epoch)

    # ── Attach DP to D ───────────────────────────────────────────────────────
    privacy_engine = None
    if CFG["dp_enabled"]:
        if remaining_epochs == 0:
            print("[INFO] Training already complete (epoch >= num_epochs). Nothing to do.")
            return
        privacy_engine, D, opt_d, loader = attach_dp_to_discriminator(
            D, opt_d, loader, remaining_epochs
        )

    # ── Load weights AFTER DP wrapping ───────────────────────────────────────
    if ckpt is not None:
        G.load_state_dict(ckpt["generator"])
        D.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        print(f"[INFO] Weights loaded. Resuming from epoch {start_epoch}.")

    # ── Losses ───────────────────────────────────────────────────────────────
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        G.train()
        D.train()

        epoch_d_loss  = 0.0
        epoch_g_loss  = 0.0
        seen_batches  = 0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{CFG['num_epochs']}]", unit="batch")

        for batch_idx, (real_img, cond) in enumerate(pbar):
            real_img = real_img.to(DEVICE)
            cond     = cond.to(DEVICE)
            seen_batches += 1

            # ── (1) Train D (DP) ──────────────────────────────────────────
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()
            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)

            fake_img  = G(cond).detach()
            real_pred = D(cond, real_img)
            fake_pred = D(cond, fake_img)

            d_loss = 0.5 * (
                criterion_gan(real_pred, torch.ones_like(real_pred)) +
                criterion_gan(fake_pred, torch.zeros_like(fake_pred))
            )
            d_loss.backward()
            opt_d.step()  # step immediately (Poisson sampling rule)

            # ── (2) Train G (hooks OFF during G step) ────────────────────
            if hasattr(D, "disable_hooks"):
                D.disable_hooks()
            set_requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)

            fake_img  = G(cond)
            fake_pred = D(cond, fake_img)

            g_loss_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            g_loss_l1  = criterion_l1(fake_img, real_img) * float(CFG["lambda_l1"])
            g_loss     = g_loss_gan + g_loss_l1

            g_loss.backward()
            opt_g.step()

            # Re-enable hooks for next D step
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()

            epoch_d_loss += float(d_loss.item())
            epoch_g_loss += float(g_loss.item())
            pbar.set_postfix(D_Loss=f"{d_loss.item():.4f}", G_Loss=f"{g_loss.item():.4f}")

        avg_d = epoch_d_loss / max(1, seen_batches)
        avg_g = epoch_g_loss / max(1, seen_batches)

        # DP budget readout
        if privacy_engine is not None:
            eps = privacy_engine.get_epsilon(CFG["target_delta"])
            print(f"  [DP] ε spent: {eps:.4f} / {CFG['target_epsilon']}")

        # Checkpoint every epoch
        state = {
            "epoch":         epoch + 1,
            "generator":     G.state_dict(),
            "discriminator": D.state_dict(),
            "opt_g":         opt_g.state_dict(),
            "opt_d":         opt_d.state_dict(),
            "g_loss":        avg_g,
            "d_loss":        avg_d,
        }
        save_checkpoint(state, CFG["checkpoint"])
        notify_crash_save(epoch + 1, CFG["checkpoint"])
        notify_epoch(epoch + 1, CFG["num_epochs"], avg_d, avg_g)
        print(f"[EPOCH] {epoch+1}/{CFG['num_epochs']} | D={avg_d:.4f} | G={avg_g:.4f}")

    notify_training_complete(CFG["num_epochs"], avg_g)
    print("[DONE] Vision training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
