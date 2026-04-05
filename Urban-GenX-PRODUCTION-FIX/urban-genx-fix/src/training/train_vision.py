"""
Urban-GenX | Vision Node Training  (PRODUCTION FIX)
=====================================================
Fixes vs previous version:
  1. Opacus 'Poisson sampling grad accumulation' — disable_hooks() during G step
  2. Opacus 'activations.pop empty' — disable_hooks() during G step
  3. GAN NaN stability — label smoothing + logit clamping + NaN-skip
  4. Corrupted checkpoint resume — validate before loading
  5. lambda_l1=0.0 for DP-safe generator release (Option C)
  6. **CRITICAL BUG FIX**: Removed `opt_d.step()` call inside checkpoint save dict.
     Previous code had:
         "opt_d": opt_d.step() or opt_d.state_dict(),
     This caused an EXTRA optimizer step during save AND relied on `step()` returning
     None for the `or` fallback. Fixed to:
         "opt_d": opt_d.state_dict(),
"""

import os
import sys
import math
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

# ─── Config ───────────────────────────────────────────────────────────────────
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
    # IMPORTANT: lambda_l1=0.0 for formal DP guarantee on released G weights
    "lambda_l1": 0.0,
    # Differential Privacy
    "dp_enabled": True,
    "max_grad_norm": 1.0,
    "target_epsilon": 10.0,
    "target_delta": 1e-5,
    # GAN stability
    "real_label": 0.9,
    "fake_label": 0.1,
    "logit_clamp": 10.0,
    "g_grad_clip": 1.0,
    "max_consecutive_nan": 20,
}

DEVICE = torch.device("cpu")


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def is_finite(x):
    return math.isfinite(float(x))


def state_dict_has_nonfinite(sd):
    for v in sd.values():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            return True
    return False


def validate_checkpoint(path):
    """Validate checkpoint integrity before loading."""
    if not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception:
        return None
    g_loss = ck.get("g_loss")
    d_loss = ck.get("d_loss")
    if g_loss is not None and not is_finite(g_loss):
        print(f"[WARN] Checkpoint has NaN g_loss; ignoring.")
        os.rename(path, path + ".corrupt")
        return None
    if d_loss is not None and not is_finite(d_loss):
        print(f"[WARN] Checkpoint has NaN d_loss; ignoring.")
        os.rename(path, path + ".corrupt")
        return None
    for name in ("generator", "discriminator"):
        if name in ck and state_dict_has_nonfinite(ck[name]):
            print(f"[WARN] Checkpoint has NaN weights in {name}; ignoring.")
            os.rename(path, path + ".corrupt")
            return None
    print(f"[INFO] Valid checkpoint: epoch={ck.get('epoch', '?')}")
    return ck


def train():
    print("=" * 70)
    print("  Urban-GenX | Vision DP-GAN Training (Production)")
    print("=" * 70)
    print(f"  DP enabled:   {CFG['dp_enabled']}")
    print(f"  lambda_l1:    {CFG['lambda_l1']}")
    print(f"  label smooth: real={CFG['real_label']}, fake={CFG['fake_label']}")
    print(f"  logit clamp:  +/-{CFG['logit_clamp']}")
    print(f"  G grad clip:  {CFG['g_grad_clip']}")
    print("=" * 70)

    # ── Data ────────────────────────────────────────────────────────
    print("[DATA] Loading Cityscapes...")
    dataset = CityscapesDataset(CFG["data_root"], split="train", img_size=CFG["img_size"])
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=False,
    )
    print(f"[DATA] {len(dataset)} samples | ~{len(loader)} batches/epoch")

    # ── Models ──────────────────────────────────────────────────────
    G = Generator(CFG["noise_dim"], CFG["num_classes"]).to(DEVICE)
    D = Discriminator(CFG["num_classes"]).to(DEVICE)
    if CFG["dp_enabled"]:
        G = ModuleValidator.fix(G)
        D = ModuleValidator.fix(D)

    opt_g = optim.Adam(G.parameters(), lr=CFG["lr_g"], betas=CFG["betas"])
    opt_d = optim.Adam(D.parameters(), lr=CFG["lr_d"], betas=CFG["betas"])

    # ── DP ──────────────────────────────────────────────────────────
    privacy_engine = None
    ckpt = validate_checkpoint(CFG["checkpoint"])
    start_epoch = int(ckpt["epoch"]) if ckpt else 0
    remaining_epochs = max(1, CFG["num_epochs"] - start_epoch)

    if CFG["dp_enabled"]:
        privacy_engine = PrivacyEngine(accountant="rdp")
        D, opt_d, loader = privacy_engine.make_private_with_epsilon(
            module=D,
            optimizer=opt_d,
            data_loader=loader,
            target_epsilon=CFG["target_epsilon"],
            target_delta=CFG["target_delta"],
            max_grad_norm=CFG["max_grad_norm"],
            epochs=remaining_epochs,
        )
        print(
            f"[DP] Attached | ε={CFG['target_epsilon']} δ={CFG['target_delta']}"
            f" remaining_epochs={remaining_epochs}"
        )

    # ── Load weights AFTER DP wrapping ──────────────────────────────
    if ckpt is not None:
        G.load_state_dict(ckpt["generator"])
        D.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        print(f"[INFO] Resumed from epoch {start_epoch}")
    else:
        print("[INFO] No checkpoint found. Starting fresh.")

    criterion = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    best_g = float("inf")

    for epoch in range(start_epoch, CFG["num_epochs"]):
        G.train()
        D.train()
        epoch_d = 0.0
        epoch_g = 0.0
        seen = 0
        consecutive_nan = 0

        pbar = tqdm(
            loader,
            desc=f"Epoch [{epoch + 1}/{CFG['num_epochs']}]",
            unit="batch",
        )
        for batch_idx, (real_img, cond) in enumerate(pbar):
            # Opacus Poisson sampling can emit empty batches
            if real_img.size(0) == 0 or cond.size(0) == 0:
                pbar.set_postfix(D_Loss="skip(empty)", G_Loss="---")
                continue

            real_img = real_img.to(DEVICE)
            cond = cond.to(DEVICE)

            # ── D step ──────────────────────────────────────────────
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()
            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake_img = G(cond).detach()

            real_pred = D(cond, real_img)
            fake_pred = D(cond, fake_img)

            # Clamp logits to prevent BCE overflow
            real_pred = torch.clamp(real_pred, -CFG["logit_clamp"], CFG["logit_clamp"])
            fake_pred = torch.clamp(fake_pred, -CFG["logit_clamp"], CFG["logit_clamp"])

            # Label smoothing
            real_labels = torch.full_like(real_pred, CFG["real_label"])
            fake_labels = torch.full_like(fake_pred, CFG["fake_label"])

            d_loss = 0.5 * (
                criterion(real_pred, real_labels) + criterion(fake_pred, fake_labels)
            )

            if not torch.isfinite(d_loss):
                consecutive_nan += 1
                if consecutive_nan >= CFG["max_consecutive_nan"]:
                    raise RuntimeError(
                        f"{consecutive_nan} consecutive NaN batches. Training unstable."
                    )
                pbar.set_postfix(D_Loss="NaN(skip)", G_Loss="---")
                if hasattr(D, "disable_hooks"):
                    D.disable_hooks()
                continue

            consecutive_nan = 0
            d_loss.backward()
            opt_d.step()

            # ── G step ──────────────────────────────────────────────
            if hasattr(D, "disable_hooks"):
                D.disable_hooks()
            set_requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)

            fake_img = G(cond)
            fake_pred = D(cond, fake_img)
            fake_pred = torch.clamp(
                fake_pred, -CFG["logit_clamp"], CFG["logit_clamp"]
            )

            g_loss_gan = criterion(
                fake_pred, torch.full_like(fake_pred, CFG["real_label"])
            )
            g_loss_l1 = criterion_l1(fake_img, real_img) * CFG["lambda_l1"]
            g_loss = g_loss_gan + g_loss_l1

            if not torch.isfinite(g_loss):
                pbar.set_postfix(
                    D_Loss=f"{d_loss.item():.4f}", G_Loss="NaN(skip)"
                )
                if hasattr(D, "enable_hooks"):
                    D.enable_hooks()
                continue

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), CFG["g_grad_clip"])
            opt_g.step()

            if hasattr(D, "enable_hooks"):
                D.enable_hooks()

            seen += 1
            epoch_d += d_loss.item()
            epoch_g += g_loss.item()
            pbar.set_postfix(
                D_Loss=f"{d_loss.item():.4f}", G_Loss=f"{g_loss.item():.4f}"
            )

        avg_d = epoch_d / max(1, seen)
        avg_g = epoch_g / max(1, seen)

        # DP budget
        if privacy_engine:
            eps = privacy_engine.get_epsilon(CFG["target_delta"])
            print(f"[DP] epsilon spent: {eps:.2f}")

        # Save checkpoint ONLY if finite
        if is_finite(avg_d) and is_finite(avg_g):
            state = {
                "epoch": epoch + 1,
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                # BUG FIX: was `opt_d.step() or opt_d.state_dict()`
                # opt_d.step() caused an extra optimizer step during save!
                "opt_d": opt_d.state_dict(),
                "g_loss": avg_g,
                "d_loss": avg_d,
            }
            os.makedirs(os.path.dirname(CFG["checkpoint"]), exist_ok=True)
            torch.save(state, CFG["checkpoint"])
            notify_crash_save(epoch + 1, CFG["checkpoint"])

            if avg_g < best_g:
                best_g = avg_g
                torch.save(state, CFG["best_checkpoint"])
        else:
            print(
                f"[WARN] Epoch {epoch + 1} has non-finite loss. "
                "Checkpoint NOT saved."
            )

        notify_epoch(epoch + 1, CFG["num_epochs"], avg_d, avg_g)
        print(
            f"[EPOCH] {epoch + 1}/{CFG['num_epochs']} | "
            f"D={avg_d:.4f} | G={avg_g:.4f} | batches={seen}"
        )

    notify_training_complete(CFG["num_epochs"], avg_g)
    print("[DONE] Training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
