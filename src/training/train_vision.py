"""
Urban-GenX | Vision Node Training — PRODUCTION FINAL
=====================================================
Compatible with the GroupNorm-native Generator/Discriminator.
No ModuleValidator.fix() needed (already GroupNorm — no BN to replace).

Key fixes:
  - Generator uses GroupNorm natively (no Opacus BatchNorm conflict)
  - disable_hooks() / enable_hooks() for GAN-DP stability
  - Checkpoint validation before load
  - NaN skip + label smoothing
  - lambda_l1=0.0 for formal DP guarantee on released G
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
    notify_epoch, notify_crash_save,
    notify_training_complete, notify_error,
)

CFG = {
    "data_root":          "data/raw/cityscapes",
    "checkpoint":         "checkpoints/vision_checkpoint.pth",
    "best_checkpoint":    "checkpoints/vision_best.pth",
    "img_size":           64,
    "batch_size":         4,
    "num_workers":        0,
    "num_epochs":         50,
    "lr_g":               1e-4,
    "lr_d":               1e-4,
    "betas":              (0.5, 0.999),
    "noise_dim":          100,
    "num_classes":        35,
    "lambda_l1":          0.0,       # DP-safe: no direct real-image loss on G
    "dp_enabled":         True,
    "max_grad_norm":      1.0,
    "target_epsilon":     10.0,
    "target_delta":       1e-5,
    "real_label":         0.9,       # label smoothing
    "fake_label":         0.1,
    "logit_clamp":        10.0,
    "g_grad_clip":        1.0,
    "max_nan_batches":    20,
}

DEVICE = torch.device("cpu")


def set_requires_grad(mod, flag):
    for p in mod.parameters():
        p.requires_grad_(flag)


def validate_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    for key in ("g_loss", "d_loss"):
        v = ck.get(key)
        if v is not None and not math.isfinite(float(v)):
            print(f"[WARN] Checkpoint {key}={v} is non-finite. Ignoring.")
            os.rename(path, path + ".corrupt")
            return None
    print(f"[INFO] Checkpoint valid — epoch {ck.get('epoch','?')}")
    return ck


def train():
    print("=" * 70)
    print("  Urban-GenX | Vision DP-GAN  (GroupNorm native, no ModuleValidator)")
    print("=" * 70)

    # ── Data ────────────────────────────────────────────────────────────────
    dataset = CityscapesDataset(CFG["data_root"], split="train", img_size=CFG["img_size"])
    loader  = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True,
                         num_workers=CFG["num_workers"], pin_memory=False)
    print(f"[DATA] {len(dataset)} samples | {len(loader)} batches/epoch")

    # ── Models (GroupNorm-native — no ModuleValidator needed) ───────────────
    G = Generator(CFG["noise_dim"], CFG["num_classes"]).to(DEVICE)
    D = Discriminator(CFG["num_classes"]).to(DEVICE)

    # Verify Opacus compatibility
    errors = ModuleValidator.validate(D, strict=False)
    if errors:
        print(f"[WARN] Discriminator has {len(errors)} Opacus issues. Fixing...")
        D = ModuleValidator.fix(D)
    else:
        print("[INFO] Discriminator is Opacus-compatible (GroupNorm native) ✓")

    opt_g = optim.Adam(G.parameters(), lr=CFG["lr_g"], betas=CFG["betas"])
    opt_d = optim.Adam(D.parameters(), lr=CFG["lr_d"], betas=CFG["betas"])

    # ── Checkpoint & DP ─────────────────────────────────────────────────────
    ckpt = validate_checkpoint(CFG["checkpoint"])
    start_epoch = int(ckpt["epoch"]) if ckpt else 0
    remaining = max(1, CFG["num_epochs"] - start_epoch)

    privacy_engine = None
    if CFG["dp_enabled"]:
        privacy_engine = PrivacyEngine(accountant="rdp")
        D, opt_d, loader = privacy_engine.make_private_with_epsilon(
            module=D, optimizer=opt_d, data_loader=loader,
            target_epsilon=CFG["target_epsilon"],
            target_delta=CFG["target_delta"],
            max_grad_norm=CFG["max_grad_norm"],
            epochs=remaining,
        )
        print(f"[DP] Attached | ε={CFG['target_epsilon']} δ={CFG['target_delta']} remaining={remaining}")

    if ckpt:
        G.load_state_dict(ckpt["generator"])
        D.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        print(f"[INFO] Resumed from epoch {start_epoch}")

    criterion     = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()
    best_g        = float("inf")

    for epoch in range(start_epoch, CFG["num_epochs"]):
        G.train(); D.train()
        epoch_d = epoch_g = 0.0
        seen = nan_count = 0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{CFG['num_epochs']}]", unit="batch")

        for real_img, cond in pbar:
            if real_img.size(0) == 0:
                continue
            real_img, cond = real_img.to(DEVICE), cond.to(DEVICE)

            # ── D step ──────────────────────────────────────────────────────
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()
            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake = G(cond).detach()

            rp = torch.clamp(D(cond, real_img), -CFG["logit_clamp"], CFG["logit_clamp"])
            fp = torch.clamp(D(cond, fake),     -CFG["logit_clamp"], CFG["logit_clamp"])

            d_loss = 0.5 * (
                criterion(rp, torch.full_like(rp, CFG["real_label"])) +
                criterion(fp, torch.full_like(fp, CFG["fake_label"]))
            )

            if not torch.isfinite(d_loss):
                nan_count += 1
                if nan_count >= CFG["max_nan_batches"]:
                    raise RuntimeError(f"{nan_count} NaN batches — training unstable")
                if hasattr(D, "disable_hooks"): D.disable_hooks()
                continue

            nan_count = 0
            d_loss.backward()
            opt_d.step()

            # ── G step ──────────────────────────────────────────────────────
            if hasattr(D, "disable_hooks"):
                D.disable_hooks()
            set_requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)

            fake2 = G(cond)
            fp2   = torch.clamp(D(cond, fake2), -CFG["logit_clamp"], CFG["logit_clamp"])
            g_loss = criterion(fp2, torch.full_like(fp2, CFG["real_label"]))
            if CFG["lambda_l1"] > 0:
                g_loss = g_loss + criterion_l1(fake2, real_img) * CFG["lambda_l1"]

            if not torch.isfinite(g_loss):
                if hasattr(D, "enable_hooks"): D.enable_hooks()
                continue

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), CFG["g_grad_clip"])
            opt_g.step()
            if hasattr(D, "enable_hooks"): D.enable_hooks()

            seen += 1
            epoch_d += d_loss.item()
            epoch_g += g_loss.item()
            pbar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")

        avg_d = epoch_d / max(1, seen)
        avg_g = epoch_g / max(1, seen)

        if privacy_engine:
            eps = privacy_engine.get_epsilon(CFG["target_delta"])
            print(f"[DP] ε spent: {eps:.2f}")

        if math.isfinite(avg_d) and math.isfinite(avg_g):
            state = {
                "epoch": epoch + 1, "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "g_loss": avg_g, "d_loss": avg_d,
            }
            os.makedirs(os.path.dirname(CFG["checkpoint"]), exist_ok=True)
            torch.save(state, CFG["checkpoint"])
            notify_crash_save(epoch + 1, CFG["checkpoint"])
            if avg_g < best_g:
                best_g = avg_g
                torch.save(state, CFG["best_checkpoint"])

        notify_epoch(epoch + 1, CFG["num_epochs"], avg_d, avg_g)
        print(f"[EPOCH] {epoch+1}/{CFG['num_epochs']} | D={avg_d:.4f} | G={avg_g:.4f} | batches={seen}")

    notify_training_complete(CFG["num_epochs"], avg_g)
    print("[DONE] Vision training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
