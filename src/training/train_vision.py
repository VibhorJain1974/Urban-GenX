"""
Urban-GenX | Vision Node Training  (FINAL STABLE)
==================================================
Fixes every issue from the conversation history:
  1. Opacus 'multiple values' error → set rdp ONLY on PrivacyEngine()
  2. Opacus "Poisson sampling grad accumulation" → disable_hooks() during G step
  3. Opacus "activations.pop empty" → disable_hooks() during G step
  4. GAN NaN at batch 17 → label smoothing + logit clamping + NaN-skip
  5. Corrupted checkpoint resume → validate before loading
  6. All stray citation tokens removed
  7. lambda_l1=0.0 for DP-safe generator release (Option C)
"""

import os, sys, math, traceback
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
    notify_epoch, notify_crash_save, notify_training_complete, notify_error,
)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
CFG = {
    "data_root":      "data/raw/cityscapes",
    "checkpoint":     "checkpoints/vision_checkpoint.pth",
    "best_checkpoint":"checkpoints/vision_best.pth",
    "img_size":       64,
    "batch_size":     4,
    "num_workers":    0,
    "num_epochs":     50,
    "lr_g":           1e-4,          # lower than 2e-4 for stability
    "lr_d":           1e-4,
    "betas":          (0.5, 0.999),
    "noise_dim":      100,
    "num_classes":    35,
    "lambda_l1":      0.0,           # MUST be 0 for DP-safe G release

    # DP settings
    "dp_enabled":     True,
    "max_grad_norm":  1.0,
    "target_epsilon": 10.0,
    "target_delta":   1e-5,

    # GAN stability
    "real_label":     0.9,           # label smoothing
    "fake_label":     0.1,
    "logit_clamp":    10.0,          # clamp D output to [-10, 10]
    "g_grad_clip":    1.0,           # clip G gradients
    "max_consecutive_nan": 20,       # crash only after this many NaN in a row
}

DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def is_finite(x):
    try:
        return math.isfinite(float(x))
    except (ValueError, TypeError):
        return False


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def state_dict_has_nonfinite(sd):
    """Return True if any tensor in a state_dict contains NaN or Inf."""
    for v in sd.values():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            return True
    return False


def validate_checkpoint(path):
    """Load a checkpoint and return it only if valid. Returns None if bad."""
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return None

    # Reject if stored losses are NaN
    g_loss = ckpt.get("g_loss", 0.0)
    d_loss = ckpt.get("d_loss", 0.0)
    if not is_finite(g_loss) or not is_finite(d_loss):
        print(f"[WARN] Checkpoint has NaN losses (d={d_loss}, g={g_loss}). Ignoring.")
        backup = path + ".corrupt"
        try:
            os.rename(path, backup)
            print(f"[WARN] Renamed to {backup}")
        except OSError:
            pass
        return None

    # Reject if weights contain NaN
    for key in ("generator", "discriminator"):
        if key in ckpt and state_dict_has_nonfinite(ckpt[key]):
            print(f"[WARN] Checkpoint {key} weights contain NaN/Inf. Ignoring.")
            return None

    return ckpt


def save_checkpoint(state, path):
    """Save checkpoint atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════
def train():
    print("\n" + "=" * 70)
    print("  Urban-GenX | Vision DP-GAN Training (Final Stable)")
    print("=" * 70)
    print(f"  DP enabled:   {CFG['dp_enabled']}")
    print(f"  lambda_l1:    {CFG['lambda_l1']}")
    print(f"  label smooth: real={CFG['real_label']}, fake={CFG['fake_label']}")
    print(f"  logit clamp:  ±{CFG['logit_clamp']}")
    print(f"  G grad clip:  {CFG['g_grad_clip']}")
    print("=" * 70 + "\n")

    # ── Data ────────────────────────────────────────────────────────
    print("[DATA] Loading Cityscapes...")
    dataset = CityscapesDataset(CFG["data_root"], split="train", img_size=CFG["img_size"])
    loader = DataLoader(
        dataset, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=False, drop_last=False,
    )
    print(f"[DATA] {len(dataset)} samples | ~{len(loader)} batches/epoch")

    # ── Models ──────────────────────────────────────────────────────
    G = Generator(CFG["noise_dim"], CFG["num_classes"]).to(DEVICE)
    D = Discriminator(CFG["num_classes"]).to(DEVICE)

    if CFG["dp_enabled"]:
        G = ModuleValidator.fix(G)
        D = ModuleValidator.fix(D)

    # ── Optimizers ──────────────────────────────────────────────────
    opt_g = optim.Adam(G.parameters(), lr=CFG["lr_g"], betas=CFG["betas"])
    opt_d = optim.Adam(D.parameters(), lr=CFG["lr_d"], betas=CFG["betas"])

    # ── Resume ──────────────────────────────────────────────────────
    ckpt = validate_checkpoint(CFG["checkpoint"])
    start_epoch = 0
    best_g_loss = float("inf")

    # ── Attach DP ───────────────────────────────────────────────────
    # KEY FIX: set accountant="rdp" ONLY on PrivacyEngine constructor.
    # Do NOT pass accountant to make_private_with_epsilon.
    privacy_engine = None
    if CFG["dp_enabled"]:
        remaining_epochs = CFG["num_epochs"] - start_epoch
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
        print(f"[DP] Attached | ε={CFG['target_epsilon']} δ={CFG['target_delta']}"
              f" remaining_epochs={remaining_epochs} accountant=rdp")

    # ── Load weights AFTER DP wrapping ──────────────────────────────
    if ckpt is not None:
        try:
            G.load_state_dict(ckpt["generator"])
            D.load_state_dict(ckpt["discriminator"])
            opt_g.load_state_dict(ckpt["opt_g"])
            opt_d.load_state_dict(ckpt["opt_d"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_g_loss = float(ckpt.get("best_g_loss", float("inf")))
            print(f"[INFO] Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint weights: {e}. Starting fresh.")
            start_epoch = 0
    else:
        print("[INFO] No checkpoint found. Starting fresh.")

    # ── Losses ──────────────────────────────────────────────────────
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    # ── Training Loop ───────────────────────────────────────────────
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

            # ════════════════════════════════════════════════════════
            # (1) TRAIN DISCRIMINATOR (DP-wrapped)
            # ════════════════════════════════════════════════════════
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()
            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)

            fake_img = G(cond).detach()

            real_pred = D(cond, real_img)
            fake_pred = D(cond, fake_img)

            # Clamp logits to prevent BCE overflow → NaN
            real_pred = torch.clamp(real_pred, -CFG["logit_clamp"], CFG["logit_clamp"])
            fake_pred = torch.clamp(fake_pred, -CFG["logit_clamp"], CFG["logit_clamp"])

            # Label smoothing: real=0.9, fake=0.1
            real_labels = torch.full_like(real_pred, CFG["real_label"])
            fake_labels = torch.full_like(fake_pred, CFG["fake_label"])

            d_loss_real = criterion_gan(real_pred, real_labels)
            d_loss_fake = criterion_gan(fake_pred, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # NaN check for D
            if not torch.isfinite(d_loss):
                consecutive_nan += 1
                if consecutive_nan >= CFG["max_consecutive_nan"]:
                    raise RuntimeError(
                        f"FATAL: {consecutive_nan} consecutive NaN batches. "
                        "Check data/model. Aborting."
                    )
                pbar.set_postfix(D_Loss="NaN(skip)", G_Loss="---")
                if hasattr(D, "enable_hooks"):
                    D.enable_hooks()
                continue

            d_loss.backward()
            opt_d.step()

            # ════════════════════════════════════════════════════════
            # (2) TRAIN GENERATOR (hooks disabled, D frozen)
            # ════════════════════════════════════════════════════════
            if hasattr(D, "disable_hooks"):
                D.disable_hooks()
            set_requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)

            fake_img = G(cond)
            fake_pred = D(cond, fake_img)
            fake_pred = torch.clamp(fake_pred, -CFG["logit_clamp"], CFG["logit_clamp"])

            g_loss_gan = criterion_gan(fake_pred, torch.full_like(fake_pred, CFG["real_label"]))
            g_loss_l1 = criterion_l1(fake_img, real_img) * CFG["lambda_l1"]
            g_loss = g_loss_gan + g_loss_l1

            # NaN check for G
            if not torch.isfinite(g_loss):
                consecutive_nan += 1
                if consecutive_nan >= CFG["max_consecutive_nan"]:
                    raise RuntimeError(
                        f"FATAL: {consecutive_nan} consecutive NaN batches. Aborting."
                    )
                pbar.set_postfix(D_Loss=f"{d_loss.item():.4f}", G_Loss="NaN(skip)")
                if hasattr(D, "enable_hooks"):
                    D.enable_hooks()
                continue

            g_loss.backward()
            # Clip G gradients for stability
            torch.nn.utils.clip_grad_norm_(G.parameters(), CFG["g_grad_clip"])
            opt_g.step()

            # Re-enable hooks for next D step
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()

            # ── Accumulate ──────────────────────────────────────────
            consecutive_nan = 0
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            valid_batches += 1

            pbar.set_postfix(
                D_Loss=f"{d_loss.item():.4f}",
                G_Loss=f"{g_loss.item():.4f}",
            )

        # ── Epoch Summary ───────────────────────────────────────────
        if valid_batches == 0:
            print(f"[WARN] Epoch {epoch+1}: all batches NaN. Skipping save.")
            continue

        avg_d = epoch_d_loss / valid_batches
        avg_g = epoch_g_loss / valid_batches

        # DP budget
        if privacy_engine is not None:
            eps = privacy_engine.get_epsilon(CFG["target_delta"])
            print(f"  [DP] ε spent: {eps:.2f}")

        # Save checkpoint only if losses are finite
        if is_finite(avg_d) and is_finite(avg_g):
            state = {
                "epoch": epoch + 1,
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "g_loss": avg_g,
                "d_loss": avg_d,
                "best_g_loss": best_g_loss,
            }
            save_checkpoint(state, CFG["checkpoint"])
            notify_crash_save(epoch + 1, CFG["checkpoint"])

            # Best checkpoint
            if avg_g < best_g_loss:
                best_g_loss = avg_g
                state["best_g_loss"] = best_g_loss
                save_checkpoint(state, CFG["best_checkpoint"])
                print(f"  ★ New best G loss: {avg_g:.4f}")
        else:
            print(f"  [WARN] Non-finite avg loss (D={avg_d}, G={avg_g}). Skipping save.")

        notify_epoch(epoch + 1, CFG["num_epochs"], avg_d, avg_g)
        print(f"  [EPOCH] {epoch+1}/{CFG['num_epochs']} | D={avg_d:.4f} | G={avg_g:.4f}"
              f" | valid_batches={valid_batches}/{len(loader)}\n")

    notify_training_complete(CFG["num_epochs"], avg_g)
    print("[DONE] Vision training complete.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
