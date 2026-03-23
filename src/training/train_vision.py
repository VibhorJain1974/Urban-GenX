"""
Urban-GenX | Vision Node Training (HARDENED)
===========================================
Goals:
  1. Remove unsafe/corrupted checkpoint resumes
  2. Stabilize GAN training with explicit grad clipping and non-finite checks
  3. Keep Opacus-compatible alternating GAN updates
  4. Make privacy/release intent explicit via CLI flags

IMPORTANT PRIVACY NOTE
----------------------
This script supports two privacy modes:

1) research
   - DP is attached to the Discriminator only
   - secure_mode=False for speed
   - suitable for development/experiments
   - generator release is blocked by default

2) release
   - DP is attached to the Discriminator only
   - secure_mode=True
   - lambda_l1 must remain 0.0
   - generator release requires --release-generator and uses the
     post-processing interpretation documented in the dashboard/report

If you need a stricter construction where Generator itself is explicitly
DP-optimized with its own privacy engine, that is a different training design
and is NOT implemented in this drop-in file because the goal here is immediate
runnability with your current repository structure.
"""

import argparse
import math
import os
import random
import shutil
import sys
import tempfile
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
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

DEVICE = torch.device("cpu")

DEFAULT_CFG: Dict[str, Any] = {
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
    "lambda_l1": 0.0,
    "dp_enabled": True,
    "max_grad_norm_d": 1.0,
    "max_grad_norm_g": 1.0,
    "target_epsilon": 10.0,
    "target_delta": 1e-5,
    "accountant": "rdp",
    "seed": 42,
    "save_every": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardened Vision GAN training for Urban-GenX")
    parser.add_argument("--data-root", default=DEFAULT_CFG["data_root"])
    parser.add_argument("--checkpoint", default=DEFAULT_CFG["checkpoint"])
    parser.add_argument("--best-checkpoint", default=DEFAULT_CFG["best_checkpoint"])
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_CFG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--img-size", type=int, default=DEFAULT_CFG["img_size"])
    parser.add_argument("--lr-g", type=float, default=DEFAULT_CFG["lr_g"])
    parser.add_argument("--lr-d", type=float, default=DEFAULT_CFG["lr_d"])
    parser.add_argument("--noise-dim", type=int, default=DEFAULT_CFG["noise_dim"])
    parser.add_argument("--num-classes", type=int, default=DEFAULT_CFG["num_classes"])
    parser.add_argument("--target-epsilon", type=float, default=DEFAULT_CFG["target_epsilon"])
    parser.add_argument("--target-delta", type=float, default=DEFAULT_CFG["target_delta"])
    parser.add_argument("--max-grad-norm-d", type=float, default=DEFAULT_CFG["max_grad_norm_d"])
    parser.add_argument("--max-grad-norm-g", type=float, default=DEFAULT_CFG["max_grad_norm_g"])
    parser.add_argument("--lambda-l1", type=float, default=DEFAULT_CFG["lambda_l1"])
    parser.add_argument("--privacy-mode", choices=["research", "release"], default="research")
    parser.add_argument("--release-generator", action="store_true", help="Allow releasing generator weights under the documented post-processing interpretation")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint and start from epoch 0")
    parser.add_argument("--disable-dp", action="store_true", help="Disable Opacus entirely (debug only)")
    parser.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    cfg.update(
        {
            "data_root": args.data_root,
            "checkpoint": args.checkpoint,
            "best_checkpoint": args.best_checkpoint,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lr_g": args.lr_g,
            "lr_d": args.lr_d,
            "noise_dim": args.noise_dim,
            "num_classes": args.num_classes,
            "target_epsilon": args.target_epsilon,
            "target_delta": args.target_delta,
            "max_grad_norm_d": args.max_grad_norm_d,
            "max_grad_norm_g": args.max_grad_norm_g,
            "lambda_l1": args.lambda_l1,
            "seed": args.seed,
            "privacy_mode": args.privacy_mode,
            "release_generator": bool(args.release_generator),
            "fresh": bool(args.fresh),
            "dp_enabled": not bool(args.disable_dp),
            "secure_mode": True if args.privacy_mode == "release" else False,
        }
    )
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)



def is_finite_scalar(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False



def state_dict_has_nonfinite(state_dict: Dict[str, Any]) -> bool:
    for _, value in state_dict.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return True
    return False



def module_has_nonfinite(module: nn.Module) -> bool:
    return any(not torch.isfinite(p.detach()).all() for p in module.parameters())



def atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_ckpt_", suffix=".pth", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



def validate_checkpoint(ckpt: Dict[str, Any]) -> Tuple[bool, str]:
    required = ["epoch", "generator", "discriminator", "opt_g", "opt_d"]
    for key in required:
        if key not in ckpt:
            return False, f"missing key: {key}"

    epoch = ckpt.get("epoch", 0)
    if not isinstance(epoch, int) or epoch < 0:
        return False, "invalid epoch"

    if state_dict_has_nonfinite(ckpt["generator"]):
        return False, "generator contains non-finite values"
    if state_dict_has_nonfinite(ckpt["discriminator"]):
        return False, "discriminator contains non-finite values"

    for loss_key in ("g_loss", "d_loss", "best_score", "epsilon_spent"):
        if loss_key in ckpt and ckpt[loss_key] is not None and not is_finite_scalar(ckpt[loss_key]):
            return False, f"non-finite {loss_key}"

    return True, "ok"



def load_checkpoint_if_valid(path: str, fresh: bool = False) -> Optional[Dict[str, Any]]:
    if fresh:
        print("[INFO] --fresh specified; ignoring existing checkpoint.")
        return None

    if not os.path.exists(path):
        print(f"[INFO] No checkpoint at {path}, starting fresh.")
        return None

    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint: {e}. Starting fresh.")
        return None

    valid, reason = validate_checkpoint(ckpt)
    if not valid:
        bad_path = path + ".corrupt"
        try:
            shutil.copy2(path, bad_path)
            print(f"[WARN] Invalid checkpoint ({reason}). Backed up to {bad_path} and starting fresh.")
        except Exception:
            print(f"[WARN] Invalid checkpoint ({reason}). Starting fresh.")
        return None

    print(f"[INFO] Found valid checkpoint: epoch={ckpt.get('epoch', '?')}")
    return ckpt



def attach_dp_to_discriminator(D: nn.Module, opt_d: optim.Optimizer, loader: DataLoader, cfg: Dict[str, Any], remaining_epochs: int):
    privacy_engine = PrivacyEngine(
        secure_mode=cfg["secure_mode"],
        accountant=cfg["accountant"],
    )
    D, opt_d, loader = privacy_engine.make_private_with_epsilon(
        module=D,
        optimizer=opt_d,
        data_loader=loader,
        target_epsilon=cfg["target_epsilon"],
        target_delta=cfg["target_delta"],
        max_grad_norm=cfg["max_grad_norm_d"],
        epochs=remaining_epochs,
    )
    print(
        f"[DP] Attached to Discriminator | ε={cfg['target_epsilon']} δ={cfg['target_delta']} | "
        f"remaining_epochs={remaining_epochs} | accountant={cfg['accountant']} | secure_mode={cfg['secure_mode']}"
    )
    return privacy_engine, D, opt_d, loader



def describe_privacy_mode(cfg: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("Urban-GenX | Hardened Vision Training")
    print("=" * 72)
    print(f"[PRIVACY] Mode: {cfg['privacy_mode']}")
    print(f"[PRIVACY] DP enabled: {cfg['dp_enabled']}")
    print(f"[PRIVACY] secure_mode: {cfg['secure_mode']}")
    print(f"[PRIVACY] release_generator: {cfg['release_generator']}")
    print(f"[PRIVACY] lambda_l1: {cfg['lambda_l1']}")
    if cfg["privacy_mode"] == "research":
        print("[PRIVACY] Research mode: fast development mode; generator release is blocked by default.")
    else:
        print("[PRIVACY] Release mode: secure RNG enabled; generator release requires explicit flag and λ_L1=0.")
    print("=" * 72 + "\n")



def enforce_privacy_policy(cfg: Dict[str, Any]) -> None:
    if cfg["release_generator"] and cfg["privacy_mode"] != "release":
        raise ValueError("Generator release is only allowed in --privacy-mode release.")
    if cfg["release_generator"] and cfg["lambda_l1"] != 0.0:
        raise ValueError("Generator release requires --lambda-l1 0.0 so no direct private reconstruction term reaches G.")
    if cfg["release_generator"] and not cfg["dp_enabled"]:
        raise ValueError("Generator release requires DP to remain enabled.")



def save_training_checkpoint(
    path: str,
    epoch: int,
    G: nn.Module,
    D: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    g_loss: float,
    d_loss: float,
    epsilon_spent: Optional[float],
    cfg: Dict[str, Any],
    best_score: Optional[float],
) -> None:
    if module_has_nonfinite(G):
        raise RuntimeError("Refusing to save checkpoint: Generator has non-finite parameters.")
    if module_has_nonfinite(D):
        raise RuntimeError("Refusing to save checkpoint: Discriminator has non-finite parameters.")
    if not is_finite_scalar(g_loss) or not is_finite_scalar(d_loss):
        raise RuntimeError(f"Refusing to save checkpoint with non-finite losses: G={g_loss}, D={d_loss}")
    if epsilon_spent is not None and not is_finite_scalar(epsilon_spent):
        raise RuntimeError(f"Refusing to save checkpoint with non-finite epsilon: {epsilon_spent}")

    state = {
        "epoch": epoch,
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "g_loss": float(g_loss),
        "d_loss": float(d_loss),
        "epsilon_spent": None if epsilon_spent is None else float(epsilon_spent),
        "best_score": None if best_score is None else float(best_score),
        "cfg": cfg,
    }
    atomic_torch_save(state, path)
    notify_crash_save(epoch, path)



def maybe_load_states(
    ckpt: Optional[Dict[str, Any]],
    G: nn.Module,
    D: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
) -> Tuple[int, Optional[float]]:
    if ckpt is None:
        return 0, None

    start_epoch = int(ckpt.get("epoch", 0))
    best_score = ckpt.get("best_score", None)

    G.load_state_dict(ckpt["generator"], strict=True)
    D.load_state_dict(ckpt["discriminator"], strict=True)

    try:
        opt_g.load_state_dict(ckpt["opt_g"])
    except Exception as e:
        print(f"[WARN] Could not restore Generator optimizer state: {e}. Reinitializing opt_g state.")

    try:
        opt_d.load_state_dict(ckpt["opt_d"])
    except Exception as e:
        print(f"[WARN] Could not restore Discriminator optimizer state: {e}. Reinitializing opt_d state.")

    print(f"[INFO] Weights restored. Resuming from epoch {start_epoch}.")
    return start_epoch, best_score



def train(cfg: Dict[str, Any]) -> None:
    set_seed(cfg["seed"])
    describe_privacy_mode(cfg)
    enforce_privacy_policy(cfg)

    print("[DATA] Loading Cityscapes...")
    dataset = CityscapesDataset(cfg["data_root"], split="train", img_size=cfg["img_size"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=False,
        drop_last=False,
    )
    print(f"[DATA] {len(dataset)} samples | ~{len(loader)} batches/epoch")

    G = Generator(cfg["noise_dim"], cfg["num_classes"]).to(DEVICE)
    D = Discriminator(cfg["num_classes"]).to(DEVICE)

    if cfg["dp_enabled"]:
        G = ModuleValidator.fix(G)
        D = ModuleValidator.fix(D)

    opt_g = optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=cfg["betas"])
    opt_d = optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=cfg["betas"])

    ckpt = load_checkpoint_if_valid(cfg["checkpoint"], fresh=cfg["fresh"])
    start_epoch = int(ckpt["epoch"]) if ckpt is not None else 0
    remaining_epochs = max(0, cfg["num_epochs"] - start_epoch)

    if remaining_epochs == 0:
        print("[INFO] Training already complete (checkpoint epoch >= num_epochs). Nothing to do.")
        return

    privacy_engine = None
    if cfg["dp_enabled"]:
        privacy_engine, D, opt_d, loader = attach_dp_to_discriminator(D, opt_d, loader, cfg, remaining_epochs)

    start_epoch, best_score = maybe_load_states(ckpt, G, D, opt_g, opt_d)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    for epoch in range(start_epoch, cfg["num_epochs"]):
        G.train()
        D.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        seen_batches = 0

        pbar = tqdm(loader, desc=f"Epoch [{epoch + 1}/{cfg['num_epochs']}]", unit="batch")

        for batch_idx, (real_img, cond) in enumerate(pbar):
            real_img = real_img.to(DEVICE)
            cond = cond.to(DEVICE)
            seen_batches += 1

            if not torch.isfinite(real_img).all():
                raise RuntimeError(f"Non-finite real_img detected at epoch {epoch+1}, batch {batch_idx}")
            if not torch.isfinite(cond).all():
                raise RuntimeError(f"Non-finite cond detected at epoch {epoch+1}, batch {batch_idx}")

            # (1) Train D with DP
            if hasattr(D, "enable_hooks"):
                D.enable_hooks()
            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)

            fake_img = G(cond).detach()
            if not torch.isfinite(fake_img).all():
                raise RuntimeError(f"Non-finite fake_img (D step) at epoch {epoch+1}, batch {batch_idx}")

            real_pred = D(cond, real_img)
            fake_pred = D(cond, fake_img)

            d_loss = 0.5 * (
                criterion_gan(real_pred, torch.ones_like(real_pred))
                + criterion_gan(fake_pred, torch.zeros_like(fake_pred))
            )
            if not torch.isfinite(d_loss):
                raise RuntimeError(f"Non-finite d_loss at epoch {epoch+1}, batch {batch_idx}")

            d_loss.backward()
            opt_d.step()

            # (2) Train G with D frozen and D hooks disabled
            if hasattr(D, "disable_hooks"):
                D.disable_hooks()
            set_requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)

            fake_img = G(cond)
            if not torch.isfinite(fake_img).all():
                raise RuntimeError(f"Non-finite fake_img (G step) at epoch {epoch+1}, batch {batch_idx}")

            fake_pred = D(cond, fake_img)
            g_loss_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            g_loss_l1 = criterion_l1(fake_img, real_img) * float(cfg["lambda_l1"])
            g_loss = g_loss_gan + g_loss_l1
            if not torch.isfinite(g_loss):
                raise RuntimeError(f"Non-finite g_loss at epoch {epoch+1}, batch {batch_idx}")

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), cfg["max_grad_norm_g"])
            opt_g.step()

            if module_has_nonfinite(G):
                raise RuntimeError(f"Generator parameters became non-finite at epoch {epoch+1}, batch {batch_idx}")
            if module_has_nonfinite(D):
                raise RuntimeError(f"Discriminator parameters became non-finite at epoch {epoch+1}, batch {batch_idx}")

            if hasattr(D, "enable_hooks"):
                D.enable_hooks()

            epoch_d_loss += float(d_loss.item())
            epoch_g_loss += float(g_loss.item())
            pbar.set_postfix(D_Loss=f"{d_loss.item():.4f}", G_Loss=f"{g_loss.item():.4f}")

        avg_d = epoch_d_loss / max(1, seen_batches)
        avg_g = epoch_g_loss / max(1, seen_batches)

        if not is_finite_scalar(avg_d) or not is_finite_scalar(avg_g):
            raise RuntimeError(f"Non-finite epoch average detected at epoch {epoch+1}: D={avg_d}, G={avg_g}")

        epsilon_spent = None
        if privacy_engine is not None:
            epsilon_spent = privacy_engine.get_epsilon(cfg["target_delta"])
            if not is_finite_scalar(epsilon_spent):
                raise RuntimeError(f"Non-finite epsilon returned at epoch {epoch+1}: {epsilon_spent}")
            print(f"  [DP] ε spent: {epsilon_spent:.4f} / {cfg['target_epsilon']}")

        combined_score = avg_g + avg_d

        if (epoch + 1) % cfg["save_every"] == 0:
            save_training_checkpoint(
                cfg["checkpoint"],
                epoch + 1,
                G,
                D,
                opt_g,
                opt_d,
                avg_g,
                avg_d,
                epsilon_spent,
                cfg,
                best_score,
            )

        if best_score is None or combined_score < best_score:
            best_score = combined_score
            save_training_checkpoint(
                cfg["best_checkpoint"],
                epoch + 1,
                G,
                D,
                opt_g,
                opt_d,
                avg_g,
                avg_d,
                epsilon_spent,
                cfg,
                best_score,
            )
            print(f"[BEST] Updated best checkpoint at epoch {epoch+1} | score={best_score:.6f}")

        notify_epoch(epoch + 1, cfg["num_epochs"], avg_d, avg_g)
        print(f"[EPOCH] {epoch+1}/{cfg['num_epochs']} | D={avg_d:.4f} | G={avg_g:.4f}")

    notify_training_complete(cfg["num_epochs"], avg_g)
    print("[DONE] Vision training complete.")


if __name__ == "__main__":
    args = parse_args()
    cfg = build_cfg(args)
    try:
        train(cfg)
    except Exception:
        notify_error(traceback.format_exc())
        raise
