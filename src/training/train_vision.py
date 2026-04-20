"""
Urban-GenX | Vision Training — Two-Phase DP-GAN
================================================
Phase 1: Standard GAN + L1 loss (no DP, 40 epochs)
Phase 2: DP fine-tune of Discriminator only (G frozen, 10 epochs)

Key fixes vs previous version:
  - Generator now uses GroupNorm (via vision_gan.py change)
  - Phase 2 checkpoint load uses strict=False to handle
    InstanceNorm -> GroupNorm key mismatch without retraining Phase 1
  - D.disable_hooks() / D.enable_hooks() around G backward
    prevents Opacus GradSampleModule activations.pop() crash
  - set_requires_grad(D, False) during G step prevents
    Opacus forbid_accumulation_hook ValueError
"""

import os
import sys
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from models.vision_gan import Generator, Discriminator, DiscriminatorDP, discriminator_to_dp
from src.utils.data_loader import CityscapesDataset
from src.utils.notifier import (
    notify_epoch,
    notify_crash_save,
    notify_training_complete,
    notify_error,
)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = {
    "data_root":   "data/raw/cityscapes",
    "checkpoint":  "checkpoints/vision_checkpoint.pth",
    "sample_dir":  "checkpoints/samples",
    "img_size":    64,
    "num_classes": 35,
    "noise_dim":   100,

    # Phase 1
    "p1_epochs":   40,
    "p1_lr_g":     2e-4,
    "p1_lr_d":     2e-4,
    "p1_lambda_l1": 10.0,
    "p1_batch":    4,

    # Phase 2 (DP)
    "p2_epochs":        10,
    "p2_lr_d":          1e-4,
    "p2_batch":         4,
    "target_epsilon":   10.0,
    "target_delta":     1e-5,
    "max_grad_norm":    1.0,
    "secure_mode":      False,

    "num_workers": 0,   # must be 0 on Windows
}

DEVICE = torch.device("cpu")
os.makedirs(CFG["sample_dir"], exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ─── Utilities ────────────────────────────────────────────────────────────────
def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str):
    if not os.path.exists(path):
        print(f"[INFO] No checkpoint at {path}, starting fresh.")
        return None
    ckpt = torch.load(path, map_location="cpu")
    print(f"[INFO] Loaded checkpoint: epoch={ckpt.get('epoch', '?')}  "
          f"best_G={ckpt.get('best_g_loss', float('inf')):.4f}")
    return ckpt


def save_sample_grid(G: nn.Module, n_samples: int, epoch: int, tag: str = "") -> None:
    """Save a grid of synthetic images to checkpoints/samples/."""
    G.eval()
    with torch.no_grad():
        cond = torch.zeros(n_samples, CFG["num_classes"], CFG["img_size"], CFG["img_size"])
        # simple condition: random dominant class per sample
        for i in range(n_samples):
            cls = torch.randint(0, CFG["num_classes"], (1,)).item()
            cond[i, cls, :, :] = 1.0
        fake = G(cond.to(DEVICE))
        fake = (fake + 1) / 2  # [-1,1] -> [0,1]
    path = os.path.join(CFG["sample_dir"], f"epoch{epoch:04d}{tag}.png")
    save_image(fake, path, nrow=n_samples)
    G.train()


# ─── Phase 1: Standard GAN + L1 ──────────────────────────────────────────────
def phase1_train(G: nn.Module, D: nn.Module, loader: DataLoader,
                 start_epoch: int, opt_g, opt_d) -> int:
    """
    Returns the epoch at which Phase 1 finished.
    If start_epoch >= p1_epochs, skips immediately.
    """
    if start_epoch >= CFG["p1_epochs"]:
        print(f"\n  [PHASE 1 SKIPPED] Already completed ({CFG['p1_epochs']} epochs).")
        return CFG["p1_epochs"]

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()

    best_g = float("inf")
    ckpt = load_checkpoint(CFG["checkpoint"])
    if ckpt and "best_g_loss" in ckpt:
        best_g = ckpt["best_g_loss"]

    for epoch in range(start_epoch, CFG["p1_epochs"]):
        G.train(); D.train()
        d_sum = g_sum = 0.0
        batches = 0

        pbar = tqdm(loader, desc=f"P1 Epoch [{epoch+1}/{CFG['p1_epochs']}]", unit="batch")
        for real_img, cond in pbar:
            real_img = real_img.to(DEVICE)
            cond     = cond.to(DEVICE)

            # ── D step ──────────────────────────────────────────────────────
            set_requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)

            fake_img  = G(cond).detach()
            real_pred = D(cond, real_img)
            fake_pred = D(cond, fake_img)

            d_loss = 0.5 * (
                criterion_gan(real_pred, torch.ones_like(real_pred)  * 0.9) +  # label smooth
                criterion_gan(fake_pred, torch.zeros_like(fake_pred) + 0.1)
            )
            d_loss.backward()
            opt_d.step()

            # ── G step ──────────────────────────────────────────────────────
            set_requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)

            fake_img  = G(cond)
            fake_pred = D(cond, fake_img)

            g_adv = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            g_l1  = criterion_l1(fake_img, real_img) * CFG["p1_lambda_l1"]
            g_loss = g_adv + g_l1
            g_loss.backward()
            opt_g.step()

            d_sum += float(d_loss.item())
            g_sum += float(g_loss.item())
            batches += 1

            pbar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")

        avg_d = d_sum / max(1, batches)
        avg_g = g_sum / max(1, batches)
        if avg_g < best_g:
            best_g = avg_g

        state = {
            "phase":       1,
            "epoch":       epoch + 1,
            "generator":   G.state_dict(),
            "discriminator": D.state_dict(),
            "opt_g":       opt_g.state_dict(),
            "opt_d":       opt_d.state_dict(),
            "best_g_loss": best_g,
        }
        save_checkpoint(state, CFG["checkpoint"])
        notify_crash_save(epoch + 1, CFG["checkpoint"])
        notify_epoch(epoch + 1, CFG["p1_epochs"], avg_d, avg_g)
        print(f"[EPOCH] {epoch+1}/{CFG['p1_epochs']} | D={avg_d:.4f} | G={avg_g:.4f} | batches={batches}")

    save_sample_grid(G, n_samples=4, epoch=CFG["p1_epochs"], tag="_p1_done")
    return CFG["p1_epochs"]


# ─── Phase 2: DP fine-tune (D only, G frozen) ────────────────────────────────
def phase2_dp_train(G: nn.Module, dp_loader: DataLoader) -> None:
    """
    Fine-tunes DiscriminatorDP under Opacus DP-SGD.
    Generator is frozen — no grads flow through G parameters.

    Crash prevention:
      - set_requires_grad(D_dp, False) during G backward
        -> prevents Opacus forbid_accumulation_hook ValueError
      - D_dp.disable_hooks() before G backward
        -> prevents GradSampleModule activations.pop() IndexError
        (happens when Poisson batch size == 1 with InstanceNorm;
         also happens if hooks fire on a forward that has no matching backward)
    """
    # Build DiscriminatorDP from Phase 1 checkpoint weights
    ckpt = load_checkpoint(CFG["checkpoint"])
    if ckpt is None:
        raise RuntimeError("Phase 2 requires a Phase 1 checkpoint. Run Phase 1 first.")

    D_src = Discriminator(CFG["num_classes"]).to(DEVICE)
    D_src.load_state_dict(ckpt["discriminator"])
    D_dp = discriminator_to_dp(D_src).to(DEVICE)

    # Load G weights with strict=False:
    #   Phase 1 checkpoint has InstanceNorm2d keys (running_mean, running_var).
    #   Current G has GroupNorm (no running stats). strict=False skips mismatched
    #   keys. Conv weights — which carry all the learned features — load correctly.
    #   G is frozen in Phase 2 so norm init values are irrelevant.
    G.load_state_dict(ckpt["generator"], strict=False)
    G.eval()
    set_requires_grad(G, False)   # freeze G entirely

    # Validate D_dp for Opacus (checks for unsupported layers)
    errors = ModuleValidator.validate(D_dp, strict=False)
    if errors:
        print(f"[DP] ModuleValidator warnings: {errors}")
        D_dp = ModuleValidator.fix(D_dp)

    opt_d = optim.Adam(D_dp.parameters(), lr=CFG["p2_lr_d"], betas=(0.5, 0.999))

    # Attach PrivacyEngine to D_dp
    privacy_engine = PrivacyEngine(secure_mode=CFG["secure_mode"])
    D_dp, opt_d, dp_loader = privacy_engine.make_private_with_epsilon(
        module=D_dp,
        optimizer=opt_d,
        data_loader=dp_loader,
        target_epsilon=CFG["target_epsilon"],
        target_delta=CFG["target_delta"],
        max_grad_norm=CFG["max_grad_norm"],
        epochs=CFG["p2_epochs"],
    )
    print(f"[DP] Engine attached | target ε={CFG['target_epsilon']} δ={CFG['target_delta']}")

    criterion_gan = nn.BCEWithLogitsLoss()

    for epoch in range(CFG["p2_epochs"]):
        D_dp.train()
        d_sum = 0.0
        batches = 0

        pbar = tqdm(dp_loader, desc=f"P2 DP Epoch [{epoch+1}/{CFG['p2_epochs']}]", unit="batch")

        for real_img, cond in pbar:
            real_img = real_img.to(DEVICE)
            cond     = cond.to(DEVICE)

            # ── D step (DP) ──────────────────────────────────────────────────
            # Hooks must be ON for D step so Opacus captures per-sample grads
            if hasattr(D_dp, "enable_hooks"):
                D_dp.enable_hooks()

            set_requires_grad(D_dp, True)
            opt_d.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake_img = G(cond)   # G is frozen, no grad needed

            real_pred = D_dp(cond, real_img)
            fake_pred = D_dp(cond, fake_img)

            d_loss = 0.5 * (
                criterion_gan(real_pred, torch.ones_like(real_pred)) +
                criterion_gan(fake_pred, torch.zeros_like(fake_pred))
            )
            d_loss.backward()
            opt_d.step()   # must step immediately under Poisson sampling

            d_sum  += float(d_loss.item())
            batches += 1
            pbar.set_postfix(D=f"{d_loss.item():.4f}")

        avg_d = d_sum / max(1, batches)
        eps   = privacy_engine.get_epsilon(CFG["target_delta"])
        print(f"[DP] epsilon spent: {eps:.2f}")
        print(f"[P2 EPOCH] {epoch+1}/{CFG['p2_epochs']} | D={avg_d:.4f} | ε={eps:.2f}")

        # Save Phase 2 checkpoint (separate key so Phase 1 ckpt is not overwritten)
        state = {
            "phase":          2,
            "p2_epoch":       epoch + 1,
            "generator":      G.state_dict(),
            "discriminator_dp": D_dp.state_dict(),
            "epsilon_spent":  eps,
        }
        p2_path = CFG["checkpoint"].replace(".pth", "_p2.pth")
        save_checkpoint(state, p2_path)
        notify_crash_save(epoch + 1, p2_path)
        notify_epoch(epoch + 1, CFG["p2_epochs"], avg_d, 0.0)

    save_sample_grid(G, n_samples=4, epoch=CFG["p2_epochs"], tag="_p2_done")
    notify_training_complete(CFG["p2_epochs"], 0.0)
    print(f"\n[P2 DONE] Final ε={eps:.4f} (target {CFG['target_epsilon']})")


# ─── Main ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 70)
    print("  Urban-GenX | Vision Training — Two-Phase DP-GAN")
    print("=" * 70)

    # ── Data ────────────────────────────────────────────────────────────────
    print("[DATA] Loading Cityscapes...")
    dataset = CityscapesDataset(
        CFG["data_root"], split="train", img_size=CFG["img_size"]
    )
    print(f"[DATA] {len(dataset)} samples loaded")

    p1_loader = DataLoader(
        dataset,
        batch_size=CFG["p1_batch"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=False,
        drop_last=True,   # avoid size-1 tail batch in Phase 1
    )

    # Phase 2 uses a separate loader; Opacus replaces its sampler internally
    p2_loader = DataLoader(
        dataset,
        batch_size=CFG["p2_batch"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=False,
        drop_last=False,
    )

    # ── Models ───────────────────────────────────────────────────────────────
    G = Generator(CFG["noise_dim"], CFG["num_classes"]).to(DEVICE)
    D = Discriminator(CFG["num_classes"]).to(DEVICE)

    print(f"[MODEL] Generator params:     {sum(p.numel() for p in G.parameters()):,}")
    print(f"[MODEL] Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    opt_g = optim.Adam(G.parameters(), lr=CFG["p1_lr_g"], betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=CFG["p1_lr_d"], betas=(0.5, 0.999))

    ckpt = load_checkpoint(CFG["checkpoint"])
    start_epoch = 0
    if ckpt is not None and ckpt.get("phase", 1) == 1:
        G.load_state_dict(ckpt["generator"], strict=False)
        D.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[INFO] Resumed: epoch={start_epoch}  best_G={ckpt.get('best_g_loss', 0):.4f}")

    print(f"\n{'='*70}")
    print(f"  PHASE 1: Standard GAN + L1 (no DP)")
    print(f"  Epochs: {CFG['p1_epochs']}  |  lambda_l1={CFG['p1_lambda_l1']}")
    print(f"{'='*70}")

    phase1_train(G, D, p1_loader, start_epoch, opt_g, opt_d)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PHASE 2: DP fine-tune (G frozen)")
    print(f"  Epochs: {CFG['p2_epochs']}  |  target_eps={CFG['target_epsilon']}")
    print(f"{'='*70}")

    phase2_dp_train(G, p2_loader)

    print("\n[DONE] Two-phase training complete.")
    print(f"       Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise