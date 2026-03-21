"""
Urban-GenX | Acoustic Node Training  (updated)
Trains AcousticVAE on UrbanSound8K MFCCs with:
  - lr = 5e-4  (tuned: was 1e-3, now more stable)
  - mean-reduction VAE loss  (fixed: was sum → inflated 50k values)
  - Beta-VAE KL annealing  (0 → 1 over 10 epochs)
  - ExponentialLR scheduler (lr × 0.95 per epoch)
  - Gradient clipping (max_norm=5.0)
  - Epoch checkpointing + best-val checkpoint
  - tqdm progress bars (recon + kl components shown)
  - ntfy.sh mobile alerts
  - Fold 10 held out as validation (standard UrbanSound8K protocol)
  - Auto-detects both kaggle & soundata folder layouts
"""

import os
import sys
import traceback

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.acoustic_vae import AcousticVAE
from src.utils.data_loader import UrbanSound8KDataset
from src.utils.notifier import (
    notify_epoch,
    notify_crash_save,
    notify_training_complete,
    notify_error,
)

# ─── Config ───────────────────────────────────────────────────────────────────
CFG = {
    # Folder containing metadata/ and audio/ subfolders
    # Common variants:
    #   "data/raw/urbansound8k/UrbanSound8K"  (soundata extract)
    #   "data/raw/urbansound8k"               (Kaggle direct extract)
    "data_root":           "data/raw/urbansound8k",
    "checkpoint":          "checkpoints/acoustic_checkpoint.pth",
    "best_checkpoint":     "checkpoints/acoustic_best.pth",   # best val loss

    "batch_size":          16,
    "num_workers":         0,       # Windows: must stay 0
    "num_epochs":          50,

    # Optimizer
    "lr":                  5e-4,    # tuned from 1e-3 → more stable convergence
    "lr_decay":            0.95,    # ExponentialLR factor per epoch

    # Model
    "latent_dim":          64,
    "n_mfcc":              40,
    "time_frames":         128,

    # Beta-VAE annealing: KL weight 0 → 1 over first 10 epochs
    # Epoch 0: beta=0.0  (pure reconstruction focus)
    # Epoch 10+: beta=1.0 (full ELBO)
    "beta_start":          0.0,
    "beta_end":            1.0,
    "beta_anneal_epochs":  10,
}

DEVICE = torch.device("cpu")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_beta(epoch: int) -> float:
    """Linear KL annealing schedule: 0 → 1 over beta_anneal_epochs."""
    if epoch >= CFG["beta_anneal_epochs"]:
        return float(CFG["beta_end"])
    frac = epoch / max(1, CFG["beta_anneal_epochs"])
    return CFG["beta_start"] + frac * (CFG["beta_end"] - CFG["beta_start"])


def resolve_data_root(root: str) -> str:
    """
    Auto-detect correct data root.
    Handles both Kaggle layout (folds directly inside root) and
    soundata layout (extra UrbanSound8K/ subfolder).
    """
    if os.path.exists(os.path.join(root, "metadata", "UrbanSound8K.csv")):
        return root
    alt = os.path.join(root, "UrbanSound8K")
    if os.path.exists(os.path.join(alt, "metadata", "UrbanSound8K.csv")):
        print(f"[DATA] Auto-detected UrbanSound8K at: {alt}")
        return alt
    # Last attempt: walk up one level
    parent = os.path.dirname(root)
    alt2 = os.path.join(parent, "UrbanSound8K")
    if os.path.exists(os.path.join(alt2, "metadata", "UrbanSound8K.csv")):
        print(f"[DATA] Auto-detected UrbanSound8K at: {alt2}")
        return alt2
    print(
        f"[WARN] Cannot find UrbanSound8K.csv under: {root}\n"
        f"       Expected: {root}/metadata/UrbanSound8K.csv\n"
        f"       Check your data path and re-run."
    )
    return root   # return as-is; will crash at Dataset init with clear error


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: AcousticVAE, optimizer) -> int:
    """Load checkpoint and return start_epoch. Returns 0 if no checkpoint."""
    if not os.path.exists(path):
        print(f"[INFO] No checkpoint at {path}. Starting fresh.")
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    epoch = int(ckpt.get("epoch", 0))
    avg   = ckpt.get("avg_loss", "?")
    val   = ckpt.get("avg_val_loss", "?")
    print(f"[INFO] Resumed from epoch {epoch} | Train={avg:.4f} | Val={val:.4f}"
          if isinstance(avg, float) else f"[INFO] Resumed from epoch {epoch}")
    return epoch


# ─── Main Training ────────────────────────────────────────────────────────────
def train():

    # ── Data ────────────────────────────────────────────────────────────────
    print("[DATA] Loading UrbanSound8K...")
    data_root = resolve_data_root(CFG["data_root"])

    # Standard UrbanSound8K split: folds 1-9 train, fold 10 val
    train_dataset = UrbanSound8KDataset(
        root=data_root,
        folds=list(range(1, 10)),   # 7,895 clips
        n_mfcc=CFG["n_mfcc"],
        time_frames=CFG["time_frames"],
    )
    val_dataset = UrbanSound8KDataset(
        root=data_root,
        folds=[10],                  # 837 clips
        n_mfcc=CFG["n_mfcc"],
        time_frames=CFG["time_frames"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
    )

    print(
        f"[DATA] Train: {len(train_dataset)} clips | "
        f"Val: {len(val_dataset)} clips | "
        f"Batches/epoch: {len(train_loader)}"
    )

    # ── Model & Optimizer ────────────────────────────────────────────────────
    model = AcousticVAE(
        mfcc_bins=CFG["n_mfcc"],
        time_frames=CFG["time_frames"],
        latent_dim=CFG["latent_dim"],
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])

    # ExponentialLR: lr = lr_initial × (lr_decay ^ epoch)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=CFG["lr_decay"]
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(CFG["checkpoint"], model, optimizer)

    # Track best validation loss for best-model checkpoint
    best_val = float("inf")

    # ── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        model.train()
        beta = get_beta(epoch)

        # Per-epoch accumulators
        sum_loss  = 0.0
        sum_recon = 0.0
        sum_kl    = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Acoustic [{epoch+1}/{CFG['num_epochs']}] β={beta:.2f}",
            unit="batch",
            dynamic_ncols=True,
        )

        for x, _label in pbar:
            x = x.to(DEVICE)     # [B, 1, 40, 128]

            optimizer.zero_grad(set_to_none=True)

            recon, mu, lv = model(x)

            # Combined Beta-VAE loss (mean reduction — fixed from sum)
            loss = AcousticVAE.loss(recon, x, mu, lv, beta=beta)

            loss.backward()

            # Gradient clipping: prevents rare exploding-gradient spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # ── Per-batch component breakdown (for tqdm display) ────────────
            with torch.no_grad():
                recon_val = torch.nn.functional.mse_loss(recon, x, reduction='mean').item()
                kl_val    = float(
                    -0.5 * torch.mean(
                        torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)
                    ) / mu.size(1)
                )

            sum_loss  += float(loss.item())
            sum_recon += recon_val
            sum_kl    += kl_val
            n_batches += 1

            pbar.set_postfix(
                loss  = f"{loss.item():.4f}",
                recon = f"{recon_val:.4f}",
                kl    = f"{kl_val:.4f}",
                lr    = f"{scheduler.get_last_lr()[0]:.2e}",
            )

        # Step LR scheduler after each epoch
        scheduler.step()

        avg_loss  = sum_loss  / max(1, n_batches)
        avg_recon = sum_recon / max(1, n_batches)
        avg_kl    = sum_kl    / max(1, n_batches)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_sum  = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_v, _ in val_loader:
                x_v = x_v.to(DEVICE)
                recon_v, mu_v, lv_v = model(x_v)
                # Always use beta=1.0 for val (stable reference point)
                val_sum += AcousticVAE.loss(recon_v, x_v, mu_v, lv_v, beta=1.0).item()
                val_batches += 1
        avg_val = val_sum / max(1, val_batches)

        # ── Console log ──────────────────────────────────────────────────────
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[ACOUSTIC] {epoch+1:02d}/{CFG['num_epochs']} | "
            f"Train={avg_loss:.4f} | Val={avg_val:.4f} | "
            f"Recon={avg_recon:.4f} | KL={avg_kl:.4f} | "
            f"β={beta:.2f} | LR={current_lr:.2e}"
        )

        # ── Checkpoint: save every epoch (crash recovery) ─────────────────
        epoch_state = {
            "epoch":        epoch + 1,
            "model":        model.state_dict(),
            "opt":          optimizer.state_dict(),
            "avg_loss":     avg_loss,
            "avg_val_loss": avg_val,
            "beta":         beta,
            "lr":           current_lr,
            # Save model config so it can be loaded without CFG
            "model_config": {
                "mfcc_bins":   CFG["n_mfcc"],
                "time_frames": CFG["time_frames"],
                "latent_dim":  CFG["latent_dim"],
            },
        }
        save_checkpoint(epoch_state, CFG["checkpoint"])
        notify_crash_save(epoch + 1, CFG["checkpoint"])

        # ── Best checkpoint: save when validation improves ─────────────────
        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(epoch_state, CFG["best_checkpoint"])
            print(f"  [★] New best val loss: {best_val:.4f} → saved to {CFG['best_checkpoint']}")

        # ── Mobile notification (once per epoch) ──────────────────────────
        notify_epoch(
            epoch + 1,
            CFG["num_epochs"],
            d_loss=avg_recon,    # recon in "D_Loss" slot (re-used for readability)
            g_loss=avg_val,      # val loss in "G_Loss" slot
        )

    # ── Done ─────────────────────────────────────────────────────────────────
    notify_training_complete(CFG["num_epochs"], avg_loss)
    print(f"[DONE] Acoustic training complete. Best val loss: {best_val:.4f}")
    print(f"       Best model saved at: {CFG['best_checkpoint']}")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        notify_error(traceback.format_exc())
        raise
