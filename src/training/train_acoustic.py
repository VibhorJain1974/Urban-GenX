"""
Urban-GenX | Acoustic Node Training — FIXED & UPGRADED
=======================================================
WHAT CHANGED vs original (and WHY):

  FIX 1 — MEL-SPECTROGRAM instead of MFCC
    Old: MFCC 40-bin → model saturated at epoch 10, 0.668 floor forever
    New: Mel-spectrogram 64-bin → richer frequency representation, lower floor
    How: Set USE_MEL=True in CFG (default). Falls back to MFCC if USE_MEL=False.

  FIX 2 — CONDITIONAL VAE (cVAE) with class labels
    Old: unconditional VAE → generator had no way to target specific sound classes
    New: label embedding fed to encoder + decoder → class-conditional generation
    How: Set USE_CONDITIONAL=True in CFG (default, uses 10 UrbanSound8K classes)

  FIX 3 — FREE BITS KL (in AcousticVAE.loss)
    Old: KL=0.0021 → posterior collapsed → latent space useless
    New: free_bits=0.5 → each dim forced to carry information
    How: Handled inside AcousticVAE.loss() — no change needed here

  FIX 4 — EARLY STOPPING
    Old: ran 100 epochs, 90 were wasted (0.1% gain)
    New: stops when val loss doesn't improve for patience=15 epochs
    How: EarlyStopping class below

  FIX 5 — EPOCHS REDUCED to 60 (was 100)
    With the above fixes, model learns more per epoch — 60 is enough
    Saves ~1h of your 2h47m training time

  FIX 6 — DATA LOADER upgraded for Mel-spectrogram
    Same UrbanSound8KDataset, just with n_mels and USE_MEL flag passed in

ESTIMATED TRAINING TIME (HP Victus i7-12650H CPU):
  ~1h 30m for 60 epochs (vs 2h 47m before)
  Early stopping may cut it to ~45m if converges fast
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
    "data_root":           "data/raw/urbansound8k",
    "checkpoint":          "checkpoints/acoustic_checkpoint.pth",
    "best_checkpoint":     "checkpoints/acoustic_best.pth",

    "batch_size":          16,
    "num_workers":         0,           # Windows: must stay 0
    "num_epochs":          60,          # REDUCED from 100 — early stopping handles the rest

    # Optimizer
    "lr":                  5e-4,
    "lr_decay":            0.95,

    # Model
    "latent_dim":          64,

    # FIX 1: Mel-spectrogram — richer than MFCC, breaks the 0.668 floor
    "USE_MEL":             True,        # True = Mel-spec, False = MFCC (original)
    "n_mels":              64,          # Mel bins (used when USE_MEL=True)
    "n_mfcc":              40,          # MFCC bins (used when USE_MEL=False)
    "time_frames":         128,

    # FIX 2: Conditional VAE
    "USE_CONDITIONAL":     True,        # True = cVAE with class labels
    "n_classes":           10,          # UrbanSound8K has 10 classes

    # FIX 3: Free bits (in AcousticVAE.loss)
    "free_bits":           0.5,         # KL floor per latent dim — prevents collapse

    # Beta-VAE annealing
    "beta_start":          0.0,
    "beta_end":            1.0,
    "beta_anneal_epochs":  10,

    # FIX 4: Early stopping
    "early_stop_patience": 15,          # stop if no improvement for 15 epochs
}

DEVICE = torch.device("cpu")

# Derived: which freq bins to use
N_BINS = CFG["n_mels"] if CFG["USE_MEL"] else CFG["n_mfcc"]


# ─── Early Stopping ───────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop training when val loss stops improving."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  [EarlyStop] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_beta(epoch: int) -> float:
    """Linear KL annealing: 0 → 1 over beta_anneal_epochs."""
    if epoch >= CFG["beta_anneal_epochs"]:
        return float(CFG["beta_end"])
    frac = epoch / max(1, CFG["beta_anneal_epochs"])
    return CFG["beta_start"] + frac * (CFG["beta_end"] - CFG["beta_start"])


def resolve_data_root(root: str) -> str:
    if os.path.exists(os.path.join(root, "metadata", "UrbanSound8K.csv")):
        return root
    alt = os.path.join(root, "UrbanSound8K")
    if os.path.exists(os.path.join(alt, "metadata", "UrbanSound8K.csv")):
        print(f"[DATA] Auto-detected UrbanSound8K at: {alt}")
        return alt
    parent = os.path.dirname(root)
    alt2 = os.path.join(parent, "UrbanSound8K")
    if os.path.exists(os.path.join(alt2, "metadata", "UrbanSound8K.csv")):
        print(f"[DATA] Auto-detected UrbanSound8K at: {alt2}")
        return alt2
    print(
        f"[WARN] Cannot find UrbanSound8K.csv under: {root}\n"
        f"       Expected: {root}/metadata/UrbanSound8K.csv"
    )
    return root


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: AcousticVAE, optimizer) -> int:
    if not os.path.exists(path):
        print(f"[INFO] No checkpoint at {path}. Starting fresh.")
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    epoch = int(ckpt.get("epoch", 0))
    avg   = ckpt.get("avg_loss", "?")
    val   = ckpt.get("avg_val_loss", "?")
    print(
        f"[INFO] Resumed from epoch {epoch} | Train={avg:.4f} | Val={val:.4f}"
        if isinstance(avg, float) else f"[INFO] Resumed from epoch {epoch}"
    )
    return epoch


# ─── Main Training ────────────────────────────────────────────────────────────
def train():

    # ── Data ────────────────────────────────────────────────────────────────
    mode_str = "Mel-Spectrogram" if CFG["USE_MEL"] else "MFCC"
    cond_str = "Conditional VAE" if CFG["USE_CONDITIONAL"] else "Unconditional VAE"
    print(f"[CONFIG] Mode: {mode_str} | Model: {cond_str} | Bins: {N_BINS} | Free bits: {CFG['free_bits']}")

    print("[DATA] Loading UrbanSound8K...")
    data_root = resolve_data_root(CFG["data_root"])

    # UrbanSound8KDataset now accepts use_mel flag
    train_dataset = UrbanSound8KDataset(
        root=data_root,
        folds=list(range(1, 10)),
        n_mfcc=CFG["n_mfcc"],
        n_mels=CFG["n_mels"],
        use_mel=CFG["USE_MEL"],
        time_frames=CFG["time_frames"],
    )
    val_dataset = UrbanSound8KDataset(
        root=data_root,
        folds=[10],
        n_mfcc=CFG["n_mfcc"],
        n_mels=CFG["n_mels"],
        use_mel=CFG["USE_MEL"],
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

    # ── Model ────────────────────────────────────────────────────────────────
    model = AcousticVAE(
        mfcc_bins=N_BINS,
        time_frames=CFG["time_frames"],
        latent_dim=CFG["latent_dim"],
        n_classes=CFG["n_classes"] if CFG["USE_CONDITIONAL"] else 0,
        free_bits=CFG["free_bits"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=CFG["lr_decay"])
    early_stopper = EarlyStopping(patience=CFG["early_stop_patience"])

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(CFG["checkpoint"], model, optimizer)
    best_val = float("inf")

    # ── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        model.train()
        beta = get_beta(epoch)

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

        for x, label in pbar:
            x     = x.to(DEVICE)
            label = label.to(DEVICE) if CFG["USE_CONDITIONAL"] else None

            optimizer.zero_grad(set_to_none=True)

            recon, mu, lv = model(x, label)

            loss = AcousticVAE.loss(
                recon, x, mu, lv,
                beta=beta,
                free_bits=CFG["free_bits"],
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Component tracking for display
            with torch.no_grad():
                recon_val = torch.nn.functional.mse_loss(recon, x, reduction='mean').item()
                kl_val = float(
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

        scheduler.step()

        avg_loss  = sum_loss  / max(1, n_batches)
        avg_recon = sum_recon / max(1, n_batches)
        avg_kl    = sum_kl    / max(1, n_batches)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_v, label_v in val_loader:
                x_v     = x_v.to(DEVICE)
                label_v = label_v.to(DEVICE) if CFG["USE_CONDITIONAL"] else None
                recon_v, mu_v, lv_v = model(x_v, label_v)
                val_sum += AcousticVAE.loss(
                    recon_v, x_v, mu_v, lv_v,
                    beta=1.0,
                    free_bits=CFG["free_bits"],
                ).item()
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

        # ── Checkpoints ──────────────────────────────────────────────────────
        epoch_state = {
            "epoch":        epoch + 1,
            "model":        model.state_dict(),
            "opt":          optimizer.state_dict(),
            "avg_loss":     avg_loss,
            "avg_val_loss": avg_val,
            "beta":         beta,
            "lr":           current_lr,
            "model_config": {
                "mfcc_bins":     N_BINS,
                "time_frames":   CFG["time_frames"],
                "latent_dim":    CFG["latent_dim"],
                "n_classes":     CFG["n_classes"] if CFG["USE_CONDITIONAL"] else 0,
                "free_bits":     CFG["free_bits"],
                "use_mel":       CFG["USE_MEL"],
            },
        }
        save_checkpoint(epoch_state, CFG["checkpoint"])
        notify_crash_save(epoch + 1, CFG["checkpoint"])

        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(epoch_state, CFG["best_checkpoint"])
            print(f"  [★] New best val loss: {best_val:.4f} → saved to {CFG['best_checkpoint']}")

        notify_epoch(
            epoch + 1,
            CFG["num_epochs"],
            d_loss=avg_recon,
            g_loss=avg_val,
        )

        # FIX 4: Early stopping check
        if early_stopper.step(avg_val):
            print(f"\n[EarlyStop] Triggered at epoch {epoch+1}. Best val: {best_val:.4f}")
            break

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