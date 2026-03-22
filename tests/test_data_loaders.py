"""
Urban-GenX | tests/test_data_loaders.py
Data loader unit tests (skip gracefully when dataset absent).
Run: python tests/test_data_loaders.py
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_urbansound8k_loader():
    """Test UrbanSound8K MFCC loader: shape, label range, no NaN."""
    data_root = ROOT / "data" / "raw" / "urbansound8k"
    csv       = data_root / "metadata" / "UrbanSound8K.csv"
    alt_csv   = data_root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"

    if not csv.exists() and not alt_csv.exists():
        print("[SKIP] test_urbansound8k_loader – dataset not installed")
        return

    from src.utils.data_loader import UrbanSound8KDataset
    ds  = UrbanSound8KDataset(str(data_root), folds=[1])   # fold 1 only for speed
    assert len(ds) > 0, "No samples found in fold 1"

    x, label = ds[0]
    assert x.shape  == (1, 40, 128),  f"Wrong MFCC shape: {x.shape}"
    assert 0 <= int(label) < 10,      f"Class label out of range: {label}"
    assert not torch.isnan(x).any(),  "NaN in MFCC tensor"
    assert x.dtype == torch.float32,  f"Wrong dtype: {x.dtype}"

    # Batch test
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, num_workers=0)
    batch_x, batch_y = next(iter(dl))
    assert batch_x.shape[0] == min(8, len(ds))
    assert batch_x.shape[1:] == (1, 40, 128)

    print(f"[PASS] test_urbansound8k_loader | {len(ds)} samples | MFCC shape OK")


def test_metrla_loader():
    """Test METR-LA sliding-window sequence shape and normalization."""
    h5_path = ROOT / "data" / "raw" / "metr-la" / "metr-la.h5"
    if not h5_path.exists():
        print("[SKIP] test_metrla_loader – metr-la.h5 not found")
        return

    from src.utils.data_loader import METRLADataset
    ds = METRLADataset(str(h5_path), seq_len=12, pred_len=12)
    assert len(ds) > 1000, f"Too few windows: {len(ds)}"

    x, y = ds[0]
    assert x.shape == (12, 207), f"Wrong x shape: {x.shape}"
    assert y.shape == (12, 207), f"Wrong y shape: {y.shape}"
    assert not torch.isnan(x).any(), "NaN in traffic x tensor"
    assert not torch.isnan(y).any(), "NaN in traffic y tensor"

    # Check normalized (mean ~0, std ~1)
    flat = x.flatten()
    assert abs(flat.mean().item()) < 5.0, f"Mean far from 0 after normalization: {flat.mean().item()}"

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=32, num_workers=0)
    bx, by = next(iter(dl))
    assert bx.shape == (32, 12, 207)
    assert by.shape == (32, 12, 207)

    print(f"[PASS] test_metrla_loader | {len(ds)} windows | x={tuple(x.shape)} OK")


def test_cityscapes_loader():
    """Test Cityscapes image+condition loader: shape, range, one-hot encoding."""
    left_path = ROOT / "data" / "raw" / "cityscapes" / "leftImg8bit" / "train"
    if not left_path.exists():
        print("[SKIP] test_cityscapes_loader – Cityscapes not installed")
        return

    from src.utils.data_loader import CityscapesDataset
    ds = CityscapesDataset(str(ROOT / "data" / "raw" / "cityscapes"), split="train")
    assert len(ds) > 100, f"Too few Cityscapes images: {len(ds)}"

    img, cond = ds[0]
    assert img.shape  == (3, 64, 64),  f"Wrong img shape: {img.shape}"
    assert cond.shape == (35, 64, 64), f"Wrong cond shape: {cond.shape}"

    # Image normalized to [-1, 1]
    assert img.min() >= -1.1 and img.max() <= 1.1, \
        f"Image not in [-1,1]: [{img.min():.3f}, {img.max():.3f}]"

    # Condition is one-hot: each pixel has exactly one active class
    class_counts = cond.sum(dim=0)          # [64, 64]
    assert (class_counts == 1.0).all(), \
        "Condition map is not one-hot (pixel class counts != 1)"

    print(f"[PASS] test_cityscapes_loader | {len(ds)} samples | img/cond shapes OK | one-hot OK")


if __name__ == "__main__":
    test_urbansound8k_loader()
    test_metrla_loader()
    test_cityscapes_loader()
    print("\n[ALL LOADER TESTS DONE]")
