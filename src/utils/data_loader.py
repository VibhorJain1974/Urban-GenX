"""
Urban-GenX | Data Loaders — FIXED (Mel-spectrogram support added)
=================================================================
CHANGE vs original:
  UrbanSound8KDataset now accepts:
    - use_mel=True/False flag
    - n_mels parameter (number of Mel bins)
  
  When use_mel=True:
    - Uses librosa.feature.melspectrogram instead of mfcc
    - Converts to log scale (log1p) for better dynamic range
    - Returns [B, 1, n_mels, time_frames] — same shape contract as before
    - n_mels=64 recommended (vs 40 MFCC bins) for richer features

  All other datasets (Cityscapes, METR-LA, WaterQuality) unchanged.
"""

import os
import torch
import numpy as np
import librosa
import pandas as pd
import h5py
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# ─── Vision Dataset ───────────────────────────────────────────────────────────
class CityscapesDataset(Dataset):
    """
    Loads Cityscapes RGB + label pairs.
    PIL Fix: explicit convert('RGB') prevents RGBA/palette mode errors.
    Memory: images resized to 64×64. Normalization: [-1, 1] for GAN training.
    """
    NUM_CLASSES = 35

    def __init__(self, root, split='train', img_size=64):
        self.img_dir  = os.path.join(root, 'leftImg8bit',  split)
        self.lbl_dir  = os.path.join(root, 'gtFine',       split)
        self.img_size = img_size

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.lbl_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
        ])

        self.samples = []
        for city in os.listdir(self.img_dir):
            img_city = os.path.join(self.img_dir, city)
            lbl_city = os.path.join(self.lbl_dir, city)
            if not os.path.isdir(img_city):
                continue
            for fname in os.listdir(img_city):
                if not fname.endswith('_leftImg8bit.png'):
                    continue
                stem = fname.replace('_leftImg8bit.png', '')
                lbl_name = f"{stem}_gtFine_labelIds.png"
                lbl_path = os.path.join(lbl_city, lbl_name)
                if os.path.exists(lbl_path):
                    self.samples.append((
                        os.path.join(img_city, fname),
                        lbl_path
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)
        img_t = self.img_transform(img)
        lbl_r = self.lbl_transform(lbl)
        lbl_np = np.array(lbl_r, dtype=np.int64)
        lbl_np = np.clip(lbl_np, 0, self.NUM_CLASSES - 1)
        lbl_t  = torch.from_numpy(lbl_np)
        cond = torch.zeros(self.NUM_CLASSES, self.img_size, self.img_size)
        cond.scatter_(0, lbl_t.unsqueeze(0), 1.0)
        return img_t, cond


# ─── Acoustic Dataset ─────────────────────────────────────────────────────────
class UrbanSound8KDataset(Dataset):
    """
    Loads UrbanSound8K audio and converts to MFCC or Mel-spectrogram.

    FIXED: Added use_mel flag and n_mels parameter.
    
    Args:
        root        : path to UrbanSound8K (must contain metadata/ and audio/)
        folds       : list of fold numbers to include (e.g. [1..9] for train)
        n_mfcc      : MFCC bins (used when use_mel=False)
        n_mels      : Mel filter banks (used when use_mel=True) — default 64
        use_mel     : if True, use Mel-spectrogram instead of MFCC
        time_frames : fixed time axis (padded/truncated to this)
        sr          : sample rate (22050 Hz standard)

    Returns per item:
        tensor : [1, n_bins, time_frames]  (n_bins = n_mfcc or n_mels)
        label  : int class ID (0–9)
    """

    def __init__(
        self,
        root,
        folds=None,
        n_mfcc=40,
        n_mels=64,          # NEW: Mel bins
        use_mel=True,       # NEW: use Mel-spectrogram (recommended)
        time_frames=128,
        sr=22050,
    ):
        self.root        = root
        self.folds       = folds or list(range(1, 11))
        self.n_mfcc      = n_mfcc
        self.n_mels      = n_mels
        self.use_mel     = use_mel
        self.time_frames = time_frames
        self.sr          = sr

        # Number of output frequency bins
        self.n_bins = n_mels if use_mel else n_mfcc

        mode = "Mel-spectrogram" if use_mel else "MFCC"
        print(f"[Dataset] Mode: {mode} | Bins: {self.n_bins} | Time frames: {time_frames}")

        self.samples = []
        meta = pd.read_csv(os.path.join(root, 'metadata', 'UrbanSound8K.csv'))
        for _, row in meta.iterrows():
            if row['fold'] in self.folds:
                fp = os.path.join(
                    root, 'audio', f"fold{row['fold']}", row['slice_file_name']
                )
                if os.path.exists(fp):
                    self.samples.append((fp, int(row['classID'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            if len(y) < 1024:
                y = np.pad(y, (0, 1024 - len(y)))

            if self.use_mel:
                # ── Mel-spectrogram (FIX 1 — richer than MFCC) ──────────────
                mel = librosa.feature.melspectrogram(
                    y=y,
                    sr=self.sr,
                    n_mels=self.n_mels,
                    n_fft=1024,
                    hop_length=512,
                    fmax=8000,      # limit to 8kHz (most urban sounds)
                )
                # Log scale: compresses dynamic range, makes patterns clearer
                features = librosa.power_to_db(mel, ref=np.max)
            else:
                # ── Original MFCC (kept for backwards compatibility) ──────────
                features = librosa.feature.mfcc(
                    y=y,
                    sr=self.sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=1024,
                    hop_length=512,
                )

        except Exception:
            # Return zeros on corrupt/unreadable file
            features = np.zeros((self.n_bins, self.time_frames), dtype=np.float32)

        # ── Pad / truncate to fixed time_frames ──────────────────────────────
        if features.shape[1] < self.time_frames:
            features = np.pad(
                features,
                ((0, 0), (0, self.time_frames - features.shape[1]))
            )
        else:
            features = features[:, :self.time_frames]

        # ── Normalize to zero mean, unit std ─────────────────────────────────
        features = (features - features.mean()) / (features.std() + 1e-8)

        return (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0),  # [1, bins, T]
            label
        )


# ─── Traffic Dataset (METR-LA) — UNCHANGED ───────────────────────────────────
class METRLADataset(Dataset):
    """
    Loads METR-LA traffic speed data from HDF5.
    Creates sliding window sequences for VAE/forecasting.
    """
    def __init__(self, h5_path, seq_len=12, pred_len=12):
        with h5py.File(h5_path, 'r') as f:
            data = f['df']['block0_values'][:]
        self.data     = torch.tensor(data, dtype=torch.float32)
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.mean = self.data.mean()
        self.std  = self.data.std() + 1e-8
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx           : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y


# ─── Water Quality Dataset (USGS) — UNCHANGED ────────────────────────────────
class WaterQualityDataset(Dataset):
    """
    Loads USGS water quality CSV and creates sliding-window sequences.
    """
    PARAM_PATTERNS = [
        "oxygen", "do", "00300",
        "ph", "00400",
        "temp", "00010",
        "turb", "63680",
        "flow", "streamflow", "00060",
        "conductiv", "00095",
        "nitro", "00631",
    ]

    def __init__(self, csv_path, seq_len=24, n_params=5, stride=1):
        self.seq_len  = seq_len
        self.n_params = n_params
        self.stride   = stride

        df = pd.read_csv(csv_path, low_memory=False)
        print(f"[WaterQuality] Loaded CSV: {len(df)} rows, {len(df.columns)} columns")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        matched_cols = []
        for col in numeric_cols:
            col_lower = col.lower()
            for pattern in self.PARAM_PATTERNS:
                if pattern in col_lower:
                    matched_cols.append(col)
                    break

        if len(matched_cols) < 2:
            matched_cols = numeric_cols[:n_params]
        else:
            matched_cols = matched_cols[:n_params]

        if len(matched_cols) < n_params:
            remaining = [c for c in numeric_cols if c not in matched_cols]
            matched_cols.extend(remaining[:n_params - len(matched_cols)])

        self.param_names = matched_cols[:n_params]
        actual_n = len(self.param_names)
        print(f"[WaterQuality] Using {actual_n} parameters: {self.param_names}")

        if actual_n == 0:
            raise ValueError(
                f"No numeric columns found in {csv_path}. "
                f"Available columns: {list(df.columns[:20])}"
            )

        data = df[self.param_names].copy()
        data = data.ffill().bfill().fillna(0.0)
        values = data.values.astype(np.float32)
        self.raw_data = torch.tensor(values, dtype=torch.float32)
        self.mean = self.raw_data.mean(dim=0, keepdim=True)
        self.std  = self.raw_data.std(dim=0, keepdim=True) + 1e-8
        self.data = (self.raw_data - self.mean) / self.std
        self.actual_n_params = actual_n

        n_samples = max(0, (len(self.data) - seq_len) // stride)
        print(f"[WaterQuality] {n_samples} sliding-window samples (seq_len={seq_len}, stride={stride})")

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len) // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        end   = start + self.seq_len
        x = self.data[start:end]
        return x, x

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean