"""
Urban-GenX | Data Loaders
Handles: Cityscapes (PIL→Tensor fix), UrbanSound8K (MFCC), METR-LA, USGS
Critical: All Cityscapes images resized to 64x64 for 12GB RAM safety.
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
            T.ToTensor(),                          # [0,1]
            T.Normalize([0.5]*3, [0.5]*3)          # → [-1, 1]
        ])
        self.lbl_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            # Do NOT use ToTensor here — label maps need LongTensor
        ])

        # Collect all image paths
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

        # Critical PIL fix: always convert to RGB (avoids RGBA/P mode errors)
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)  # Label IDs, single channel

        img_t = self.img_transform(img)

        # Label → one-hot condition tensor
        lbl_r = self.lbl_transform(lbl)
        lbl_np = np.array(lbl_r, dtype=np.int64)
        lbl_np = np.clip(lbl_np, 0, self.NUM_CLASSES - 1)
        lbl_t  = torch.from_numpy(lbl_np)                    # [H, W]
        # One-hot encode: [H, W] → [C, H, W]
        cond = torch.zeros(self.NUM_CLASSES, self.img_size, self.img_size)
        cond.scatter_(0, lbl_t.unsqueeze(0), 1.0)

        return img_t, cond   # (real_image, condition)


# ─── Acoustic Dataset ─────────────────────────────────────────────────────────
class UrbanSound8KDataset(Dataset):
    """
    Loads UrbanSound8K audio files and converts to MFCC spectrograms.
    Pads/truncates to fixed time_frames for batch consistency.
    """
    def __init__(self, root, folds=None, n_mfcc=40, time_frames=128, sr=22050):
        self.root        = root
        self.folds       = folds or list(range(1, 11))
        self.n_mfcc      = n_mfcc
        self.time_frames = time_frames
        self.sr          = sr

        self.samples = []
        meta = pd.read_csv(os.path.join(root, 'metadata', 'UrbanSound8K.csv'))
        for _, row in meta.iterrows():
            if row['fold'] in self.folds:
                fp = os.path.join(root, 'audio', f"fold{row['fold']}", row['slice_file_name'])
                if os.path.exists(fp):
                    self.samples.append((fp, int(row['classID'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        except Exception:
            # Return zeros on corrupt files (UrbanSound8K has a few)
            mfcc = np.zeros((self.n_mfcc, self.time_frames), dtype=np.float32)

        # Pad or truncate to fixed time_frames
        if mfcc.shape[1] < self.time_frames:
            mfcc = np.pad(mfcc, ((0,0), (0, self.time_frames - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :self.time_frames]

        # Normalize to [-1, 1]
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), label


# ─── Traffic Dataset (METR-LA) ────────────────────────────────────────────────
class METRLADataset(Dataset):
    """
    Loads METR-LA traffic speed data from HDF5.
    Creates sliding window sequences for VAE/forecasting.
    """
    def __init__(self, h5_path, seq_len=12, pred_len=12):
        with h5py.File(h5_path, 'r') as f:
            data = f['df']['block0_values'][:]  # [T, N_sensors]
        self.data     = torch.tensor(data, dtype=torch.float32)
        self.seq_len  = seq_len
        self.pred_len = pred_len
        # Normalize
        self.mean = self.data.mean()
        self.std  = self.data.std() + 1e-8
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx           : idx + self.seq_len]
        y = self.data[idx+self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y
