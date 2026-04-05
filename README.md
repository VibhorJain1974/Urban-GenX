# Urban-GenX 🏙️
**Privacy-Preserving Synthetic City Digital Twin for SDG 11 Research**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Differential Privacy](https://img.shields.io/badge/Differential%20Privacy-%CE%B5%20%E2%89%A4%2010.0-red)](https://opacus.ai)
[![SDG 11](https://img.shields.io/badge/SDG-11%20Sustainable%20Cities%20%26%20Communities-2e7d32)](https://sdgs.un.org/goals/goal11)
[![Federated Learning](https://img.shields.io/badge/FL-Flower%20FedAvg-purple)](https://flower.dev)

Urban-GenX bridges the **privacy–utility gap** in urban data science by combining:
- 🔒 **Differential Privacy** (Opacus DP-SGD, ε ≤ 10.0, δ = 1e-5) on the Vision cGAN Discriminator
- 🌐 **Federated Learning** (Flower FedAvg, 2 clients) — data never leaves the node
- 🎨 **Multi-modal synthesis** — street-view images, urban soundscapes, traffic forecasts
- 💬 **Natural-language interface** (SBERT semantic query → urban scene preset)
- 📊 **SDG 11 dashboard** — counterfactual sliders, radar chart, membership-inference audit

---

## Directory Structure

```
Urban-GenX/
├── data/raw/                    # Datasets (NOT in Git)
│   ├── cityscapes/              # leftImg8bit/ + gtFine/
│   ├── urbansound8k/            # metadata/ + audio/
│   ├── metr-la/                 # metr-la.h5
│   └── usgs_water/              # water_quality.csv
├── models/
│   ├── vision_gan.py            # UNet cGAN (Generator + PatchGAN Discriminator)
│   ├── acoustic_vae.py          # Beta-VAE for MFCC spectrograms
│   ├── utility_vae.py           # FC-VAE for traffic/water time-series
│   └── transformer_core.py     # SBERT semantic interface (8 urban presets)
├── src/
│   ├── training/
│   │   ├── train_vision.py      # DP-GAN training (RDP accountant, D-freeze)
│   │   ├── train_acoustic.py    # Acoustic VAE training (mean-reduction loss)
│   │   └── train_utility.py     # Traffic/water VAE training
│   ├── federated/
│   │   ├── server.py            # Flower FedAvg simulation server
│   │   ├── client_vision.py     # Vision FL client (D federated, G local)
│   │   └── client_acoustic.py  # Acoustic FL client (all VAE params federated)
│   └── utils/
│       ├── data_loader.py       # Cityscapes, UrbanSound8K, METR-LA loaders
│       ├── notifier.py          # ntfy.sh mobile alerts
│       ├── privacy_audit.py     # Shadow-model membership inference attack
│       ├── download_datasets.py # Automated dataset downloader
│       └── download_water_data.py
├── checkpoints/                 # Model .pth files (NOT in Git)
├── dashboard/
│   ├── app.py                   # Streamlit multi-modal UI
│   └── README.md
├── tests/
│   └── check_all.py             # Sanity-check suite
├── environment.yml
└── requirements.txt
```

---

## Quick Start

### 1. Environment Setup

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate urban-genx

# Option B: pip
pip install -r requirements.txt
pip install "setuptools<81"   # silence pkg_resources deprecation warning
```

### 2. Dataset Setup

#### UrbanSound8K (manual Kaggle download)
```powershell
# After downloading from https://www.kaggle.com/datasets/chrisfilo/urbansound8k
# Extract and move to correct layout:
New-Item -ItemType Directory -Force -Path data\raw\urbansound8k\metadata, data\raw\urbansound8k\audio
Move-Item data\raw\urbansound8k\UrbanSound8K.csv data\raw\urbansound8k\metadata\UrbanSound8K.csv
Get-ChildItem data\raw\urbansound8k -Directory -Filter "fold*" | Move-Item -Destination data\raw\urbansound8k\audio

# Verify:
Test-Path data\raw\urbansound8k\metadata\UrbanSound8K.csv   # True
Test-Path data\raw\urbansound8k\audio\fold1                  # True
```

#### METR-LA Traffic
```powershell
# Download from https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset
# Place the .h5 file at:
New-Item -ItemType Directory -Force data\raw\metr-la
# → data\raw\metr-la\metr-la.h5
```

#### Cityscapes (requires free registration)
1. Register at https://www.cityscapes-dataset.com/register/
2. Download `leftImg8bit_trainvaltest.zip` (~11 GB) and `gtFine_trainvaltest.zip` (~241 MB)
3. Extract both into `data/raw/cityscapes/`

Expected layout:
```
data/raw/cityscapes/
├── leftImg8bit/train/<city>/*.png
└── gtFine/train/<city>/*_gtFine_labelIds.png
```

#### USGS Water Quality (automated)
```bash
python src/utils/download_water_data.py
# → data/raw/usgs_water/water_quality.csv
```

---

## Training

> ⚠️ CPU-only, 12 GB RAM. Run Acoustic + Utility in parallel; run Vision alone.

### Verify datasets first
```powershell
python -c "from src.utils.data_loader import UrbanSound8KDataset; d=UrbanSound8KDataset('data/raw/urbansound8k'); print('Acoustic OK:', len(d), 'samples'); x,y=d[0]; print('shape', x.shape)"
python -c "from src.utils.data_loader import METRLADataset; d=METRLADataset('data/raw/metr-la/metr-la.h5'); print('Traffic OK:', len(d), 'samples')"
python -c "from src.utils.data_loader import CityscapesDataset; d=CityscapesDataset('data/raw/cityscapes'); print('Vision OK:', len(d), 'samples')"
```

### Step 1 — Train Acoustic VAE (Terminal A)
```powershell
python src\training\train_acoustic.py
# Checkpoints: checkpoints/acoustic_checkpoint.pth
#              checkpoints/acoustic_best.pth (best val loss)
# Expected: Train loss ~0.3–1.5 (mean reduction, NOT 50k)
# Duration: ~2–4 hours on i5 CPU
```

### Step 2 — Train Utility/Traffic VAE (Terminal B, simultaneously with Step 1)
```powershell
python src\training\train_utility.py
# Checkpoint: checkpoints/utility_traffic_checkpoint.pth
# Status: ✅ Already completed (epoch 50, val loss ~181.74)
```

### Step 3 — Train Vision DP-GAN (alone, after Steps 1+2 complete)
```powershell
# First, clear any corrupted checkpoint:
Remove-Item -ErrorAction SilentlyContinue checkpoints\vision_checkpoint.pth

python src\training\train_vision.py
# Checkpoint: checkpoints/vision_checkpoint.pth
# DP: RDP accountant, ε ≤ 10.0, δ = 1e-5
# Duration: ~8–16 hours on i5 CPU
# Monitor: [DP] ε spent: x.xxxx / 10.0 (printed each epoch)
```

---

## Dashboard

```powershell
streamlit run dashboard\app.py
# Opens http://localhost:8501
# Requires: acoustic_checkpoint.pth + utility_traffic_checkpoint.pth (minimum)
# Full demo: all 3 checkpoints
```

Features:
- Natural-language scene query (SBERT or keyword fallback)
- Counterfactual policy sliders (noise, traffic density, green space, time of day)
- Vision tab: synthetic 64×64 street-view images
- Acoustic tab: MFCC soundscape generation
- Traffic tab: 207-sensor speed heatmap (12-step horizon)
- Privacy tab: DP-SGD budget (ε ≤ 10.0) + MIA AUC ≈ 0.54
- SDG 11 tab: radar chart of Transport, Air/Noise, Green Space scores

---

## Federated Learning Simulation

> Run AFTER all checkpoints are available.

```powershell
python src\federated\server.py
# Runs 5 FL rounds (in-process, no network required)
# Vision: discriminator weights federated, generator stays local
# Acoustic: all VAE parameters federated
# RAM limit: 1 CPU / 12 GB safe (ray_init_args)
```

---

## Privacy & DP Compliance

| Parameter | Value |
|-----------|-------|
| Mechanism | DP-SGD (Opacus) |
| Accountant | RDP (stable on Windows CPU) |
| Target ε | 10.0 |
| Target δ | 1e-5 |
| Clipping C | 1.0 |
| Applied to | Discriminator only |
| λ_L1 | 0.0 (DP-safe for releasing G weights) |
| MIA AUC | ~0.54 (near-random = private) |

**Why D-only DP?** The Generator learns only from gradients passing through D (never touches raw data). Setting `lambda_l1 = 0.0` eliminates direct image pixel access by G, making G weights safe to release under DP guarantees.

---

## Sanity Tests

```powershell
python tests\check_all.py
# Runs 8 checks: data loaders, model forward passes,
# checkpoint loading, semantic interface, DP attachment
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: UrbanSound8K.csv` | Check `data/raw/urbansound8k/metadata/UrbanSound8K.csv` exists |
| `librosa n_fft too large` | Fixed in data_loader.py (n_fft=1024, pad short clips) |
| `pkg_resources deprecated` | `pip install "setuptools<81"` |
| Opacus `GradSampleModule` error | Ensure `disable_hooks()` before G-step (already in train_vision.py) |
| Vision checkpoint NaN | Delete checkpoint, retrain from scratch |
| `sentence-transformers not found` | `pip install sentence-transformers==2.2.2`; fallback to keyword matching is OK |
| `num_workers` DataLoader freeze | Always use `num_workers=0` on Windows |
| FL `min_available_clients` error | Run `python src/federated/server.py` (not client scripts directly) |

---

## Citation & Credits

```
Urban-GenX: Privacy-Preserving Synthetic City Digital Twin
Vibhor Jain, 2024
Federated Learning: Flower (flwr 1.5.0)
Differential Privacy: Opacus (1.4.0)
Datasets: Cityscapes, UrbanSound8K, METR-LA, USGS Water Quality
```
