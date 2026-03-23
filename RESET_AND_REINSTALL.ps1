##############################################################
# Urban-GenX | COMPLETE RESET & REINSTALL GUIDE
# Run these commands in PowerShell as Administrator if needed
# Working directory: E:\Urban-GenX (your project root)
##############################################################

# ─── STEP 0: Navigate to your project root ────────────────
cd E:\Urban-GenX

# ─── STEP 1: Deactivate and remove old environment ────────
conda deactivate
conda env remove -n urban-genx -y

# Verify it is gone:
conda env list

# ─── STEP 2: Create fresh environment ─────────────────────
conda create -n urban-genx python=3.10 -y
conda activate urban-genx

# Verify Python:
python --version
# Expected: Python 3.10.x

# ─── STEP 3: Update pip & setuptools first ────────────────
python -m pip install --upgrade pip setuptools wheel packaging

# ─── STEP 4: Install requirements ─────────────────────────
# Replace requirements.txt with the FINAL FIXED version first, then:
pip install -r requirements.txt

# IMPORTANT: If torch takes too long from PyPI, use:
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# ─── STEP 5: Verify critical imports ──────────────────────
python -c "import torch; print('torch', torch.__version__)"
python -c "import flwr; print('flwr', flwr.__version__)"
python -c "import opacus; print('opacus', opacus.__version__)"
python -c "import librosa; print('librosa', librosa.__version__)"
python -c "import streamlit; print('streamlit', streamlit.__version__)"
python -c "from sentence_transformers import SentenceTransformer; print('SBERT OK')"
python -c "import ray; print('ray', ray.__version__)"
python -c "import dataretrieval; print('dataretrieval OK')"

# ─── STEP 6: Create required folders ─────────────────────
New-Item -ItemType Directory -Force -Path data\raw\cityscapes, data\raw\urbansound8k, data\raw\metr-la, data\raw\usgs_water, checkpoints | Out-Null
Write-Host "[OK] Folder structure created."

# ─── STEP 7: Verify datasets ──────────────────────────────
# UrbanSound8K (should already be in place):
Test-Path data\raw\urbansound8k\metadata\UrbanSound8K.csv
Test-Path data\raw\urbansound8k\audio\fold1

# METR-LA:
Test-Path data\raw\metr-la\metr-la.h5

# Cityscapes (manual download required):
Test-Path data\raw\cityscapes\leftImg8bit\train
Test-Path data\raw\cityscapes\gtFine\train

# ─── STEP 8: Run sanity tests (no training needed) ────────
python tests\test_models.py
python tests\test_data_loaders.py

# ─── STEP 9: Train (in order) ─────────────────────────────
# Run these in SEPARATE terminals (do not run Vision alongside others):

# Terminal 1 — Traffic (fastest, ~35 min):
python src\training\train_utility.py

# Terminal 2 — Acoustic (~2-4 hrs):
python src\training\train_acoustic.py

# Terminal 3 — Vision DP-GAN (ALONE, requires Cityscapes, ~8-12 hrs):
# Delete any corrupted checkpoint first:
Remove-Item -ErrorAction SilentlyContinue checkpoints\vision_checkpoint.pth
python src\training\train_vision.py

# ─── STEP 10: Launch dashboard ────────────────────────────
streamlit run dashboard\app.py
# Opens at: http://localhost:8501

# ─── STEP 11: Federated Learning demo ────────────────────
# (Run after at least acoustic and traffic checkpoints exist)
python src\federated\server.py
# Or modality-specific:
python src\federated\server.py --acoustic
python src\federated\server.py --vision

# ─── STEP 12: Full validation ─────────────────────────────
python tests\check_all.py
# All 15 checks should PASS or WARN.
# No FAIL items before submission.

##############################################################
# COMMON ERRORS & FIXES
##############################################################

# Error: sentence-transformers import fails → keyword fallback active
#   Fix: pip install sentence-transformers==2.7.0 --upgrade

# Error: Opacus "Poisson sampling not compatible"
#   Fix: train_vision.py uses disable_hooks() during G step — already fixed.

# Error: Opacus PRV numerical crash on resume
#   Fix: train_vision.py uses accountant='rdp' — already fixed.
#   Also: delete checkpoint before resuming if NaN losses observed.

# Error: Vision checkpoint has NaN d_loss/g_loss
#   Fix: Remove-Item checkpoints\vision_checkpoint.pth
#        python src\training\train_vision.py

# Error: Flower "shape mismatch" during aggregation
#   Fix: server.py now runs separate simulations per modality — already fixed.

# Error: num_examples = 0 in Flower logs (wrong weighting)
#   Fix: clients now return len(dataset), not len(dataloader) — already fixed.

# Error: AcousticVAE loss ~50,000 (very large)
#   Fix: acoustic_vae.py uses reduction='mean' — already fixed.
#        If old checkpoint was trained with 'sum', delete it and retrain.

# Error: UrbanSound8K metadata not found
#   Fix: ensure data/raw/urbansound8k/metadata/UrbanSound8K.csv exists
#        New-Item -ItemType Directory -Force data\raw\urbansound8k\metadata, data\raw\urbansound8k\audio
#        Move-Item data\raw\urbansound8k\UrbanSound8K.csv data\raw\urbansound8k\metadata\

# Error: streamlit run fails (ran as 'python dashboard\app.py')
#   Fix: ALWAYS use: streamlit run dashboard\app.py
#        NOT: python dashboard\app.py

# Error: n_fft=2048 too large warnings (librosa)
#   Fix: data_loader.py already uses n_fft=1024 and pads short clips.
#        These are just INFO warnings, not errors — training continues.
