# Urban-GenX Changelog

## [2.1.0] — Production Fix (2026-04-05)

### Critical Bug Fixes
- **`.gitignore` was PowerShell script, not a gitignore file** — replaced with proper syntax
- **`train_vision.py` checkpoint save called `opt_d.step()`** — caused an extra optimizer step every epoch during save. Fixed to `opt_d.state_dict()`.
- **Water data download only returned 1 parameter** (discharge) — rewrote `download_water_data.py` to use daily values service with multi-parameter sites, plus synthetic fallback
- **Dashboard hardcoded `n_params=5` for water model** — now auto-detects from checkpoint, preventing shape mismatch errors

### Enhancements
- Docker support (Dockerfile + docker-compose.yml)
- Water training launcher script (`scripts/train_water.py`)
- This CHANGELOG

## [2.0.0] — Final Phase (2026-04-04)

### Added
- 4th modality: Water Quality VAE (USGS data)
- WaterQualityDataset in data_loader.py
- Standalone MIA privacy audit CLI (`python src/utils/privacy_audit.py --model all`)
- Streamlit deprecation fixes (use_column_width → width="stretch")
- Unit tests for water modality

### Training Results
- Vision GAN: 50 epochs, DP ε = 9.93 (under 10.0 budget)
- Acoustic VAE: 50 epochs, best val = 0.62
- Traffic VAE: 50 epochs
- Water VAE: 50 epochs
- MIA Audit: Acoustic AUC=0.42 (SAFE), Traffic AUC=0.52 (SAFE)

## [1.0.0] — Phase 1 (2026-03)

### Added
- Vision cGAN (64×64, Cityscapes)
- Acoustic VAE (MFCC, UrbanSound8K)
- Traffic VAE (METR-LA)
- Federated Learning (Flower FedAvg, 2 clients)
- DP-SGD on Vision Discriminator (Opacus)
- Streamlit dashboard with 5 tabs
- Semantic interface (Sentence-BERT)
- ntfy.sh mobile notifications
- Epoch checkpointing with crash recovery

## [0.1.0] — Foundation

### Added
- Project structure
- Model architectures
- Data loaders
- requirements.txt + environment.yml
