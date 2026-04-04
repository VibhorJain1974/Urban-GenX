# Urban-GenX Changelog

## [2.0.0] — Final Phase (2026-04-05)

### Added
- **Water Quality (4th modality)**: `WaterQualityDataset` in `data_loader.py` for USGS CSV files
- **Water VAE training**: `train_utility.py` now supports `mode="water"` for USGS water quality
- **Water Quality dashboard tab**: 💧 tab showing synthetic water parameter time-series
- **Standalone MIA audit**: `privacy_audit.py` is now a runnable CLI script (`python src/utils/privacy_audit.py --model all`)
- **Docker support**: `Dockerfile` + `docker-compose.yml` for reproducible deployment
- **CHANGELOG.md**: This file

### Fixed
- **Streamlit deprecation warnings**: Replaced all `use_column_width=True` with `width="stretch"` and `use_container_width=True` with `width="stretch"`
- **Dashboard tab count**: Extended from 5 to 6 tabs (added Water Quality)

### Changed
- **Dashboard**: Updated footer to show "4 Modalities (Vision + Acoustic + Traffic + Water)"
- **Privacy audit**: Now shows member vs non-member mean loss comparison + accuracy metric
- **train_utility.py**: Clean implementation of water mode with auto-detection of CSV columns

---

## [1.5.0] — Phase 1 Complete (2026-04-04)

### Added
- **Vision cGAN**: 50 epochs trained with DP-SGD (ε=9.93, δ=1e-5)
- **Acoustic VAE**: 50 epochs trained with beta-annealing, fold-10 validation
- **Traffic VAE**: 50 epochs trained on METR-LA (207 sensors × 12 steps)
- **Semantic Interface**: `transformer_core.py` with SBERT + 8 scene presets
- **Federated Learning**: Vision + Acoustic FL simulation via Flower FedAvg
- **Privacy Audit**: MIA showing AUC ~0.54 (privacy confirmed)
- **Streamlit Dashboard**: 5-tab UI with counterfactual sliders
- **ntfy.sh integration**: Mobile alerts for training progress
- **Crash recovery**: Epoch checkpointing with validation

### Fixed
- Opacus "Poisson sampling grad accumulation" error → disable_hooks() during G step
- Opacus "activations.pop empty list" error → disable_hooks() during G step
- GAN NaN instability → label smoothing + logit clamping + NaN-skip
- PIL RGBA/P mode crash → `.convert('RGB')` fix
- Flower FedAvg weighted aggregation → `len(dataset)` not `len(dataloader)`
- UrbanSound8K MFCC crash on short audio → padding to 1024 samples minimum
- METR-LA h5 path auto-detection for different download layouts

---

## [1.0.0] — Foundation (2026-03)

### Added
- Initial project structure
- Vision GAN architecture (UNet Generator + PatchGAN Discriminator)
- Acoustic VAE architecture
- Cityscapes + UrbanSound8K data loaders
- Basic training scripts
- requirements.txt + environment.yml
