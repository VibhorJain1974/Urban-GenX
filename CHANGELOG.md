# Urban-GenX Changelog

## [3.0.0] — Final Production Release (2026-04-05)
### Added
- **DP-SGD for Acoustic VAE**: `src/training/train_acoustic_dp.py` (ε ≤ 10.0)
- **DP-SGD for Traffic/Water VAE**: `src/training/train_utility_dp.py --mode traffic|water`
- **Beautiful dark-theme dashboard** with custom CSS, consistent color palette
- **Dockerfile + docker-compose.yml** for one-command deployment
- **GitHub Copilot prompt** for automated project completion

### Fixed
- Dashboard: all Streamlit deprecation warnings resolved
- Dashboard: Water tab f-string concatenation bug (ternary expression)
- Dashboard: matplotlib dark theme matching Streamlit dark background
- Charts: consistent color palette across all tabs

### Changed
- Privacy tab: updated to show DP status for ALL models
- SDG 11 radar: improved styling with scatter points and dark background

## [2.0.0] — Phase 1 Complete (2026-04-04)
### Completed
- Vision cGAN: 50 epochs, DP ε = 9.93
- Acoustic VAE: 50 epochs
- Traffic VAE: 50 epochs
- Federated Learning: Vision + Acoustic (3 rounds each)
- Semantic Interface: 8 scene presets via SBERT
- MIA audit: AUC 0.42–0.52 (SAFE)
- Dashboard: 6 tabs working

## [1.0.0] — Foundation (2026-03)
### Initial
- Project structure, model architectures, data loaders
- requirements.txt + environment.yml
