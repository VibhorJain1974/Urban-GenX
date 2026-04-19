# Graph Report - .  (2026-04-18)

## Corpus Check
- 40 files · ~24,470 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 330 nodes · 466 edges · 31 communities detected
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 93 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Utility Training Pipeline|Utility Training Pipeline]]
- [[_COMMUNITY_Dashboard & UI Layer|Dashboard & UI Layer]]
- [[_COMMUNITY_Data Loading & Preprocessing|Data Loading & Preprocessing]]
- [[_COMMUNITY_Core Generative Models|Core Generative Models]]
- [[_COMMUNITY_Semantic Interface|Semantic Interface]]
- [[_COMMUNITY_Vision Generator|Vision Generator]]
- [[_COMMUNITY_Acoustic FL Client|Acoustic FL Client]]
- [[_COMMUNITY_Vision FL Client|Vision FL Client]]
- [[_COMMUNITY_Acoustic VAE|Acoustic VAE]]
- [[_COMMUNITY_Privacy Auditing|Privacy Auditing]]
- [[_COMMUNITY_Federated Server|Federated Server]]
- [[_COMMUNITY_Testing & Validation|Testing & Validation]]
- [[_COMMUNITY_Dataset Management|Dataset Management]]
- [[_COMMUNITY_Water Data Pipeline|Water Data Pipeline]]
- [[_COMMUNITY_Project Overview|Project Overview]]
- [[_COMMUNITY_Training Configuration|Training Configuration]]
- [[_COMMUNITY_Utility VAE|Utility VAE]]
- [[_COMMUNITY_Server Utilities|Server Utilities]]
- [[_COMMUNITY_Train Vision|Train Vision]]
- [[_COMMUNITY_Train Acoustic|Train Acoustic]]
- [[_COMMUNITY_Metrics & Evaluation|Metrics & Evaluation]]
- [[_COMMUNITY_Dataset References|Dataset References]]
- [[_COMMUNITY_Dependencies|Dependencies]]
- [[_COMMUNITY_Architecture Docs|Architecture Docs]]
- [[_COMMUNITY_Release Info|Release Info]]
- [[_COMMUNITY_Code Structure|Code Structure]]
- [[_COMMUNITY_Federated Components|Federated Components]]
- [[_COMMUNITY_Data Utilities|Data Utilities]]
- [[_COMMUNITY_Model Components|Model Components]]
- [[_COMMUNITY_Infrastructure Config|Infrastructure Config]]
- [[_COMMUNITY_Community 30|Community 30]]

## God Nodes (most connected - your core abstractions)
1. `AcousticVAE` - 14 edges
2. `train()` - 13 edges
3. `SemanticInterface` - 12 edges
4. `UtilityVAE` - 12 edges
5. `Generator` - 12 edges
6. `train()` - 12 edges
7. `loss()` - 11 edges
8. `train()` - 11 edges
9. `UrbanSound8KDataset` - 11 edges
10. `WaterQualityDataset` - 11 edges

## Surprising Connections (you probably didn't know these)
- `check_utility_checkpoint()` --calls--> `UtilityVAE`  [INFERRED]
  tests\check_all.py → models\utility_vae.py
- `Vision DP-GAN Training (RDP Accountant, Opacus)` --shares_data_with--> `Differential Privacy (DP-SGD via Opacus)`  [INFERRED]
  src/training/train_vision.py → PROJECT_REPORT.txt
- `load_semantic_interface()` --calls--> `SemanticInterface`  [INFERRED]
  dashboard\app.py → models\transformer_core.py
- `load_vision_model()` --calls--> `Generator`  [INFERRED]
  dashboard\app.py → models\vision_gan.py
- `load_acoustic_model()` --calls--> `AcousticVAE`  [INFERRED]
  dashboard\app.py → models\acoustic_vae.py

## Hyperedges (group relationships)
- **Multi-Modal Training Pipeline: Vision + Acoustic + Traffic** — train_vision, train_acoustic, train_utility [EXTRACTED 1.00]
- **Generative Models Ecosystem** — vision_cgan, acoustic_vae, utility_vae [EXTRACTED 1.00]
- **Privacy-Preserving Framework: DP + FL + Opacus** — differential_privacy, federated_learning, opacus_library [EXTRACTED 1.00]
- **Federated Learning Infrastructure** — fl_server, fl_client_vision, fl_client_acoustic [EXTRACTED 1.00]
- **Multi-Modal Dataset Ecosystem** — cityscapes_dataset, urbansound8k_dataset, metr_la_dataset, usgs_water_dataset [EXTRACTED 1.00]
- **Dashboard Analytics & SDG 11 Integration** — streamlit_dashboard, counterfactual_policy, sdg11_indicators, mia_audit [EXTRACTED 1.00]

## Communities

### Community 0 - "Utility Training Pipeline"
Cohesion: 0.07
Nodes (20): CityscapesDataset, METRLADataset, Urban-GenX | Data Loaders (FINAL — all 4 modalities) Handles: Cityscapes (PIL→Te, Loads METR-LA traffic speed data from HDF5.     Creates sliding window sequences, Loads USGS water quality CSV and creates sliding-window sequences     for VAE tr, Loads Cityscapes RGB + label pairs.     PIL Fix: explicit convert('RGB') prevent, Convert normalized values back to original scale., Loads UrbanSound8K audio files and converts to MFCC spectrograms.     Pads/trunc (+12 more)

### Community 1 - "Dashboard & UI Layer"
Cohesion: 0.07
Nodes (33): enhance_vision_output(), generate_mfcc_with_temperature(), load_water_model(), Urban-GenX | Streamlit Dashboard — FINAL PRODUCTION v3.0 =======================, Strong post-processing pipeline for DP-GAN 64×64 outputs.          DP-GAN with l, Generate MFCC with temperature scaling to fix flat output.          Temperature, Generate MFCC with temperature scaling to fix flat output.          Temperature, FIX: Auto-detect n_params from checkpoint shape to prevent size mismatch. (+25 more)

### Community 2 - "Data Loading & Preprocessing"
Cohesion: 0.11
Nodes (26): notify(), notify_crash_save(), notify_epoch(), notify_error(), notify_training_complete(), Urban-GenX | Remote Monitoring Sends push notifications to ntfy.sh topic: vibho, Send push notification to mobile via ntfy.sh.     Priority: min / low / default, get_beta() (+18 more)

### Community 3 - "Core Generative Models"
Cohesion: 0.12
Nodes (20): load_traffic_model(), Urban-GenX | tests/test_models.py (FINAL — all 4 modalities + water) Run: python, test_acoustic_vae_forward(), test_acoustic_vae_generate(), test_utility_vae_traffic(), test_utility_vae_water(), get_beta(), Urban-GenX | Utility VAE Training WITH DP-SGD (Phase 2) ======================= (+12 more)

### Community 4 - "Semantic Interface"
Cohesion: 0.1
Nodes (26): Acoustic VAE Training Log (50 epochs), Acoustic VAE: Beta-VAE for MFCC Spectrograms, Cityscapes: Semantic Urban Scene Dataset, Data Loader: Cityscapes / UrbanSound8K / METR-LA, Differential Privacy (DP-SGD via Opacus), Federated Learning (Flower FedAvg), FL Client Acoustic (all VAE params federated), FL Client Vision (D federated, G local) (+18 more)

### Community 5 - "Vision Generator"
Cohesion: 0.13
Nodes (14): AcousticVAE, generate(), interpolate(), Urban-GenX | Acoustic Node VAE over MFCC spectrograms from UrbanSound8K.  Inp, Reparameterization trick: z = mu + eps * sigma  (differentiable sampling)., x: [B, 1, mfcc_bins, time_frames] → (mu, log_var) each [B, latent_dim], z: [B, latent_dim] → [B, 1, mfcc_bins, time_frames], Full forward pass. Returns (reconstruction, mu, log_var). (+6 more)

### Community 6 - "Acoustic FL Client"
Cohesion: 0.13
Nodes (21): math, models_vision_gan, opacus, opacus_validators, src_utils_data_loader, src_utils_notifier, torch_nn, torch_optim (+13 more)

### Community 7 - "Vision FL Client"
Cohesion: 0.13
Nodes (13): load_semantic_interface(), check_semantic_interface(), test_semantic_interface(), test_semantic_list_scenes(), list_scenes(), Urban-GenX | Semantic Interface (Cross-Modal Transformer) Maps natural language, Maps natural language queries to urban scene presets.     Primary: sentence-tra, Pre-compute embeddings for all anchor phrases of all presets. (+5 more)

### Community 8 - "Acoustic VAE"
Cohesion: 0.16
Nodes (9): AcousticClient, Urban-GenX | Federated Acoustic Client  (FINAL FIXED VERSION) ==================, Return number of examples for Flower.         FIX: use len(dataset), NOT len(dat, Return all VAE parameter tensors as numpy arrays., Set VAE parameters from server aggregation result., Local training round.          Returns:             (updated_parameters, num_exa, Local evaluation round.          Returns:             (loss, num_examples, metri, Flower 1.5 NumPyClient for AcousticVAE.     Federated: all VAE parameters (encod (+1 more)

### Community 9 - "Privacy Auditing"
Cohesion: 0.15
Nodes (10): check_federated_client(), Urban-GenX | Federated Vision Client  (FINAL FIXED VERSION) ====================, Return number of EXAMPLES for Flower FedAvg weighting.         FIX: use len(data, Return Discriminator parameter tensors as numpy arrays., Apply server-aggregated Discriminator parameters., Local training: run N GAN steps, return updated D params.          Returns:, Evaluate Discriminator on local data.          Returns:             (d_loss, num, Flower 1.5 NumPyClient for Vision cGAN.     Federated : Discriminator (D) weight (+2 more)

### Community 10 - "Federated Server"
Cohesion: 0.17
Nodes (9): load_vision_model(), check_vision_gan(), test_vision_discriminator_shape(), test_vision_generator_shape(), Discriminator, Generator, Urban-GenX | Vision Node — cGAN Architecture (FIXED) ===========================, UNet-style conditional generator with skip connections.     Input:  condition [B (+1 more)

### Community 11 - "Testing & Validation"
Cohesion: 0.2
Nodes (4): check_utility_checkpoint(), tests/check_all.py ================== Urban-GenX end-to-end smoke-test suite (9, Decorator: wraps a check function, catches exceptions, records result., run_check()

### Community 12 - "Dataset Management"
Cohesion: 0.36
Nodes (8): main(), make_strategy(), Urban-GenX | Federated Learning Server (IMPROVED) ==============================, Flower metric aggregation callback.     Receives [(num_examples, {metric_name: m, run_acoustic_federation(), run_vision_federation(), _to_float_if_numeric(), weighted_metric_avg()

### Community 13 - "Water Data Pipeline"
Cohesion: 0.29
Nodes (1): Urban-GenX | Automated Dataset Downloader Run once: python src/utils/download_d

### Community 14 - "Project Overview"
Cohesion: 0.38
Nodes (6): download_from_usgs(), generate_synthetic_water_data(), main(), Urban-GenX | USGS Water Data Downloader (FIXED — multi-parameter) ==============, Try to download real USGS data via dataretrieval., Fallback: generate realistic synthetic water quality data.     Based on typical

### Community 15 - "Training Configuration"
Cohesion: 0.33
Nodes (6): Multi-Modal Synthesis: Vision, Acoustic, Traffic, Semantic, Privacy-Utility Gap: Barrier to Data-Driven Research, Sentence-BERT Semantic Interface, SDG 11: Sustainable Cities and Communities, Urban-GenX: Privacy-Preserving Synthetic City Digital Twin, Release v3.0.0: Final Production Release

### Community 16 - "Utility VAE"
Cohesion: 0.67
Nodes (1): fl_debug_plan.py ================ Urban-GenX — Federated Learning Debug & Fix Pl

### Community 17 - "Server Utilities"
Cohesion: 0.67
Nodes (3): Counterfactual Policy Simulation, SDG 11 Indicators: Transport, Noise/Air, Green Space, Streamlit Interactive Dashboard

### Community 18 - "Train Vision"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Train Acoustic"
Cohesion: 1.0
Nodes (1): Beta-VAE ELBO loss.          FIX: reduction='mean' (not 'sum').           - '

### Community 20 - "Metrics & Evaluation"
Cohesion: 1.0
Nodes (1): Sample from N(0,I) latent space → synthesize MFCC spectrogram.         Returns:

### Community 21 - "Dataset References"
Cohesion: 1.0
Nodes (1): Linearly interpolate in latent space between two MFCC inputs.         Useful fo

### Community 22 - "Dependencies"
Cohesion: 1.0
Nodes (1): Sample from latent space → synthesize utility vector

### Community 23 - "Architecture Docs"
Cohesion: 1.0
Nodes (1): Latent space interpolation between two utility states (counterfactual)

### Community 24 - "Release Info"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Code Structure"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Federated Components"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Data Utilities"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Model Components"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Infrastructure Config"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): USGS Water Quality Dataset

## Knowledge Gaps
- **104 isolated node(s):** `fl_debug_plan.py ================ Urban-GenX — Federated Learning Debug & Fix Pl`, `Urban-GenX | Streamlit Dashboard — FINAL PRODUCTION v3.0 =======================`, `Strong post-processing pipeline for DP-GAN 64×64 outputs.          DP-GAN with l`, `Generate MFCC with temperature scaling to fix flat output.          Temperature`, `FIX: Auto-detect n_params from checkpoint shape to prevent size mismatch.` (+99 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Train Vision`** (2 nodes): `main()`, `run_graphify.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Train Acoustic`** (1 nodes): `Beta-VAE ELBO loss.          FIX: reduction='mean' (not 'sum').           - '`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Metrics & Evaluation`** (1 nodes): `Sample from N(0,I) latent space → synthesize MFCC spectrogram.         Returns:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dataset References`** (1 nodes): `Linearly interpolate in latent space between two MFCC inputs.         Useful fo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dependencies`** (1 nodes): `Sample from latent space → synthesize utility vector`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Architecture Docs`** (1 nodes): `Latent space interpolation between two utility states (counterfactual)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Release Info`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Code Structure`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Federated Components`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Utilities`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Model Components`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Infrastructure Config`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `USGS Water Quality Dataset`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train()` connect `Acoustic FL Client` to `Utility Training Pipeline`, `Data Loading & Preprocessing`, `Acoustic VAE`, `Privacy Auditing`, `Federated Server`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `AcousticVAE` connect `Vision Generator` to `Acoustic VAE`, `Dashboard & UI Layer`, `Data Loading & Preprocessing`, `Core Generative Models`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `train()` connect `Data Loading & Preprocessing` to `Utility Training Pipeline`, `Core Generative Models`, `Vision Generator`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `AcousticVAE` (e.g. with `load_acoustic_model()` and `.__init__()`) actually correct?**
  _`AcousticVAE` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `train()` (e.g. with `Generator` and `Discriminator`) actually correct?**
  _`train()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `SemanticInterface` (e.g. with `load_semantic_interface()` and `check_semantic_interface()`) actually correct?**
  _`SemanticInterface` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `UtilityVAE` (e.g. with `load_water_model()` and `train()`) actually correct?**
  _`UtilityVAE` has 3 INFERRED edges - model-reasoned connections that need verification._