# 🏙️ Urban-GenX
**Privacy-Preserving Synthetic City Digital Twin for SDG 11 Research**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

> A research-grade framework that bridges the **Privacy-Utility Gap** in urban planning by creating a fully synthetic Digital Twin. Uses Federated Learning (Flower/FedAvg) and Differential Privacy (Opacus DP-SGD) to simulate SDG 11 scenarios without touching real citizen data.

---

## 🗂️ Project Structure
Urban-GenX/ ├── data/raw/ # Datasets (NOT in Git — too large) ├── models/ │ ├── vision_gan.py # cGAN Generator + PatchGAN Discriminator │ ├── acoustic_vae.py # VAE for MFCC audio synthesis │ ├── utility_vae.py # VAE for traffic + water time-series │ └── transformer_core.py # Sentence-BERT semantic interface ├── src/ │ ├── training/ │ │ ├── train_vision.py # DP-SGD cGAN training │ │ ├── train_acoustic.py # Acoustic VAE training │ │ └── train_utility.py # Traffic VAE training │ ├── federated/ │ │ ├── server.py # Flower FL server (FedAvg) │ │ ├── client_vision.py # FL Vision node │ │ └── client_acoustic.py # FL Acoustic node │ └── utils/ │ ├── data_loader.py # All dataset classes │ ├── notifier.py # ntfy.sh mobile alerts │ ├── privacy_audit.py # MIA audit │ └── download_datasets.py ├── checkpoints/ # .pth files (NOT in Git) ├── dashboard/ │ └── app.py # Streamlit multi-modal UI ├── environment.yml └── requirements.txt

