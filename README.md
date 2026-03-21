# Urban-GenX
**Privacy-Preserving Synthetic City Digital Twin for SDG 11 Research**

Urban-GenX is a research-grade, multi-modal synthetic city framework designed to bridge the privacy–utility gap in urban planning research. It creates a privacy-preserving digital twin using generative models, federated learning concepts, and differential privacy so that researchers can simulate sustainable city scenarios without exposing real citizen data.

## Project Goals
Urban-GenX is built to:
- generate synthetic urban street scenes from semantic priors,
- synthesize urban acoustic fingerprints,
- model utility / traffic behavior from time-series data,
- support natural-language semantic querying,
- provide an interactive dashboard for counterfactual scenario exploration,
- demonstrate privacy-preserving training using Differential Privacy and Federated Learning principles.

## Current Modalities
### Implemented
- **Vision Node**: cGAN for Cityscapes-conditioned synthetic street images
- **Acoustic Node**: VAE for MFCC-based UrbanSound8K synthesis
- **Utility/Traffic Node**: VAE for METR-LA traffic sequence synthesis
- **Semantic Interface**: text query → scene preset → condition tensor
- **Dashboard**: Streamlit interface with multi-modal outputs and policy sliders

### Planned / Future Extension
- Water-quality synthesis from USGS
- Energy/population expansion
- Diffusion-model upgrade for high-resolution image generation
- Cloud/GPU training path for final-year scale-up

---

## Repository Structure
```text
Urban-GenX/
├── data/
│   └── raw/
│       ├── cityscapes/
│       ├── urbansound8k/
│       ├── metr-la/
│       └── usgs_water/
├── models/
│   ├── vision_gan.py
│   ├── acoustic_vae.py
│   ├── utility_vae.py
│   └── transformer_core.py
├── src/
│   ├── training/
│   │   ├── train_vision.py
│   │   ├── train_acoustic.py
│   │   └── train_utility.py
│   ├── federated/
│   │   ├── server.py
│   │   ├── client_vision.py
│   │   └── client_acoustic.py
│   └── utils/
│       ├── data_loader.py
│       ├── download_datasets.py
│       ├── download_water_data.py
│       ├── notifier.py
│       └── privacy_audit.py
├── dashboard/
│   └── app.py
├── checkpoints/
├── tests/
│   └── check_all.py
├── requirements.txt
└── environment.yml
