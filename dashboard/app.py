"""
Urban-GenX | Streamlit Dashboard
Phase 1: Visualize synthetic street views + acoustic fingerprints.
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.vision_gan   import Generator
from models.acoustic_vae import AcousticVAE

st.set_page_config(page_title="Urban-GenX | Digital Twin", layout="wide",
                   page_icon="🏙️")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏙️ Urban-GenX | Synthetic City Digital Twin")
st.caption("Privacy-Preserving Urban Simulation for SDG 11 Research")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Simulation Controls")
modality   = st.sidebar.selectbox("Modality", ["Vision (Street View)", "Acoustic (Noise Map)"])
noise_seed = st.sidebar.slider("Noise Seed", 0, 999, 42)
n_samples  = st.sidebar.slider("Number of Samples", 1, 8, 4)

# ── Load Models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_vision_model():
    G = Generator(noise_dim=100, num_classes=35)
    ckpt_path = "checkpoints/vision_checkpoint.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        G.load_state_dict(ckpt['generator'])
        st.sidebar.success(f"✅ Loaded from Epoch {ckpt.get('epoch','?')}")
    else:
        st.sidebar.warning("⚠️ No checkpoint found — using random weights")
    G.eval()
    return G

@st.cache_resource
def load_acoustic_model():
    V = AcousticVAE(latent_dim=64)
    V.eval()
    return V

# ── Vision Tab ────────────────────────────────────────────────────────────────
if modality == "Vision (Street View)":
    st.subheader("🖼️ Synthetic Street View Generator")
    
    if st.button("🎲 Generate Synthetic Scene"):
        G = load_vision_model()
        torch.manual_seed(noise_seed)
        
        # Random semantic condition (would come from real map in full system)
        fake_cond = torch.zeros(n_samples, 35, 64, 64)
        fake_cond[:, torch.randint(0,35,(1,)).item(), :, :] = 1.0
        
        with torch.no_grad():
            synth = G(fake_cond)          # [B, 3, 64, 64] in [-1,1]
            synth = (synth + 1) / 2       # → [0,1]

        cols = st.columns(n_samples)
        for i, col in enumerate(cols):
            img_np = synth[i].permute(1,2,0).numpy()
            col.image(img_np, caption=f"Synthetic Scene {i+1}", use_column_width=True)

        st.info("💡 These images contain **zero real citizen data** — fully synthetic via cGAN.")

# ── Acoustic Tab ──────────────────────────────────────────────────────────────
elif modality == "Acoustic (Noise Map)":
    st.subheader("🔊 Synthetic Urban Soundscape Generator")
    
    if st.button("🎲 Generate Acoustic Fingerprint"):
        V = load_acoustic_model()
        torch.manual_seed(noise_seed)
        
        with torch.no_grad():
            mfcc = V.generate(n_samples=n_samples)   # [B, 1, 40, 128]

        fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 3))
        if n_samples == 1: axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(mfcc[i,0].numpy(), aspect='auto', origin='lower', cmap='magma')
            ax.set_title(f"Soundscape {i+1}")
            ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        st.info("💡 Synthetic MFCC spectrograms — generated from the Acoustic VAE latent space.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Urban-GenX | Research Framework | SDG 11 | DP-SGD Privacy | Federated Learning")
