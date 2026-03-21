"""
Urban-GenX | Quick sanity check for all trained models
Run: python tests/check_all.py
"""

import os
import sys
from unittest import result
from sentence_transformers import models
import torch
sys.path.insert(0, os.path.abspath("."))

print("=" * 60)
print("  Urban-GenX | Model Sanity Check")
print("=" * 60)

# ── 1. Utility/Traffic VAE ────────────────────────────────────────
print("\n[1] Utility/Traffic VAE")
try:
    from torch.utils.data import DataLoader
    from src.utils.data_loader import METRLADataset
    from models.utility_vae import build_traffic_vae, UtilityVAE

    ds = METRLADataset("data/raw/metr-la/metr-la.h5")
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    x, _ = next(iter(dl))
    x_flat = x.view(x.size(0), -1)

    m = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=64)
    ck = torch.load("checkpoints/utility_traffic_checkpoint.pth", map_location="cpu")
    m.load_state_dict(ck["model"])
    m.eval()

    with torch.no_grad():
        recon, mu, lv = m(x_flat)
        loss = UtilityVAE.loss(recon, x_flat, mu, lv, beta=1.0)

    print(f"  ✅ Epoch: {ck['epoch']} | Train loss: {ck['avg_loss']:.2f} | Val loss: {ck['avg_val']:.2f}")
    print(f"  ✅ Batch reconstruction loss: {float(loss):.2f}")
    print(f"  ✅ Decoded shape: {recon.shape} | range: [{float(recon.min()):.3f}, {float(recon.max()):.3f}]")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# ── 2. Acoustic VAE ──────────────────────────────────────────────
print("\n[2] Acoustic VAE")
try:
    from models.acoustic_vae import AcousticVAE

    model = AcousticVAE(latent_dim=64, time_frames=128)
    ck = torch.load("checkpoints/acoustic_checkpoint.pth", map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        synth = model.generate(n_samples=4)

    print(f"  ✅ Epoch: {ck['epoch']} | Val loss: {ck.get('avg_val', 'N/A')}")
    print(f"  ✅ Generated MFCC shape: {synth.shape}")
    print(f"  ✅ Value range: [{float(synth.min()):.3f}, {float(synth.max()):.3f}]")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# ── 3. Vision GAN ────────────────────────────────────────────────
print("\n[3] Vision GAN (Generator)")
try:
    from models.vision_gan import Generator

    G = Generator(noise_dim=100, num_classes=35)
    from opacus.validators import ModuleValidator
    G = ModuleValidator.fix(G)
    ck = torch.load("checkpoints/vision_checkpoint.pth", map_location="cpu")
    G.load_state_dict(ck["generator"])
    G.eval()

    cond = torch.zeros(2, 35, 64, 64)
    cond[:, 7, :, :] = 1.0  # road class
    with torch.no_grad():
        fake = G(cond)

    print(f"  ✅ Epoch: {ck['epoch']} | D: {ck['d_loss']:.4f} | G: {ck['g_loss']:.4f}")
    print(f"  ✅ Generated image shape: {fake.shape}")
    print(f"  ✅ Value range: [{float(fake.min()):.3f}, {float(fake.max()):.3f}]")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# ── 4. Semantic Interface ─────────────────────────────────────────
print("\n[4] Semantic Interface")
try:
    from models.transformer_core import SemanticInterface
    si = SemanticInterface()
    preset = si.query("construction site near busy road")
    cond = si.build_condition_tensor(preset, img_size=64, num_classes=35, batch_size=1)
    print(preset["scene_name"], cond.shape)
    print(f"  ✅ Query resolved to preset: {result.get('preset', 'N/A')}")
    print(f"  ✅ Condition tensor shape: {result.get('condition', torch.zeros(1)).shape}")
except Exception as e:
    print(f"  ❌ Failed: {e}")

print("\n" + "=" * 60)
print("  Sanity check complete.")
print("=" * 60)