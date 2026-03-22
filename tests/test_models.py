"""
Urban-GenX | tests/test_models.py
Model forward-pass unit tests.
Run: python tests/test_models.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F


def test_acoustic_vae_forward():
    from models.acoustic_vae import AcousticVAE
    m = AcousticVAE(mfcc_bins=40, time_frames=128, latent_dim=64)
    m.eval()
    x = torch.randn(2, 1, 40, 128)
    with torch.no_grad():
        recon, mu, lv = m(x)
        loss = AcousticVAE.loss(recon, x, mu, lv, beta=1.0)
    assert recon.shape == x.shape, f"recon shape mismatch: {recon.shape}"
    assert mu.shape == (2, 64)
    assert float(loss.item()) < 20.0, f"Loss too large (check reduction='mean'): {float(loss.item())}"
    print("[PASS] test_acoustic_vae_forward")


def test_acoustic_vae_generate():
    from models.acoustic_vae import AcousticVAE
    m = AcousticVAE()
    with torch.no_grad():
        synth = m.generate(n_samples=3)
    assert synth.shape == (3, 1, 40, 128)
    print("[PASS] test_acoustic_vae_generate")


def test_vision_generator_shape():
    from models.vision_gan import Generator
    G = Generator(noise_dim=100, num_classes=35)
    G.eval()
    cond = torch.zeros(2, 35, 64, 64)
    with torch.no_grad():
        out = G(cond)
    assert out.shape == (2, 3, 64, 64), f"Generator output shape: {out.shape}"
    assert -1.1 <= float(out.min()) and float(out.max()) <= 1.1
    print("[PASS] test_vision_generator_shape")


def test_vision_discriminator_shape():
    from models.vision_gan import Generator, Discriminator
    G, D = Generator(), Discriminator()
    G.eval(); D.eval()
    cond = torch.zeros(2, 35, 64, 64)
    img  = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        pred = D(cond, img)
    assert pred.dim() == 4 and pred.shape[1] == 1
    print(f"[PASS] test_vision_discriminator_shape | patch_map={tuple(pred.shape)}")


def test_utility_vae_traffic():
    from models.utility_vae import build_traffic_vae, UtilityVAE
    m = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=64)
    x = torch.randn(4, 12 * 207)
    m.eval()
    with torch.no_grad():
        recon, mu, lv = m(x)
        loss          = UtilityVAE.loss(recon, x, mu, lv, beta=1.0)
        synth         = m.generate(n_samples=2)
    assert recon.shape == x.shape
    assert synth.shape == (2, 12 * 207)
    print(f"[PASS] test_utility_vae_traffic | loss={float(loss.item()):.4f}")


def test_semantic_interface():
    from models.transformer_core import SemanticInterface
    si = SemanticInterface()
    for query in ["construction site", "park", "highway", "busy intersection"]:
        preset = si.query(query)
        cond   = si.build_condition_tensor(preset, img_size=64, num_classes=35, batch_size=1)
        assert "scene_name" in preset
        assert cond.shape == (1, 35, 64, 64), f"cond shape: {cond.shape}"
    print("[PASS] test_semantic_interface")


def test_semantic_list_scenes():
    from models.transformer_core import SemanticInterface
    si     = SemanticInterface()
    scenes = si.list_scenes()
    assert len(scenes) >= 6, f"Expected at least 6 presets, got {len(scenes)}"
    print(f"[PASS] test_semantic_list_scenes | {len(scenes)} presets: {scenes}")


if __name__ == "__main__":
    test_acoustic_vae_forward()
    test_acoustic_vae_generate()
    test_vision_generator_shape()
    test_vision_discriminator_shape()
    test_utility_vae_traffic()
    test_semantic_interface()
    test_semantic_list_scenes()
    print("\n[ALL TESTS PASSED]")
