"""
Urban-GenX | Sanity Check Suite
Run: python tests/check_all.py
Verifies: data loaders, model architectures, checkpoints, semantic interface, DP attachment.
All checks are self-contained — no training or GPU needed.
Exit code 0 = all passed. Non-zero = check failed (error printed).
"""

import os
import sys
import traceback

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
SKIP = "  ⏭️  SKIP"
results = []


def check(name, fn):
    """Run a check function and record pass/fail."""
    print(f"\n[CHECK] {name}")
    try:
        msg = fn()
        print(f"{PASS}" + (f" — {msg}" if msg else ""))
        results.append((name, True, ""))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"{FAIL} — {e}")
        print(tb)
        results.append((name, False, str(e)))


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: UrbanSound8K data loader
# ─────────────────────────────────────────────────────────────────────────────
def check_urbansound():
    from src.utils.data_loader import UrbanSound8KDataset
    root = "data/raw/urbansound8k"
    if not os.path.exists(os.path.join(root, "metadata", "UrbanSound8K.csv")):
        raise FileNotFoundError(f"Missing {root}/metadata/UrbanSound8K.csv — run folder-fix commands first")
    d = UrbanSound8KDataset(root, folds=[1])   # folds=[1] is fast (subset)
    assert len(d) > 0, "Dataset is empty"
    x, y = d[0]
    assert x.shape == torch.Size([1, 40, 128]), f"Expected [1,40,128], got {x.shape}"
    assert isinstance(int(y), int), f"Label not int: {y}"
    return f"{len(d)} clips in fold 1, MFCC shape {tuple(x.shape)}, label {y}"

check("UrbanSound8K loader", check_urbansound)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: METR-LA data loader
# ─────────────────────────────────────────────────────────────────────────────
def check_metrla():
    from src.utils.data_loader import METRLADataset
    h5 = "data/raw/metr-la/metr-la.h5"
    if not os.path.exists(h5):
        raise FileNotFoundError(f"Missing {h5} — download from Kaggle (annnnguyen/metr-la-dataset)")
    d = METRLADataset(h5, seq_len=12, pred_len=12)
    assert len(d) > 0, "Dataset empty"
    x, y = d[0]
    assert x.shape == torch.Size([12, 207]), f"Expected [12,207], got {x.shape}"
    return f"{len(d)} windows, x shape {tuple(x.shape)}, y shape {tuple(y.shape)}"

check("METR-LA loader", check_metrla)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Cityscapes data loader (skipped if data absent)
# ─────────────────────────────────────────────────────────────────────────────
def check_cityscapes():
    from src.utils.data_loader import CityscapesDataset
    root = "data/raw/cityscapes"
    left = os.path.join(root, "leftImg8bit", "train")
    gt   = os.path.join(root, "gtFine", "train")
    if not os.path.exists(left) or not os.path.exists(gt):
        raise FileNotFoundError(
            f"Cityscapes not found. Expected:\n"
            f"  {left}\n  {gt}\n"
            f"Download from https://www.cityscapes-dataset.com/register/"
        )
    d = CityscapesDataset(root, split="train", img_size=64)
    assert len(d) > 0, "Dataset empty"
    img, cond = d[0]
    assert img.shape  == torch.Size([3, 64, 64]),  f"Image shape: {img.shape}"
    assert cond.shape == torch.Size([35, 64, 64]), f"Cond shape: {cond.shape}"
    assert float(img.min()) >= -1.05 and float(img.max()) <= 1.05, "Image not in [-1,1]"
    return f"{len(d)} samples, img {tuple(img.shape)}, cond {tuple(cond.shape)}"

check("Cityscapes loader", check_cityscapes)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: AcousticVAE forward pass + loss
# ─────────────────────────────────────────────────────────────────────────────
def check_acoustic_vae():
    from models.acoustic_vae import AcousticVAE
    model = AcousticVAE(mfcc_bins=40, time_frames=128, latent_dim=64)
    model.eval()
    x = torch.randn(2, 1, 40, 128)
    with torch.no_grad():
        recon, mu, lv = model(x)
    assert recon.shape == x.shape, f"Recon shape {recon.shape} != {x.shape}"
    assert mu.shape    == torch.Size([2, 64])
    assert lv.shape    == torch.Size([2, 64])

    loss = AcousticVAE.loss(recon, x, mu, lv, beta=1.0)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert float(loss) < 10.0, f"Loss too large ({float(loss):.4f}) — check mean-reduction fix"

    # Generation
    gen = model.generate(n_samples=3)
    assert gen.shape == torch.Size([3, 1, 40, 128])
    return f"loss={float(loss):.4f}, recon {tuple(recon.shape)}, gen {tuple(gen.shape)}"

check("AcousticVAE forward + loss + generate", check_acoustic_vae)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: Vision GAN forward pass
# ─────────────────────────────────────────────────────────────────────────────
def check_vision_gan():
    from models.vision_gan import Generator, Discriminator
    G = Generator(noise_dim=100, num_classes=35)
    D = Discriminator(num_classes=35)
    G.eval(); D.eval()

    cond = torch.randn(2, 35, 64, 64)
    with torch.no_grad():
        fake = G(cond)
        score = D(cond, fake)

    assert fake.shape  == torch.Size([2, 3, 64, 64]), f"G output: {fake.shape}"
    assert float(fake.min()) >= -1.05 and float(fake.max()) <= 1.05, "G output not in [-1,1]"
    assert torch.isfinite(fake).all(), "G output has NaN/Inf"
    assert torch.isfinite(score).all(), f"D output has NaN/Inf"
    return f"G: {tuple(fake.shape)}, D: {tuple(score.shape)}, G_range=[{float(fake.min()):.3f}, {float(fake.max()):.3f}]"

check("Vision GAN (G + D) forward pass", check_vision_gan)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6: UtilityVAE forward pass
# ─────────────────────────────────────────────────────────────────────────────
def check_utility_vae():
    from models.utility_vae import build_traffic_vae, UtilityVAE
    model = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=64)
    model.eval()
    x = torch.randn(2, 12 * 207)
    with torch.no_grad():
        recon, mu, lv = model(x)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()

    # Decode
    z = torch.randn(1, model.latent_dim)
    out = model.decode(z)
    assert out.shape == torch.Size([1, 12 * 207])
    return f"recon {tuple(recon.shape)}, decode {tuple(out.shape)}, finite={torch.isfinite(recon).all().item()}"

check("UtilityVAE (traffic) forward + decode", check_utility_vae)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7: Checkpoint loading — acoustic + utility
# ─────────────────────────────────────────────────────────────────────────────
def check_checkpoints():
    msgs = []

    # Utility checkpoint (should exist after train_utility.py completed)
    util_path = "checkpoints/utility_traffic_checkpoint.pth"
    if os.path.exists(util_path):
        from models.utility_vae import build_traffic_vae
        ck = torch.load(util_path, map_location="cpu")
        assert "model" in ck, "No 'model' key in utility checkpoint"
        m = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=64)
        m.load_state_dict(ck["model"])
        # Verify no NaN in weights
        for name, p in m.named_parameters():
            assert torch.isfinite(p).all(), f"NaN/Inf in utility param: {name}"
        msgs.append(f"Utility ✅ epoch={ck.get('epoch','?')} val={ck.get('avg_val', '?')}")
    else:
        msgs.append(f"Utility ⏭️  not found at {util_path}")

    # Acoustic checkpoint (may not exist if training incomplete)
    acou_path = "checkpoints/acoustic_checkpoint.pth"
    if os.path.exists(acou_path):
        from models.acoustic_vae import AcousticVAE
        ck = torch.load(acou_path, map_location="cpu")
        assert "model" in ck, "No 'model' key in acoustic checkpoint"
        cfg = ck.get("model_config", {"mfcc_bins": 40, "time_frames": 128, "latent_dim": 64})
        m = AcousticVAE(**cfg)
        m.load_state_dict(ck["model"])
        for name, p in m.named_parameters():
            assert torch.isfinite(p).all(), f"NaN/Inf in acoustic param: {name}"
        msgs.append(f"Acoustic ✅ epoch={ck.get('epoch','?')} val={ck.get('avg_val_loss', '?'):.4f}"
                    if isinstance(ck.get('avg_val_loss'), float)
                    else f"Acoustic ✅ epoch={ck.get('epoch','?')}")
    else:
        msgs.append(f"Acoustic ⏭️  not found (run train_acoustic.py)")

    # Vision checkpoint
    vis_path = "checkpoints/vision_checkpoint.pth"
    if os.path.exists(vis_path):
        from models.vision_gan import Generator, Discriminator
        ck = torch.load(vis_path, map_location="cpu")
        assert "generator" in ck, "No 'generator' key in vision checkpoint"
        G = Generator(noise_dim=100, num_classes=35)
        D = Discriminator(num_classes=35)
        G.load_state_dict(ck["generator"])
        D.load_state_dict(ck["discriminator"])
        for name, p in G.named_parameters():
            assert torch.isfinite(p).all(), f"NaN/Inf in G param: {name}"
        for name, p in D.named_parameters():
            assert torch.isfinite(p).all(), f"NaN/Inf in D param: {name}"
        msgs.append(f"Vision ✅ epoch={ck.get('epoch','?')}")
    else:
        msgs.append(f"Vision ⏭️  not found (run train_vision.py)")

    return " | ".join(msgs)

check("Checkpoint loading (all models)", check_checkpoints)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8: Semantic Interface
# ─────────────────────────────────────────────────────────────────────────────
def check_semantic_interface():
    from models.transformer_core import SemanticInterface
    si = SemanticInterface()

    # Test query
    preset = si.query("busy intersection")
    required_keys = {"scene_name", "acoustic_class", "traffic_multiplier", "noise_level", "green_space"}
    missing = required_keys - set(preset.keys())
    assert not missing, f"Preset missing keys: {missing}"

    # Test build_condition_tensor
    cond = si.build_condition_tensor(preset, img_size=64, num_classes=35, batch_size=1)
    assert cond.shape == torch.Size([1, 35, 64, 64]), f"Cond tensor shape: {cond.shape}"
    assert torch.isfinite(cond).all()

    # List scenes
    scenes = si.list_scenes()
    assert len(scenes) >= 8, f"Expected ≥8 presets, got {len(scenes)}"

    return (f"scene='{preset['scene_name']}' | "
            f"acoustic={preset.get('acoustic_class_name', preset['acoustic_class'])} | "
            f"traffic_mult={preset['traffic_multiplier']} | "
            f"scenes available: {len(scenes)}")

check("Semantic Interface (query + condition tensor)", check_semantic_interface)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 9: DP attachment (dry-run with tiny dataset)
# ─────────────────────────────────────────────────────────────────────────────
def check_dp_attachment():
    """Verify Opacus PrivacyEngine can attach to Discriminator cleanly."""
    from models.vision_gan import Discriminator
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from torch.utils.data import TensorDataset, DataLoader

    D = Discriminator(num_classes=35)
    D = ModuleValidator.fix(D)
    errors = ModuleValidator.validate(D, strict=False)
    assert len(errors) == 0, f"ModuleValidator errors: {errors}"

    opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Minimal DataLoader for PrivacyEngine (need >batch_size samples)
    dummy = TensorDataset(torch.randn(20, 38, 64, 64))  # 38 = 35 cond + 3 rgb
    loader = DataLoader(dummy, batch_size=4)

    pe = PrivacyEngine(secure_mode=False, accountant="rdp")
    D, opt, loader = pe.make_private_with_epsilon(
        module=D,
        optimizer=opt,
        data_loader=loader,
        target_epsilon=10.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        epochs=50,
    )
    eps = pe.get_epsilon(1e-5)
    assert eps > 0, "Privacy engine returned non-positive epsilon"
    assert hasattr(D, "disable_hooks"), "D missing disable_hooks (Opacus GradSampleModule)"
    assert hasattr(D, "enable_hooks"),  "D missing enable_hooks"
    return f"DP attached OK | σ noise inferred | ε(δ=1e-5)={eps:.4f} | disable_hooks: ✅"

check("Opacus DP attachment to Discriminator", check_dp_attachment)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("URBAN-GENX SANITY CHECK SUMMARY")
print("=" * 60)
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)

for name, ok, err in results:
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status}  {name}")
    if not ok:
        print(f"         └─ {err[:120]}")

print(f"\nResult: {passed}/{total} passed" + (f" | {failed} FAILED" if failed else " | All clear 🎉"))
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
