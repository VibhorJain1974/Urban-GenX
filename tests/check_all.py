"""
Urban-GenX | Final System Validation Suite
==========================================
Run: python tests/check_all.py

Validates all implemented components of Urban-GenX.
Reports PASS / WARN / FAIL / SKIP per check.
Exit code 0 if all checks pass (or only have WARNs), 1 if any FAIL.

Checks:
  1.  Core imports (torch, flwr, opacus, librosa, streamlit)
  2.  sentence-transformers import
  3.  UrbanSound8K dataset loader shape
  4.  METR-LA dataset loader shape
  5.  Cityscapes dataset loader (SKIP if absent)
  6.  AcousticVAE forward + loss value sanity
  7.  Vision Generator forward + output range
  8.  Vision Discriminator forward
  9.  UtilityVAE forward + decode
  10. Semantic interface query + condition tensor shape
  11. Utility checkpoint loading + finite metrics
  12. Acoustic checkpoint loading (if exists)
  13. Vision checkpoint loading + finite loss check
  14. Opacus DP attachment sanity (GradSampleModule.disable_hooks present)
  15. Flower 1.5 client import + parameter shape sanity
"""

import os
import sys
import math
import traceback
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

# ── Output helpers ────────────────────────────────────────────────────────────
passed  = 0
failed  = 0
warned  = 0
skipped = 0


def banner(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def ok(msg: str):
    global passed
    passed += 1
    print(f"  [PASS] {msg}")


def warn(msg: str):
    global warned
    warned += 1
    print(f"  [WARN] {msg}")


def fail(msg: str):
    global failed
    failed += 1
    print(f"  [FAIL] {msg}")


def skip(msg: str):
    global skipped
    skipped += 1
    print(f"  [SKIP] {msg}")


def is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# ─── CHECK 1: Core imports ───────────────────────────────────────────────────
def check_core_imports():
    banner("1. Core Library Imports")
    for name, pkg in [
        ("PyTorch",    "torch"),
        ("Torchvision","torchvision"),
        ("Flower",     "flwr"),
        ("Opacus",     "opacus"),
        ("librosa",    "librosa"),
        ("streamlit",  "streamlit"),
        ("numpy",      "numpy"),
        ("pandas",     "pandas"),
        ("h5py",       "h5py"),
        ("sklearn",    "sklearn"),
        ("PIL",        "PIL"),
        ("tqdm",       "tqdm"),
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{name} v{ver}")
        except ImportError as e:
            fail(f"{name} not found: {e}")


# ─── CHECK 2: sentence-transformers ─────────────────────────────────────────
def check_sbert():
    banner("2. sentence-transformers (SBERT)")
    try:
        from sentence_transformers import SentenceTransformer
        ok("sentence-transformers import OK")
    except Exception as e:
        warn(
            f"sentence-transformers import failed: {e}\n"
            "  → keyword-fallback is active in transformer_core.py (OK for demo)\n"
            "  → fix: pip install sentence-transformers==2.7.0"
        )


# ─── CHECK 3: UrbanSound8K loader ────────────────────────────────────────────
def check_urbansound8k():
    banner("3. UrbanSound8K Data Loader")
    data_path = ROOT / "data" / "raw" / "urbansound8k"

    # Try both common layouts
    meta_path = data_path / "metadata" / "UrbanSound8K.csv"
    alt_path  = data_path / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"

    if not meta_path.exists() and not alt_path.exists():
        fail(
            f"UrbanSound8K metadata CSV not found.\n"
            f"  Expected: {meta_path}"
        )
        return

    try:
        from src.utils.data_loader import UrbanSound8KDataset
        ds = UrbanSound8KDataset(str(data_path))
        x, label = ds[0]
        assert tuple(x.shape) == (1, 40, 128), f"Wrong MFCC shape: {x.shape}"
        ok(f"UrbanSound8K loader: {len(ds)} samples | MFCC shape {tuple(x.shape)}")
    except Exception as e:
        fail(f"UrbanSound8K loader: {e}")
        traceback.print_exc()


# ─── CHECK 4: METR-LA loader ─────────────────────────────────────────────────
def check_metrla():
    banner("4. METR-LA Data Loader")
    h5_path = ROOT / "data" / "raw" / "metr-la" / "metr-la.h5"
    if not h5_path.exists():
        fail(
            f"metr-la.h5 not found at {h5_path}.\n"
            "  Download: kaggle datasets download -d annnnguyen/metr-la-dataset "
            "-p data/raw/metr-la --unzip"
        )
        return
    try:
        from src.utils.data_loader import METRLADataset
        ds = METRLADataset(str(h5_path))
        x, y = ds[0]
        assert x.dim() == 2, f"Expected 2D sequence, got {x.shape}"
        ok(f"METR-LA loader: {len(ds)} windows | x={tuple(x.shape)}, y={tuple(y.shape)}")
    except Exception as e:
        fail(f"METR-LA loader: {e}")


# ─── CHECK 5: Cityscapes loader (SKIP if absent) ─────────────────────────────
def check_cityscapes():
    banner("5. Cityscapes Data Loader")
    left_path = ROOT / "data" / "raw" / "cityscapes" / "leftImg8bit" / "train"
    gt_path   = ROOT / "data" / "raw" / "cityscapes" / "gtFine" / "train"

    if not left_path.exists() or not gt_path.exists():
        skip(
            "Cityscapes not found (large dataset, manual download required).\n"
            "  Register at https://www.cityscapes-dataset.com/register/\n"
            "  OR: kaggle datasets download -d electraawais/cityscape-dataset -p data/raw/cityscapes --unzip"
        )
        return
    try:
        from src.utils.data_loader import CityscapesDataset
        ds = CityscapesDataset(str(ROOT / "data" / "raw" / "cityscapes"))
        img, cond = ds[0]
        assert tuple(img.shape)  == (3, 64, 64), f"Wrong img shape: {img.shape}"
        assert tuple(cond.shape) == (35, 64, 64), f"Wrong cond shape: {cond.shape}"
        ok(f"Cityscapes loader: {len(ds)} samples | img={tuple(img.shape)} cond={tuple(cond.shape)}")
    except Exception as e:
        fail(f"Cityscapes loader: {e}")


# ─── CHECK 6: AcousticVAE forward + loss ─────────────────────────────────────
def check_acoustic_vae():
    banner("6. AcousticVAE Model Forward Pass + Loss")
    try:
        from models.acoustic_vae import AcousticVAE
        model = AcousticVAE(mfcc_bins=40, time_frames=128, latent_dim=64)
        model.eval()

        x = torch.randn(4, 1, 40, 128)
        with torch.no_grad():
            recon, mu, lv = model(x)
            loss = AcousticVAE.loss(recon, x, mu, lv, beta=1.0)

        assert tuple(recon.shape) == (4, 1, 40, 128), f"Wrong recon shape: {recon.shape}"
        assert is_finite(loss.item()), f"Loss is not finite: {loss.item()}"

        # With mean reduction, loss should be < 10.0 for random input
        if float(loss.item()) > 10.0:
            warn(
                f"AcousticVAE loss={float(loss.item()):.4f} > 10.0. "
                "Check reduction='mean' in AcousticVAE.loss(). "
                "If using 'sum', inflate by 5120× per sample."
            )
        else:
            ok(f"AcousticVAE forward OK | recon={tuple(recon.shape)} | loss={float(loss.item()):.4f}")

        # Generation test
        synth = model.generate(n_samples=2)
        assert tuple(synth.shape) == (2, 1, 40, 128)
        ok(f"AcousticVAE generate OK | shape={tuple(synth.shape)}")

    except Exception as e:
        fail(f"AcousticVAE check: {e}")
        traceback.print_exc()


# ─── CHECK 7: Vision Generator forward ───────────────────────────────────────
def check_vision_generator():
    banner("7. Vision Generator Forward Pass")
    try:
        from models.vision_gan import Generator
        G = Generator(noise_dim=100, num_classes=35)
        G.eval()

        cond = torch.zeros(2, 35, 64, 64)
        cond[:, 7, :, :] = 1.0   # road class

        with torch.no_grad():
            fake = G(cond)

        assert tuple(fake.shape) == (2, 3, 64, 64), f"Wrong shape: {fake.shape}"
        assert float(fake.min()) >= -1.1, f"Output below -1: {float(fake.min())}"
        assert float(fake.max()) <=  1.1, f"Output above  1: {float(fake.max())}"
        ok(
            f"Generator forward OK | shape={tuple(fake.shape)} | "
            f"range=[{float(fake.min()):.3f}, {float(fake.max()):.3f}]"
        )
    except Exception as e:
        fail(f"Vision Generator check: {e}")
        traceback.print_exc()


# ─── CHECK 8: Discriminator forward ─────────────────────────────────────────
def check_vision_discriminator():
    banner("8. Vision Discriminator Forward Pass")
    try:
        from models.vision_gan import Generator, Discriminator
        G = Generator(noise_dim=100, num_classes=35)
        D = Discriminator(num_classes=35)
        G.eval(); D.eval()

        cond = torch.zeros(2, 35, 64, 64)
        cond[:, 7, :, :] = 1.0

        with torch.no_grad():
            fake      = G(cond)
            real_pred = D(cond, torch.randn(2, 3, 64, 64))
            fake_pred = D(cond, fake)

        ok(
            f"Discriminator forward OK | "
            f"real_pred={tuple(real_pred.shape)} fake_pred={tuple(fake_pred.shape)}"
        )
    except Exception as e:
        fail(f"Vision Discriminator check: {e}")
        traceback.print_exc()


# ─── CHECK 9: UtilityVAE forward + decode ────────────────────────────────────
def check_utility_vae():
    banner("9. UtilityVAE (Traffic) Forward Pass")
    try:
        from models.utility_vae import build_traffic_vae, UtilityVAE
        model = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=64)
        model.eval()

        x = torch.randn(4, 12 * 207)
        with torch.no_grad():
            recon, mu, lv = model(x)
            loss          = UtilityVAE.loss(recon, x, mu, lv, beta=1.0)

        assert tuple(recon.shape) == (4, 12 * 207), f"Wrong shape: {recon.shape}"
        assert is_finite(loss.item()), f"Loss not finite: {loss.item()}"

        z     = torch.randn(1, 64)
        synth = model.decode(z)
        assert tuple(synth.shape) == (1, 12 * 207), f"Wrong decode shape: {synth.shape}"
        ok(f"UtilityVAE forward OK | recon={tuple(recon.shape)} | decode={tuple(synth.shape)}")

    except Exception as e:
        fail(f"UtilityVAE check: {e}")
        traceback.print_exc()


# ─── CHECK 10: Semantic interface ────────────────────────────────────────────
def check_semantic_interface():
    banner("10. Semantic Interface (Text → Scene Preset)")
    try:
        from models.transformer_core import SemanticInterface
        si     = SemanticInterface()
        preset = si.query("busy intersection with heavy traffic")
        cond   = si.build_condition_tensor(preset, img_size=64, num_classes=35, batch_size=2)

        assert "scene_name" in preset, "preset missing 'scene_name'"
        assert tuple(cond.shape) == (2, 35, 64, 64), f"Wrong cond shape: {cond.shape}"
        ok(
            f"SemanticInterface OK | "
            f"preset='{preset['scene_name']}' | cond_shape={tuple(cond.shape)}"
        )
    except Exception as e:
        fail(f"SemanticInterface check: {e}")
        traceback.print_exc()


# ─── CHECK 11: Utility checkpoint ────────────────────────────────────────────
def check_utility_checkpoint():
    banner("11. Utility/Traffic Checkpoint")
    ckpt_path = ROOT / "checkpoints" / "utility_traffic_checkpoint.pth"
    if not ckpt_path.exists():
        warn(f"Utility checkpoint not found: {ckpt_path}. Run train_utility.py first.")
        return
    try:
        from models.utility_vae import build_traffic_vae, UtilityVAE
        ck    = torch.load(ckpt_path, map_location="cpu")
        epoch = int(ck.get("epoch", 0))
        model = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=64)
        model.load_state_dict(ck["model"])
        model.eval()

        # Decode test
        z     = torch.randn(1, 64)
        synth = model.decode(z)
        assert is_finite(synth.min().item()), "Decoded output has NaN/Inf"

        avg_loss = ck.get("avg_loss", None)
        avg_val  = ck.get("avg_val",  None)
        ok(
            f"Utility checkpoint epoch={epoch} | "
            f"train={avg_loss:.4f} | val={avg_val:.4f} | decode_ok"
            if (avg_loss is not None and is_finite(avg_loss))
            else f"Utility checkpoint epoch={epoch} | decode_ok"
        )
    except Exception as e:
        fail(f"Utility checkpoint: {e}")
        traceback.print_exc()


# ─── CHECK 12: Acoustic checkpoint ───────────────────────────────────────────
def check_acoustic_checkpoint():
    banner("12. Acoustic Checkpoint")
    best_path  = ROOT / "checkpoints" / "acoustic_best.pth"
    epoch_path = ROOT / "checkpoints" / "acoustic_checkpoint.pth"
    chosen     = best_path if best_path.exists() else epoch_path

    if not chosen.exists():
        warn(f"No acoustic checkpoint found. Run train_acoustic.py first.")
        return

    try:
        from models.acoustic_vae import AcousticVAE
        ck  = torch.load(chosen, map_location="cpu")
        cfg = ck.get("model_config", {})
        model = AcousticVAE(
            mfcc_bins=cfg.get("mfcc_bins", 40),
            time_frames=cfg.get("time_frames", 128),
            latent_dim=cfg.get("latent_dim", 64),
        )
        model.load_state_dict(ck["model"])
        model.eval()

        synth = model.generate(n_samples=2)
        assert tuple(synth.shape) == (2, 1, 40, 128)
        assert is_finite(synth.min().item())

        avg_val = ck.get("avg_val_loss", None)
        ok(
            f"Acoustic checkpoint {chosen.name} | epoch={ck.get('epoch','?')} | "
            f"val={avg_val:.4f} | generate_ok"
            if (avg_val is not None and is_finite(avg_val))
            else f"Acoustic checkpoint {chosen.name} | generate_ok"
        )
    except Exception as e:
        fail(f"Acoustic checkpoint: {e}")
        traceback.print_exc()


# ─── CHECK 13: Vision checkpoint ─────────────────────────────────────────────
def check_vision_checkpoint():
    banner("13. Vision Checkpoint")
    ckpt_path = ROOT / "checkpoints" / "vision_checkpoint.pth"
    if not ckpt_path.exists():
        warn(f"No vision checkpoint. Run train_vision.py first.")
        return
    try:
        from models.vision_gan import Generator
        from opacus.validators import ModuleValidator
        ck    = torch.load(ckpt_path, map_location="cpu")
        G     = Generator(noise_dim=100, num_classes=35)
        G     = ModuleValidator.fix(G)   # match training architecture
        G.load_state_dict(ck["generator"])
        G.eval()

        cond = torch.zeros(2, 35, 64, 64); cond[:, 7, :, :] = 1.0
        with torch.no_grad():
            fake = G(cond)
        assert is_finite(fake.min().item()), "Generator output contains NaN/Inf"

        d_loss = ck.get("d_loss", None)
        g_loss = ck.get("g_loss", None)

        d_ok = d_loss is not None and is_finite(d_loss)
        g_ok = g_loss is not None and is_finite(g_loss)

        if d_ok and g_ok:
            ok(
                f"Vision checkpoint epoch={ck.get('epoch','?')} | "
                f"D={d_loss:.4f} G={g_loss:.4f} | output_finite"
            )
        else:
            warn(
                f"Vision checkpoint epoch={ck.get('epoch','?')} loaded "
                f"but d_loss={d_loss} g_loss={g_loss} (possible NaN). "
                "Delete checkpoint and retrain from scratch."
            )
    except Exception as e:
        fail(f"Vision checkpoint: {e}")
        traceback.print_exc()


# ─── CHECK 14: Opacus DP attachment ──────────────────────────────────────────
def check_dp_attachment():
    banner("14. Opacus DP Attachment Sanity")
    try:
        from models.vision_gan import Discriminator
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        from torch.utils.data import TensorDataset, DataLoader
        import torch.optim as optim

        D   = ModuleValidator.fix(Discriminator(num_classes=35))
        opt = optim.Adam(D.parameters(), lr=1e-3)

        # Tiny dummy loader
        dummy = DataLoader(
            TensorDataset(
                torch.randn(16, 38, 64, 64),  # condition+image concatenated
                torch.randn(16, 1),
            ),
            batch_size=4,
        )

        pe = PrivacyEngine(secure_mode=False, accountant="rdp")
        D_priv, _, _ = pe.make_private_with_epsilon(
            module=D, optimizer=opt, data_loader=dummy,
            target_epsilon=10.0, target_delta=1e-5,
            max_grad_norm=1.0, epochs=5,
        )

        has_disable = hasattr(D_priv, "disable_hooks")
        has_enable  = hasattr(D_priv, "enable_hooks")

        if has_disable and has_enable:
            ok("DP attached | disable_hooks() + enable_hooks() available")
        else:
            warn(
                "DP attached but disable_hooks/enable_hooks not found. "
                "Verify Opacus version is 1.4.0."
            )
    except Exception as e:
        fail(f"Opacus DP attachment: {e}")
        traceback.print_exc()


# ─── CHECK 15: Flower clients ────────────────────────────────────────────────
def check_fl_clients():
    banner("15. Federated Learning Client Imports")
    try:
        from src.federated.client_acoustic import AcousticClient
        a  = AcousticClient(client_id="0")
        ap = a.get_parameters({})
        ok(f"AcousticClient import OK | parameter tensors: {len(ap)}")
    except Exception as e:
        fail(f"AcousticClient: {e}")

    try:
        from src.federated.client_vision import VisionClient
        v  = VisionClient(client_id="0")
        vp = v.get_parameters({})
        ok(f"VisionClient import OK | parameter tensors: {len(vp)}")
    except Exception as e:
        fail(f"VisionClient: {e}")

    try:
        from src.federated.server import make_strategy
        s = make_strategy()
        ok(f"Server strategy import OK: {s.__class__.__name__}")
    except Exception as e:
        fail(f"Server strategy import: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    banner("Urban-GenX | Final System Validation Suite")

    check_core_imports()
    check_sbert()
    check_urbansound8k()
    check_metrla()
    check_cityscapes()
    check_acoustic_vae()
    check_vision_generator()
    check_vision_discriminator()
    check_utility_vae()
    check_semantic_interface()
    check_utility_checkpoint()
    check_acoustic_checkpoint()
    check_vision_checkpoint()
    check_dp_attachment()
    check_fl_clients()

    banner("Validation Summary")
    total = passed + failed + warned + skipped
    print(f"\n  Total checks : {total}")
    print(f"  ✅ PASS      : {passed}")
    print(f"  ⚠️  WARN      : {warned}")
    print(f"  ❌ FAIL      : {failed}")
    print(f"  ⏭️  SKIP      : {skipped}")

    if failed == 0:
        print("\n  🎉 All critical checks passed. Project is submission-ready.")
    else:
        print(f"\n  ⛔ {failed} check(s) failed. Resolve FAIL items before submission.")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
