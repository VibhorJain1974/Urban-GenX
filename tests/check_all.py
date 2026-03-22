import os
import sys
import math
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def ok(msg: str):
    print(f"[PASS] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def fail(msg: str):
    print(f"[FAIL] {msg}")


def skip(msg: str):
    print(f"[SKIP] {msg}")


def exists(path: str) -> bool:
    return Path(path).exists()


def is_finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def check_environment():
    banner("1. Environment")
    try:
        import torch
        ok(f"PyTorch: {torch.__version__}")
    except Exception as e:
        fail(f"PyTorch import failed: {e}")

    try:
        import flwr
        ok(f"Flower: {flwr.__version__}")
    except Exception as e:
        fail(f"Flower import failed: {e}")

    try:
        import librosa
        ok(f"librosa: {librosa.__version__}")
    except Exception as e:
        fail(f"librosa import failed: {e}")

    try:
        from sentence_transformers import SentenceTransformer  # noqa
        ok("sentence-transformers import OK")
    except Exception as e:
        warn(f"sentence-transformers import failed, fallback mode likely active: {e}")


def check_utility():
    banner("2. Utility / Traffic VAE")

    ckpt_path = ROOT / "checkpoints" / "utility_traffic_checkpoint.pth"
    if not ckpt_path.exists():
        fail(f"Missing checkpoint: {ckpt_path}")
        return

    try:
        from torch.utils.data import DataLoader
        from src.utils.data_loader import METRLADataset
        from models.utility_vae import build_traffic_vae, UtilityVAE

        ck = torch.load(ckpt_path, map_location="cpu")
        ok(f"Checkpoint found at epoch {ck.get('epoch', '?')}")

        ds = METRLADataset(str(ROOT / "data" / "raw" / "metr-la" / "metr-la.h5"))
        dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
        x, _ = next(iter(dl))
        x = x.view(x.size(0), -1)

        model = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=64)
        model.load_state_dict(ck["model"])
        model.eval()

        with torch.no_grad():
            recon, mu, lv = model(x)
            loss = UtilityVAE.loss(recon, x, mu, lv, beta=1.0)

        ok(f"Traffic forward pass OK | recon={tuple(recon.shape)} | loss={float(loss):.4f}")

        if "avg_loss" in ck and "avg_val" in ck:
            ok(f"Stored metrics | train={ck['avg_loss']:.4f} | val={ck['avg_val']:.4f}")

    except Exception as e:
        fail(f"Utility check failed: {e}")
        traceback.print_exc()


def check_acoustic():
    banner("3. Acoustic VAE")

    ckpt_path = ROOT / "checkpoints" / "acoustic_checkpoint.pth"
    best_path = ROOT / "checkpoints" / "acoustic_best.pth"

    if not ckpt_path.exists() and not best_path.exists():
        warn("No acoustic checkpoint found yet. Finish train_acoustic.py first.")
        return

    chosen = best_path if best_path.exists() else ckpt_path

    try:
        from models.acoustic_vae import AcousticVAE

        ck = torch.load(chosen, map_location="cpu")
        cfg = ck.get("model_config", {})
        model = AcousticVAE(
            mfcc_bins=cfg.get("mfcc_bins", 40),
            time_frames=cfg.get("time_frames", 128),
            latent_dim=cfg.get("latent_dim", 64),
        )
        model.load_state_dict(ck["model"])
        model.eval()

        with torch.no_grad():
            synth = model.generate(n_samples=4)

        ok(f"Acoustic checkpoint OK: {chosen.name}")
        ok(f"Generated MFCC batch shape: {tuple(synth.shape)}")
        ok(f"Value range: [{float(synth.min()):.4f}, {float(synth.max()):.4f}]")

        if "avg_loss" in ck:
            ok(f"Stored train loss: {float(ck['avg_loss']):.4f}")
        if "avg_val_loss" in ck:
            ok(f"Stored val loss: {float(ck['avg_val_loss']):.4f}")

    except Exception as e:
        fail(f"Acoustic check failed: {e}")
        traceback.print_exc()


def check_vision():
    banner("4. Vision DP-GAN")

    ckpt_path = ROOT / "checkpoints" / "vision_checkpoint.pth"
    if not ckpt_path.exists():
        warn("No vision checkpoint found yet. Finish train_vision.py first.")
        return

    try:
        from models.vision_gan import Generator

        ck = torch.load(ckpt_path, map_location="cpu")
        G = Generator(noise_dim=100, num_classes=35)
        G.load_state_dict(ck["generator"])
        G.eval()

        cond = torch.zeros(2, 35, 64, 64)
        cond[:, 7, :, :] = 1.0  # simple road prior

        with torch.no_grad():
            fake = G(cond)

        ok(f"Vision checkpoint OK: epoch {ck.get('epoch', '?')}")
        ok(f"Generated image tensor shape: {tuple(fake.shape)}")
        ok(f"Value range: [{float(fake.min()):.4f}, {float(fake.max()):.4f}]")

        d_loss = ck.get("d_loss", None)
        g_loss = ck.get("g_loss", None)

        if d_loss is not None:
            if is_finite_number(d_loss):
                ok(f"d_loss finite: {float(d_loss):.4f}")
            else:
                warn(f"d_loss is not finite: {d_loss}")
        if g_loss is not None:
            if is_finite_number(g_loss):
                ok(f"g_loss finite: {float(g_loss):.4f}")
            else:
                warn(f"g_loss is not finite: {g_loss}")

    except Exception as e:
        fail(f"Vision check failed: {e}")
        traceback.print_exc()


def check_semantic():
    banner("5. Semantic Interface")

    try:
        from models.transformer_core import SemanticInterface

        si = SemanticInterface(use_sbert=True)
        preset = si.query("busy intersection with heavy traffic")
        cond = si.build_condition_tensor(preset, img_size=64, num_classes=35, batch_size=2)

        ok(f"Semantic query matched preset: {preset.get('scene_name', 'UNKNOWN')}")
        ok(f"Condition tensor shape: {tuple(cond.shape)}")

    except Exception as e:
        fail(f"Semantic interface check failed: {e}")
        traceback.print_exc()


def check_population():
    banner("6. Population Module")

    candidates = [
        ROOT / "models" / "population_vae.py",
        ROOT / "models" / "population_model.py",
        ROOT / "src" / "training" / "train_population.py",
    ]

    found = [str(p) for p in candidates if p.exists()]
    if not found:
        skip("Population module not implemented in current repo. Mark as future work.")
        return

    ok(f"Population-related files found: {found}")


def check_federated():
    banner("7. Federated Learning")

    try:
        from src.federated.client_vision import VisionClient
        from src.federated.client_acoustic import AcousticClient

        v = VisionClient(client_id="0")
        a = AcousticClient(client_id="0")

        vp = v.get_parameters({})
        ap = a.get_parameters({})

        ok(f"Vision client parameter tensors: {len(vp)}")
        ok(f"Acoustic client parameter tensors: {len(ap)}")

        # This is the key sanity check:
        # Current repo mixes heterogeneous models in one FedAvg run, which is not valid.
        same_length = len(vp) == len(ap)
        same_shapes = same_length and all(
            tuple(x.shape) == tuple(y.shape)
            for x, y in zip(vp, ap)
        )

        if same_shapes:
            ok("Vision and Acoustic parameter structures match (unexpected but valid for FedAvg).")
        else:
            warn(
                "Current FL design mixes heterogeneous models (Vision D vs Acoustic VAE). "
                "Do NOT aggregate them in one FedAvg server. Use separate modality servers."
            )

        # Server import
        from src.federated.server import get_strategy
        strategy = get_strategy()
        ok(f"Server strategy import OK: {strategy.__class__.__name__}")

    except Exception as e:
        fail(f"Federated check failed: {e}")
        traceback.print_exc()


def main():
    banner("Urban-GenX | Final System Validation")

    check_environment()
    check_utility()
    check_acoustic()
    check_vision()
    check_semantic()
    check_population()
    check_federated()

    banner("Validation Finished")
    print("Review WARN / FAIL items before final submission.")


if __name__ == "__main__":
    main()
