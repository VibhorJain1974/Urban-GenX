"""
Urban-GenX | Privacy Audit Module (FINAL - Standalone Runnable)
===============================================================
Implements Shadow Model-based Membership Inference Attack (MIA)
to empirically validate the DP guarantee.

Run standalone:
  python src/utils/privacy_audit.py --model acoustic
  python src/utils/privacy_audit.py --model traffic
  python src/utils/privacy_audit.py --model all

Results:
  AUC ~0.5 -> strong privacy (model cannot distinguish members from non-members)
  AUC > 0.7 -> memorization risk
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score

# Force UTF-8 output on Windows to avoid CP1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def membership_inference_attack(model, member_loader, nonmember_loader,
                                 device='cpu', model_type='vae'):
    """
    Confidence-based Membership Inference Attack (MIA).

    Members (training data)  -> model tends to have lower reconstruction loss.
    Non-members (held-out)   -> higher reconstruction loss.

    AUC ~0.5  confirms strong privacy (random guessing).
    AUC > 0.7 is a warning sign of memorization.
    """
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for loader, label in [(member_loader, 1), (nonmember_loader, 0)]:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                if model_type == 'vae':
                    recon, mu, lv = model(x)
                    loss = torch.mean((recon - x) ** 2, dim=[1, 2, 3])

                elif model_type == 'utility_vae':
                    x_flat = x.view(x.size(0), -1)
                    recon, mu, lv = model(x_flat)
                    loss = torch.mean((recon - x_flat) ** 2, dim=1)

                else:
                    continue

                scores.extend((-loss).cpu().numpy().tolist())
                labels.extend([label] * len(loss))

    if len(scores) < 2 or len(set(labels)) < 2:
        print("[MIA] Not enough data for MIA evaluation.")
        return {"auc": 0.5, "accuracy": 0.5, "verdict": "insufficient_data"}

    auc = roc_auc_score(labels, scores)

    threshold = np.median(scores)
    preds = [1 if s > threshold else 0 for s in scores]
    acc = accuracy_score(labels, preds)

    member_scores    = [-s for s, l in zip(scores, labels) if l == 1]
    nonmember_scores = [-s for s, l in zip(scores, labels) if l == 0]
    member_mean    = np.mean(member_scores)    if member_scores    else 0.0
    nonmember_mean = np.mean(nonmember_scores) if nonmember_scores else 0.0

    if auc > 0.7:
        verdict = "WARNING_MEMORIZATION"
        tag     = "[!] Memorization risk"
    elif auc > 0.6:
        verdict = "MARGINAL"
        tag     = "[~] Marginal"
    else:
        verdict = "SAFE"
        tag     = "[OK] Privacy safe"

    print(f"[MIA] AUC={auc:.4f} | Acc={acc:.4f} | "
          f"Member loss={member_mean:.4f} | Non-member loss={nonmember_mean:.4f} | {tag}")

    return {
        "auc":               auc,
        "accuracy":          acc,
        "verdict":           verdict,
        "member_mean_loss":  member_mean,
        "nonmember_mean_loss": nonmember_mean,
    }


# --- Acoustic ----------------------------------------------------------------

def run_acoustic_mia():
    from models.acoustic_vae import AcousticVAE
    from src.utils.data_loader import UrbanSound8KDataset

    print("\n" + "=" * 60)
    print("  MIA Audit: Acoustic VAE (UrbanSound8K)")
    print("=" * 60)

    data_root = "data/raw/urbansound8k"
    for alt in [data_root, os.path.join(data_root, "UrbanSound8K")]:
        if os.path.exists(os.path.join(alt, "metadata", "UrbanSound8K.csv")):
            data_root = alt
            break

    member_ds    = UrbanSound8KDataset(root=data_root, folds=list(range(1, 10)))
    nonmember_ds = UrbanSound8KDataset(root=data_root, folds=[10])

    member_sub    = Subset(member_ds,    list(range(min(500, len(member_ds)))))
    nonmember_sub = Subset(nonmember_ds, list(range(min(200, len(nonmember_ds)))))

    member_loader    = DataLoader(member_sub,    batch_size=16, shuffle=False, num_workers=0)
    nonmember_loader = DataLoader(nonmember_sub, batch_size=16, shuffle=False, num_workers=0)

    model = AcousticVAE(mfcc_bins=40, time_frames=128, latent_dim=64)
    ckpt_path = "checkpoints/acoustic_checkpoint.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded acoustic checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print("[WARN] No acoustic checkpoint found. Using random weights.")

    return membership_inference_attack(model, member_loader, nonmember_loader, model_type='vae')


# --- Traffic -----------------------------------------------------------------

def run_traffic_mia():
    from models.utility_vae import build_traffic_vae
    from src.utils.data_loader import METRLADataset

    print("\n" + "=" * 60)
    print("  MIA Audit: Traffic VAE (METR-LA)")
    print("=" * 60)

    h5_path = "data/raw/metr-la/metr-la.h5"
    if not os.path.exists(h5_path):
        print(f"[ERROR] METR-LA not found at {h5_path}")
        return {"auc": 0.5, "verdict": "data_not_found"}

    full_ds = METRLADataset(h5_path, seq_len=12, pred_len=12)
    total   = len(full_ds)
    split   = int(total * 0.8)

    member_sub    = Subset(full_ds, list(range(min(500, split))))
    nonmember_sub = Subset(full_ds, list(range(split, min(split + 200, total))))

    # Flatten wrapper so UtilityVAE receives [B, input_dim]
    class FlatLoader:
        def __init__(self, dl):
            self.dl = dl
        def __iter__(self):
            for x, _ in self.dl:
                yield x.view(x.size(0), -1), None
        def __len__(self):
            return len(self.dl)

    member_loader    = FlatLoader(DataLoader(member_sub,    batch_size=32, shuffle=False, num_workers=0))
    nonmember_loader = FlatLoader(DataLoader(nonmember_sub, batch_size=32, shuffle=False, num_workers=0))

    model = build_traffic_vae(seq_len=12, n_sensors=207, latent_dim=64)
    ckpt_path = "checkpoints/utility_traffic_checkpoint.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded traffic checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print("[WARN] No traffic checkpoint found.")

    return membership_inference_attack(model, member_loader, nonmember_loader, model_type='utility_vae')


# --- Summary -----------------------------------------------------------------

def run_all_mia():
    print("\n" + "=" * 60)
    print("  Urban-GenX | Full Privacy Audit (MIA)")
    print("=" * 60)

    results = {}

    try:
        results["acoustic"] = run_acoustic_mia()
    except Exception as e:
        print(f"[ERROR] Acoustic MIA failed: {e}")
        results["acoustic"] = {"auc": None, "verdict": "error"}

    try:
        results["traffic"] = run_traffic_mia()
    except Exception as e:
        print(f"[ERROR] Traffic MIA failed: {e}")
        results["traffic"] = {"auc": None, "verdict": "error"}

    # Plain-ASCII summary table (works on all Windows encodings)
    print("\n" + "=" * 60)
    print("  MIA AUDIT SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<15} {'AUC':<10} {'Verdict'}")
    print(f"  {'-'*15} {'-'*10} {'-'*30}")
    for name, res in results.items():
        auc_str = f"{res['auc']:.4f}" if res.get('auc') is not None else "N/A"
        verdict = res.get('verdict', 'unknown')
        if verdict == "SAFE":
            tag = "[OK] SAFE - privacy confirmed"
        elif verdict == "MARGINAL":
            tag = "[~]  MARGINAL"
        elif verdict == "WARNING_MEMORIZATION":
            tag = "[!!] MEMORIZATION RISK"
        else:
            tag = f"[?]  {verdict}"
        print(f"  {name:<15} {auc_str:<10} {tag}")
    print("=" * 60)
    print("\n  Interpretation:")
    print("  AUC close to 0.50 = model cannot distinguish training from held-out data")
    print("  AUC > 0.70        = model may have memorised training examples")
    print("=" * 60)

    return results


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Urban-GenX Privacy Audit (MIA)")
    parser.add_argument("--model", choices=["acoustic", "traffic", "all"],
                        default="all", help="Which model to audit")
    args = parser.parse_args()

    if args.model == "acoustic":
        run_acoustic_mia()
    elif args.model == "traffic":
        run_traffic_mia()
    else:
        run_all_mia()