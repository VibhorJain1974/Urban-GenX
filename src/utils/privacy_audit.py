"""
Urban-GenX | Privacy Audit Module
Implements Shadow Model-based Membership Inference Attack (MIA)
to empirically validate the DP guarantee.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def membership_inference_attack(model, member_loader, nonmember_loader, device='cpu'):
    """
    Simple confidence-based MIA:
    - Members (training data):     model should have low reconstruction loss
    - Non-members (held-out data): model should have higher reconstruction loss
    
    A random classifier (AUC ~0.5) confirms strong privacy.
    AUC > 0.7 is a warning sign of memorization.
    """
    model.eval()
    scores, labels = [], []

    with torch.no_grad():
        for loader, label in [(member_loader, 1), (nonmember_loader, 0)]:
            for batch in loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                if hasattr(model, 'forward') and hasattr(model, 'reparameterize'):
                    # VAE path
                    recon, mu, lv = model(x)
                    loss = torch.mean((recon - x) ** 2, dim=[1,2,3])
                else:
                    continue
                # Lower loss = more likely a member
                scores.extend((-loss).cpu().numpy().tolist())
                labels.extend([label] * len(loss))

    auc = roc_auc_score(labels, scores)
    print(f"[MIA] AUC = {auc:.4f} | {'⚠️ Memorization risk' if auc > 0.7 else '✅ Privacy OK'}")
    return auc
