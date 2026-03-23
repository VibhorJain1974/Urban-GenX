"""
fl_debug_plan.py
================
Urban-GenX — Federated Learning Debug & Fix Plan
=================================================

This file is both documentation AND a runnable diagnostic script.
Run it to auto-detect the most common FL configuration problems:

    python fl_debug_plan.py

It checks Flower version, client API compatibility, checkpoint availability,
RAM budget, and ray/in-process simulation config.

---

STEP-BY-STEP FL DEBUG PLAN
===========================

PROBLEM CATEGORIES
------------------
A. Import / version errors
B. Flower API mismatch (0.x vs 1.x)
C. Missing or incompatible checkpoints
D. RAM / CPU resource exhaustion
E. Ray initialisation errors
F. Client parameter shape mismatch after FedAvg
G. Opacus + FL interaction (DP budget counting)
H. Data path errors in client workers

---

A. IMPORT / VERSION ERRORS
---------------------------

Symptom:
    ImportError: cannot import name 'NumPyClient' from 'flwr.client'
    OR
    ModuleNotFoundError: No module named 'ray'

Fix:
    pip install flwr==1.5.0

Verify:
    python -c "import flwr; print(flwr.__version__)"  # expect 1.5.0

---

B. FLOWER 1.x API — CORRECT CLIENT SIGNATURE
---------------------------------------------

Flower 1.5 requires these exact method signatures.
Any deviation causes silent failures or AttributeError at round 1.

CORRECT (Flower 1.5):
    class VisionClient(fl.client.NumPyClient):
        def get_parameters(self, config: dict) -> list[np.ndarray]:
            ...
        def set_parameters(self, parameters: list[np.ndarray]) -> None:
            ...
        def fit(self, parameters: list[np.ndarray], config: dict
                ) -> tuple[list[np.ndarray], int, dict]:
            ...
        def evaluate(self, parameters: list[np.ndarray], config: dict
                     ) -> tuple[float, int, dict]:
            ...

WRONG (old Flower 0.x):
    def get_parameters(self) -> list:           # missing config
    def fit(self, parameters, config):          # wrong return type
    def evaluate(self, parameters, config):     # wrong return type

Fix for client_acoustic.py (Flower 1.5 template):
    ─────────────────────────────────────────────
    import flwr as fl
    import numpy as np
    import torch
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from models.acoustic_vae import AcousticVAE

    class AcousticClient(fl.client.NumPyClient):
        def __init__(self):
            self.model = AcousticVAE()
            self.device = torch.device("cpu")
            self.model.to(self.device)
            ckpt = "checkpoints/acoustic_checkpoint.pth"
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location="cpu")
                self.model.load_state_dict(state["model_state_dict"])

        def get_parameters(self, config):
            return [v.cpu().numpy() for v in self.model.state_dict().values()]

        def set_parameters(self, parameters):
            keys = list(self.model.state_dict().keys())
            state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
            self.model.load_state_dict(state, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            self.model.train()
            optim = torch.optim.Adam(self.model.parameters(), lr=5e-4)
            # Minimal local training loop (real impl loads UrbanSound8K)
            losses = []
            for _ in range(config.get("local_steps", 5)):
                x = torch.randn(4, 1, 40, 128)   # replace with real batch
                recon, mu, lv = self.model(x)
                recon_loss = torch.nn.functional.mse_loss(recon, x)
                kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()) / 64
                loss = recon_loss + kl
                optim.zero_grad(); loss.backward(); optim.step()
                losses.append(loss.item())
            avg_loss = float(np.mean(losses))
            return self.get_parameters(config), 4, {"loss": avg_loss}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            self.model.eval()
            x = torch.randn(4, 1, 40, 128)
            with torch.no_grad():
                recon, mu, lv = self.model(x)
                loss = torch.nn.functional.mse_loss(recon, x).item()
            return loss, 4, {"val_loss": loss}
    ─────────────────────────────────────────────

---

C. CHECKPOINT AVAILABILITY
---------------------------

server.py starts clients in-process; clients attempt to load checkpoints
from the current working directory.  If checkpoints are missing:

Symptom:
    FileNotFoundError: checkpoints/acoustic_checkpoint.pth

Fix:
    1. Run acoustic training first:
       python src/training/train_acoustic.py

    2. Run vision training for at least 1 epoch (creates partial checkpoint):
       python src/training/train_vision.py

    3. Or: comment out checkpoint loading in client __init__ to allow
       FL simulation to run with random weights (for debugging only).

---

D. RAM / CPU EXHAUSTION
------------------------

System: Intel i5-8250U, 12 GB RAM, Windows.
Each client runs in the same process to avoid multiprocessing overhead.

Symptom:
    OSError: [WinError 1455] The paging file is too small
    OR process killed / OOM

Fixes:
    1. In server.py, ensure in-process simulation:
       fl.simulation.start_simulation(
           client_fn=client_fn,
           num_clients=2,
           config=fl.server.ServerConfig(num_rounds=5),
           strategy=strategy,
           ray_init_args={"num_cpus": 1, "include_dashboard": False},
       )

    2. Reduce client batch sizes:
       VisionClient: batch_size=2 (already set)
       AcousticClient: batch_size=4

    3. Reduce local_steps to 2–3 for debugging:
       config={"local_steps": 2}

    4. Free memory between rounds by calling:
       import gc; gc.collect()
       torch.cuda.empty_cache()  # no-op on CPU but harmless

---

E. RAY INITIALISATION ERRORS
------------------------------

Symptom (Windows):
    RuntimeError: ray.init() has already been called
    OR
    AssertionError: Ray must be initialized before calling ray.get()

Fix:
    # At top of server.py, add guard:
    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)

Alternative (avoid Ray entirely — Flower virtual client engine):
    # Flower 1.5 supports in-process simulation without Ray:
    fl.simulation.start_simulation(
        ...
        ray_init_args={"num_cpus": 1},
    )
    # OR use the VirtualClientEngine (no Ray required):
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

---

F. PARAMETER SHAPE MISMATCH AFTER FEDAVG
------------------------------------------

Symptom:
    RuntimeError: Error(s) in loading state_dict for AcousticVAE:
        size mismatch for enc.0.weight: copying a param with shape torch.Size([...])
        from checkpoint, the shape in current model is torch.Size([...]).

Root cause: get_parameters() returns ALL layer weights (including non-trainable buffers).
If model architecture changed between rounds, shapes diverge.

Fix:
    # Only federate trainable parameters, not buffers:
    def get_parameters(self, config):
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for server_param, client_param in zip(parameters, self.model.parameters()):
            client_param.data = torch.tensor(server_param)

Note: state_dict() includes buffers (e.g., BatchNorm running_mean).
Parameters() only returns trainable tensors. Choose consistently.

---

G. OPACUS + FEDERATED LEARNING
--------------------------------

Opacus DP and FLWR interact in non-obvious ways for the Vision client:

Issue 1: DP budget counts LOCAL steps, not FL rounds.
    With 5 rounds × 5 local_steps = 25 DP steps.
    Ensure remaining_epochs accounts for FL rounds:
        total_fl_steps = num_rounds * local_steps
        # Pass to PrivacyEngine as equivalent epochs

Issue 2: VisionClient federates only Discriminator (D).
    Generator (G) is trained locally and NEVER sent to server.
    This is correct for DP-GAN: only D benefits from cross-city aggregation.
    Ensure get_parameters() returns ONLY D parameters:

        def get_parameters(self, config):
            return [p.cpu().detach().numpy() for p in self.D.parameters()]

Issue 3: PrivacyEngine is NOT compatible with parameter replacement via set_parameters.
    After FedAvg, server sends averaged D weights back.
    To load these into a DP-wrapped model without breaking the accountant:

        # Do NOT use model.load_state_dict() on DP-wrapped model mid-training.
        # Instead, update the original (unwrapped) D, then re-wrap:
        with torch.no_grad():
            for p_local, p_server in zip(self.D_original.parameters(), parameters):
                p_local.data.copy_(torch.tensor(p_server))
        # The DP engine wraps self.D_original, so updates propagate automatically.

Issue 4: Gradient accumulation incompatible with Opacus Poisson sampling.
    Do NOT use gradient accumulation steps > 1 in FL mode.

---

H. DATA PATH ERRORS IN CLIENT WORKERS
---------------------------------------

Symptom:
    FileNotFoundError: data/raw/cityscapes/leftImg8bit/train
    (even though the path exists from repo root)

Root cause: Flower simulation workers may have different CWD.

Fix:
    # Use absolute paths based on __file__ location:
    DATA_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    )
    CITYSCAPES_PATH = os.path.join(DATA_ROOT, "cityscapes")
    US8K_PATH       = os.path.join(DATA_ROOT, "UrbanSound8K")

---

COMPLETE RECOMMENDED server.py (minimal working version)
---------------------------------------------------------

    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    import flwr as fl
    import ray
    from federated.client_vision   import VisionClient
    from federated.client_acoustic import AcousticClient

    def client_fn(cid: str) -> fl.client.NumPyClient:
        if cid == "0":
            return VisionClient(client_id="0")
        elif cid == "1":
            return AcousticClient()
        raise ValueError(f"Unknown client id: {cid}")

    def run_federated_training(num_rounds: int = 5) -> None:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
        if not ray.is_initialized():
            ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=2,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            ray_init_args={"num_cpus": 1},
        )
        print(f"FL training complete. History: {history.metrics_distributed}")

    if __name__ == "__main__":
        run_federated_training(num_rounds=5)

---

DEBUGGING CHECKLIST (run in order)
------------------------------------
"""

import os, sys, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) \
       if "__file__" in dir() else os.getcwd()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def check(label, condition, fix):
    icon = f"{GREEN}✅{RESET}" if condition else f"{RED}❌{RESET}"
    print(f"  {icon}  {label}")
    if not condition:
        print(f"      {YELLOW}FIX: {fix}{RESET}")
    return condition


print("\n" + "="*60)
print("  Urban-GenX — FL Debug Diagnostics")
print("="*60 + "\n")

# 1. Flower version
try:
    import flwr
    flwr_ok = flwr.__version__ >= "1.5"
    check(f"flwr=={flwr.__version__}", flwr_ok,
          "pip install flwr==1.5.0")
except ImportError:
    check("flwr installed", False, "pip install flwr==1.5.0")
    flwr_ok = False

# 2. Ray installed
try:
    import ray
    check(f"ray=={ray.__version__}", True, "")
    ray_ok = True
except ImportError:
    check("ray installed", False, "pip install ray==2.6.3")
    ray_ok = False

# 3. Checkpoints
ckpts = {
    "utility_traffic_checkpoint.pth": "python src/training/train_utility.py",
    "acoustic_checkpoint.pth":        "python src/training/train_acoustic.py",
    "vision_checkpoint.pth":          "python src/training/train_vision.py",
}
for ckpt_name, fix_cmd in ckpts.items():
    path = os.path.join(ROOT, "checkpoints", ckpt_name)
    exists = os.path.exists(path)
    check(f"checkpoints/{ckpt_name}", exists, fix_cmd)

# 4. RAM estimate
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    check(f"RAM available ({ram_gb:.1f} GB ≥ 8 GB)", ram_gb >= 8,
          "Close other applications; reduce batch_size in clients")
except ImportError:
    print(f"  {YELLOW}⚠️   psutil not installed — RAM check skipped{RESET}")

# 5. Client files
client_files = [
    "src/federated/server.py",
    "src/federated/client_vision.py",
    "src/federated/client_acoustic.py",
]
for f in client_files:
    path = os.path.join(ROOT, f)
    check(f"{f} exists", os.path.exists(path),
          f"Create {f} using templates in this debug plan")

# 6. Flower API check (get_parameters signature)
if flwr_ok:
    try:
        SRC = os.path.join(ROOT, "src")
        if SRC not in sys.path:
            sys.path.insert(0, SRC)
        from federated.client_vision import VisionClient
        import inspect
        sig = inspect.signature(VisionClient.get_parameters)
        has_config = "config" in sig.parameters
        check("VisionClient.get_parameters(config) — Flower 1.5 API", has_config,
              "Add 'config: dict' parameter to get_parameters()")
    except Exception as e:
        print(f"  {YELLOW}⚠️   Could not inspect VisionClient: {e}{RESET}")

print()
print("="*60)
print("  See inline comments above for full fix instructions.")
print("="*60 + "\n")
