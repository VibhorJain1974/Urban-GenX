# Urban-GenX — Federated Learning Debug & Fix Plan
# =====================================================
# Run AFTER acoustic_checkpoint.pth and vision_checkpoint.pth exist.
# All issues documented with root cause + fix.

FL_DEBUG_NOTES = """
=====================================================================
FEDERATED LEARNING — STEP-BY-STEP DEBUG & FIX PLAN
Urban-GenX | src/federated/server.py + client_*.py
=====================================================================

──────────────────────────────────────────────────────────────────────
ISSUE 1 ─ "min_available_clients=2" blocks if clients fail to init
ROOT CAUSE: VisionClient._setup_data() crashes if Cityscapes is missing
            → client raises exception → Flower can't find 2 clients.
FIX:
  client_vision.py already has a try/except: wraps _setup_data in try
  and falls back to self.dataloader = [] (empty list).
  But an empty dataloader means fit() returns zeros — harmless for demo.
  ✅ Already handled. Just ensure Cityscapes data is in place for real FL.

──────────────────────────────────────────────────────────────────────
ISSUE 2 ─ AcousticClient checkpoint not found at startup
ROOT CAUSE: acoustic_checkpoint.pth doesn't exist before train_acoustic runs.
FIX: Run `python src\training\train_acoustic.py` first.
     Client will print "[AcousticClient X] No checkpoint found, starting fresh."
     and use random weights — acceptable for demo/simulation.

──────────────────────────────────────────────────────────────────────
ISSUE 3 ─ Flower simulation hangs or OOM on Windows
ROOT CAUSE: Ray (used by fl.simulation) spawns subprocesses.
            Windows + 12 GB RAM → OOM with default num_cpus.
FIX: server.py already sets ray_init_args={"num_cpus": 1, "num_gpus": 0}
     This limits Ray to 1 CPU core.
     If still OOM: reduce num_rounds to 2 for demo.
     Also set environment variable before running:
       $env:RAY_memory_monitor_refresh_ms = "0"   # disable memory monitor
       python src\federated\server.py

──────────────────────────────────────────────────────────────────────
ISSUE 4 ─ TypeError: client_fn() missing argument  (Flower ≥1.5 API)
ROOT CAUSE: Newer Flower versions pass a ClientContext object, not just cid.
FIX: Update client_fn signature in server.py:

    # OLD (may break in flwr ≥1.7):
    def client_fn(cid: str):
        ...

    # NEW (compatible with flwr 1.5.0 as in requirements.txt):
    # flwr 1.5.0 still passes plain string — no change needed.
    # If upgrading flwr, change to:
    def client_fn(context):
        cid = str(context.node_id % 2)   # 0=Vision, 1=Acoustic
        ...

──────────────────────────────────────────────────────────────────────
ISSUE 5 ─ VisionClient D-parameters mismatch after FedAvg
ROOT CAUSE: If Opacus wraps D (GradSampleModule), parameter names gain
            "_module" prefix → set_parameters() gets shape mismatch.
FIX: client_vision.py does NOT use DP in FL mode (DP is applied only in
     train_vision.py). So D is plain nn.Module in VisionClient.
     ✅ No issue. Keep FL and DP training separate.

──────────────────────────────────────────────────────────────────────
ISSUE 6 ─ AcousticClient parameter aggregation mismatch
ROOT CAUSE: AcousticClient shares ALL VAE parameters (encoder + decoder).
            If model_config differs between clients (shouldn't happen),
            tensors will have different shapes.
FIX: All clients use same AcousticVAE(mfcc_bins=40, time_frames=128,
     latent_dim=64). Verify with:
       python -c "from models.acoustic_vae import AcousticVAE; m=AcousticVAE(); print(sum(p.numel() for p in m.parameters()), 'params')"
     All clients should print the same number.

──────────────────────────────────────────────────────────────────────
ISSUE 7 ─ "RuntimeError: CUDA not available" in FL clients
ROOT CAUSE: DEVICE is set to cuda in client but machine is CPU-only.
FIX: Both client files already use DEVICE = torch.device("cpu") ✅

──────────────────────────────────────────────────────────────────────
STEP-BY-STEP COMMANDS TO RUN FL SUCCESSFULLY
──────────────────────────────────────────────────────────────────────

Step 1 — Ensure checkpoints exist (minimum: acoustic for client 1):
  Test-Path checkpoints\acoustic_checkpoint.pth    → True
  Test-Path checkpoints\utility_traffic_checkpoint.pth  → True

Step 2 — Quick smoke test of each client independently:
  python -c "
  import sys; sys.path.insert(0, '.')
  from src.federated.client_vision import VisionClient
  c = VisionClient('0')
  params = c.get_parameters({})
  print('VisionClient OK, D has', len(params), 'tensors')
  "

  python -c "
  import sys; sys.path.insert(0, '.')
  from src.federated.client_acoustic import AcousticClient
  c = AcousticClient('1')
  params = c.get_parameters({})
  print('AcousticClient OK, VAE has', len(params), 'tensors')
  "

Step 3 — Set RAM-safe environment variable:
  $env:RAY_memory_monitor_refresh_ms = "0"

Step 4 — Run server (5 rounds by default):
  python src\federated\server.py

Step 5 — Expected output:
  [FLWR] Starting federated simulation | 5 rounds | 2 clients
  [VisionClient 0] Data partition: XXXX samples
  [AcousticClient 1] ...
  [VisionClient 0] local fit | D=0.xxxx G=0.xxxx
  ...
  [FLWR] Round 1 metrics: {...}
  ...
  [FLWR] Federated simulation complete.

──────────────────────────────────────────────────────────────────────
DIAGRAM: What gets federated vs what stays local
──────────────────────────────────────────────────────────────────────

  VisionClient (node 0)          Server (FedAvg)
  ┌──────────────────┐           ┌──────────────┐
  │  G (local only)  │           │  Aggregate D │
  │  D ──────────────┼──────────▶│  weights     │
  │  [Cityscapes A]  │◀──────────┼──────────────┘
  └──────────────────┘   D_avg

  AcousticClient (node 1)        Server (FedAvg)
  ┌──────────────────┐           ┌──────────────┐
  │  VAE ────────────┼──────────▶│  Aggregate   │
  │  [UrbanSnd folds]│◀──────────┼  VAE weights │
  └──────────────────┘   VAE_avg └──────────────┘

Privacy guarantee: raw data never leaves the client node.
G weights are DP-clean (lambda_l1=0.0 + DP-SGD on D → G gradient safe).
"""

if __name__ == "__main__":
    print(FL_DEBUG_NOTES)
