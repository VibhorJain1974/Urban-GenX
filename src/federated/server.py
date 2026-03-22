"""
Urban-GenX | Federated Learning Server  (FINAL FIXED VERSION)
==============================================================
Fix: heterogeneous model problem solved.
  - Vision clients federate Discriminator (D) weights.
  - Acoustic clients federate VAE weights.
  - These have DIFFERENT parameter shapes → cannot mix in one FedAvg run.
  SOLUTION: Run two separate Flower simulations sequentially.
            One for Vision nodes, one for Acoustic nodes.

Fix: num_examples correctly uses len(dataset), not len(dataloader).

Usage:
    python src/federated/server.py              # runs both modalities
    python src/federated/server.py --vision     # vision only
    python src/federated/server.py --acoustic   # acoustic only
"""

import os
import sys
import argparse

# ── Project root on sys.path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import flwr as fl
from flwr.server.strategy import FedAvg


# ─── Shared strategy factory ─────────────────────────────────────────────────
def make_strategy(min_clients: int = 2, local_steps: int = 5) -> FedAvg:
    """
    FedAvg strategy.
    Each client is run with local_steps gradient steps per round.
    """
    return FedAvg(
        fraction_fit=1.0,           # use all available clients per round
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=lambda rnd: {"local_steps": local_steps},
        on_evaluate_config_fn=lambda rnd: {"round": rnd},
    )


# ─── Modality-specific simulations ───────────────────────────────────────────
def run_vision_federation(num_rounds: int = 3, num_clients: int = 2) -> None:
    """
    Federate Vision Discriminator (D) weights across clients.
    Each client trains on a partition of Cityscapes.
    Requires: data/raw/cityscapes/  (skips gracefully if absent)
    """
    from src.federated.client_vision import VisionClient

    def client_fn(cid: str) -> VisionClient:
        return VisionClient(client_id=cid)

    print(f"\n{'='*60}")
    print(f"[FLWR-VISION] Starting Vision FL | rounds={num_rounds} | clients={num_clients}")
    print(f"{'='*60}")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=make_strategy(min_clients=num_clients, local_steps=5),
        ray_init_args={"num_cpus": 1, "num_gpus": 0},  # RAM-safe on 12GB
    )
    print("[FLWR-VISION] Simulation complete.")


def run_acoustic_federation(num_rounds: int = 3, num_clients: int = 2) -> None:
    """
    Federate AcousticVAE weights across clients.
    Each client trains on a partition of UrbanSound8K folds.
    Requires: data/raw/urbansound8k/  (already set up)
    """
    from src.federated.client_acoustic import AcousticClient

    def client_fn(cid: str) -> AcousticClient:
        return AcousticClient(client_id=cid)

    print(f"\n{'='*60}")
    print(f"[FLWR-ACOUSTIC] Starting Acoustic FL | rounds={num_rounds} | clients={num_clients}")
    print(f"{'='*60}")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=make_strategy(min_clients=num_clients, local_steps=10),
        ray_init_args={"num_cpus": 1, "num_gpus": 0},
    )
    print("[FLWR-ACOUSTIC] Simulation complete.")


# ─── CLI entry point ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Urban-GenX Federated Learning (modality-specific simulations)"
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Run only Vision FL simulation"
    )
    parser.add_argument(
        "--acoustic",
        action="store_true",
        help="Run only Acoustic FL simulation"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of FL rounds per modality (default: 3)"
    )
    args = parser.parse_args()

    # If neither flag is passed, run both
    run_both = not args.vision and not args.acoustic

    print("\n" + "="*60)
    print("  Urban-GenX | Multi-Modal Federated Learning Demo")
    print("  Modality-specific simulations (technically correct)")
    print("="*60)

    if args.vision or run_both:
        run_vision_federation(num_rounds=args.rounds)

    if args.acoustic or run_both:
        run_acoustic_federation(num_rounds=args.rounds)

    print("\n[FLWR] All federation rounds complete.")


if __name__ == "__main__":
    main()
