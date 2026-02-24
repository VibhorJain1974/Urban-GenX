"""
Urban-GenX | Federated Learning Server
Strategy: FedAvg over Vision + Acoustic clients.
Simulation mode: runs all clients in-process (no network needed for local dev).
"""
# src/federated/server.py (add at the very top)

import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import flwr as fl
from flwr.server.strategy import FedAvg

def get_strategy():
    return FedAvg(
        fraction_fit=1.0,          # Use all available clients each round
        fraction_evaluate=1.0,
        min_fit_clients=2,         # Need both Vision + Acoustic nodes
        min_evaluate_clients=2,
        min_available_clients=2,
        # Optional: custom aggregation weights can be added here
    )

def run_federated_training(num_rounds: int = 10):
    """
    Run simulation-mode federated learning.
    All clients run in-process — no TCP sockets, no network config.
    Safe for Windows + 12GB RAM.
    """
    from src.federated.client_vision   import VisionClient
    from src.federated.client_acoustic import AcousticClient

    def client_fn(cid: str):
        """Factory: return the right client based on client ID."""
        if cid == "0":
            return VisionClient()
        elif cid == "1":
            return AcousticClient()
        raise ValueError(f"Unknown client ID: {cid}")

    print(f"[FLWR] Starting federated simulation | {num_rounds} rounds | 2 clients")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=get_strategy(),
        # RAM cap: limits concurrent clients to 1 to protect 12GB RAM
        ray_init_args={"num_cpus": 1, "num_gpus": 0}
    )

if __name__ == "__main__":
    run_federated_training(num_rounds=5)
