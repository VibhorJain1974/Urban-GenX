"""
Urban-GenX | Federated Learning Server (IMPROVED)
=================================================
What changed:
  1. Keeps Vision and Acoustic as separate Flower simulations
  2. Adds weighted fit/evaluate metrics aggregation to remove FedAvg warnings
  3. Prints aggregated round metrics more clearly
  4. Keeps the same CLI contract: --vision / --acoustic / --rounds
"""

import argparse
import numbers
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import flwr as fl
from flwr.server.strategy import FedAvg


MetricList = List[Tuple[int, Dict[str, float]]]


def _to_float_if_numeric(value):
    if isinstance(value, numbers.Number):
        return float(value)
    return None



def weighted_metric_avg(metrics: MetricList) -> Dict[str, float]:
    """
    Flower metric aggregation callback.
    Receives [(num_examples, {metric_name: metric_value}), ...]
    and returns weighted averages for all numeric metrics.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics if num_examples is not None)
    if total_examples <= 0:
        return {}

    keys = set()
    for _, metric_dict in metrics:
        keys.update(metric_dict.keys())

    aggregated: Dict[str, float] = {}
    for key in sorted(keys):
        weighted_sum = 0.0
        used_examples = 0
        for num_examples, metric_dict in metrics:
            value = _to_float_if_numeric(metric_dict.get(key, None))
            if value is None:
                continue
            weighted_sum += num_examples * value
            used_examples += num_examples

        if used_examples > 0:
            aggregated[key] = weighted_sum / used_examples

    if aggregated:
        pretty = " | ".join(f"{k}={v:.4f}" for k, v in aggregated.items())
        print(f"[METRICS] Aggregated: {pretty}")

    return aggregated



def make_strategy(min_clients: int = 2, local_steps: int = 5) -> FedAvg:
    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=lambda rnd: {"local_steps": local_steps},
        on_evaluate_config_fn=lambda rnd: {"round": rnd},
        fit_metrics_aggregation_fn=weighted_metric_avg,
        evaluate_metrics_aggregation_fn=weighted_metric_avg,
    )



def run_vision_federation(num_rounds: int = 3, num_clients: int = 2) -> None:
    from src.federated.client_vision import VisionClient

    def client_fn(cid: str) -> VisionClient:
        return VisionClient(client_id=cid)

    print(f"\n{'=' * 60}")
    print(f"[FLWR-VISION] Starting Vision FL | rounds={num_rounds} | clients={num_clients}")
    print(f"{'=' * 60}")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=make_strategy(min_clients=num_clients, local_steps=5),
        ray_init_args={"num_cpus": 1, "num_gpus": 0},
    )
    print("[FLWR-VISION] Simulation complete.")



def run_acoustic_federation(num_rounds: int = 3, num_clients: int = 2) -> None:
    from src.federated.client_acoustic import AcousticClient

    def client_fn(cid: str) -> AcousticClient:
        return AcousticClient(client_id=cid)

    print(f"\n{'=' * 60}")
    print(f"[FLWR-ACOUSTIC] Starting Acoustic FL | rounds={num_rounds} | clients={num_clients}")
    print(f"{'=' * 60}")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=make_strategy(min_clients=num_clients, local_steps=10),
        ray_init_args={"num_cpus": 1, "num_gpus": 0},
    )
    print("[FLWR-ACOUSTIC] Simulation complete.")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Urban-GenX Federated Learning (modality-specific simulations)"
    )
    parser.add_argument("--vision", action="store_true", help="Run only Vision FL simulation")
    parser.add_argument("--acoustic", action="store_true", help="Run only Acoustic FL simulation")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds per modality (default: 3)")
    args = parser.parse_args()

    run_both = not args.vision and not args.acoustic

    print("\n" + "=" * 60)
    print("  Urban-GenX | Multi-Modal Federated Learning Demo")
    print("  Modality-specific simulations with metrics aggregation")
    print("=" * 60)

    if args.vision or run_both:
        run_vision_federation(num_rounds=args.rounds)

    if args.acoustic or run_both:
        run_acoustic_federation(num_rounds=args.rounds)

    print("\n[FLWR] All federation rounds complete.")


if __name__ == "__main__":
    main()
