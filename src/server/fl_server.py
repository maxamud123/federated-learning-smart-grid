"""
Federated Learning Server - Flower Framework
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025

Runs FedAvg aggregation across all connected clients.
Logs per-round metrics (loss, RMSE, MAE, communication cost).
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict
import numpy as np
import json
import os
from datetime import datetime


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = "experiments/logs"
os.makedirs(LOG_DIR, exist_ok=True)
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")


def log_round(round_num, metrics, log_path):
    """Append round metrics to a JSON-lines log file."""
    entry = {"round": round_num, **metrics}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Metric aggregation callbacks ─────────────────────────────────────────────
def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate client evaluation metrics using weighted average.
    Called by Flower after each round's evaluate phase.
    """
    total_samples = sum(num for num, _ in metrics)

    rmse = sum(m["rmse"] * num for num, m in metrics) / total_samples
    mae  = sum(m["mae"]  * num for num, m in metrics) / total_samples

    print(f"  Aggregated -> RMSE: {rmse:.6f} | MAE: {mae:.6f}")
    return {"rmse": rmse, "mae": mae}


# ── Custom FedAvg strategy with logging ───────────────────────────────────────
class LoggingFedAvg(FedAvg):
    """
    FedAvg with per-round metric logging.
    Extends Flower's built-in FedAvg strategy.
    """

    def __init__(self, log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path    = log_path
        self.round_count = 0

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client model updates (FedAvg)."""
        self.round_count = server_round
        aggregated       = super().aggregate_fit(server_round, results, failures)

        # Log communication info
        total_params = 0
        for _, fit_res in results:
            for arr in fit_res.parameters.tensors:
                total_params += len(arr)

        log_round(server_round, {
            "phase":        "fit",
            "num_clients":  len(results),
            "num_failures": len(failures),
            "total_bytes":  total_params,
        }, self.log_path)

        print(f"[Server] Round {server_round} - "
              f"Clients: {len(results)} | Failures: {len(failures)}")
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from all clients."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if results:
            loss_values = [r.loss for _, r in results]
            avg_loss    = np.mean(loss_values)
            log_round(server_round, {
                "phase":    "evaluate",
                "avg_loss": avg_loss,
            }, self.log_path)

        return aggregated


def build_strategy(num_rounds, min_clients, log_path, initial_parameters=None):
    """Build the FedAvg strategy with configuration for this experiment."""
    return LoggingFedAvg(
        log_path=log_path,
        fraction_fit=1.0,               # Use all available clients each round
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        initial_parameters=initial_parameters,
    )


def start_server(num_rounds=50, min_clients=3, server_address="0.0.0.0:8080",
                 experiment_name="baseline"):
    """
    Start the Flower FL server.

    Args:
        num_rounds:       Number of federated rounds (50 for full experiment)
        min_clients:      Minimum clients required before training starts
        server_address:   Host:port to listen on
        experiment_name:  Label for log file naming
    """
    log_path = os.path.join(LOG_DIR, f"{RUN_ID}_{experiment_name}.jsonl")
    print(f"[Server] Starting FL server")
    print(f"[Server] Address:    {server_address}")
    print(f"[Server] Rounds:     {num_rounds}")
    print(f"[Server] Min clients:{min_clients}")
    print(f"[Server] Log file:   {log_path}")

    strategy = build_strategy(
        num_rounds=num_rounds,
        min_clients=min_clients,
        log_path=log_path,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
