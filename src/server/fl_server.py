"""
Federated Learning Server - Flower Framework
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025

Fixes applied:
  - Early stopping with configurable patience
  - RMSE/MAE logged every round (not just avg_loss)
  - Actual vs compressed bytes logged with savings %
  - Per-client metrics (device, loss, bytes) logged per round
  - Round number injected into client config for LR scheduling
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, FitIns
from typing import List, Tuple, Optional, Dict
import numpy as np
import json
import os
from datetime import datetime


LOG_DIR = "experiments/logs"
os.makedirs(LOG_DIR, exist_ok=True)
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")


def log_round(round_num, metrics, log_path):
    entry = {"round": round_num, **metrics}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_samples = sum(num for num, _ in metrics)
    rmse = sum(m["rmse"] * num for num, m in metrics) / total_samples
    mae  = sum(m["mae"]  * num for num, m in metrics) / total_samples
    print(f"  Aggregated -> RMSE: {rmse:.6f} | MAE: {mae:.6f}")
    return {"rmse": rmse, "mae": mae}


class LoggingFedAvg(FedAvg):

    def __init__(self, log_path, patience=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path       = log_path
        self.patience       = patience
        self.best_loss      = float("inf")
        self.patience_count = 0
        self.stop_training  = False
        self.best_round     = 1

    def configure_fit(self, server_round, parameters, client_manager):
        """Inject round number into client config; stop if early stopping triggered."""
        if self.stop_training:
            return []
        instructions = super().configure_fit(server_round, parameters, client_manager)
        # Pass current round to each client so they can adjust learning rate
        return [
            (client, FitIns(fit_ins.parameters, {**fit_ins.config, "round": server_round}))
            for client, fit_ins in instructions
        ]

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if not results:
            return aggregated

        actual_bytes     = 0
        compressed_bytes = 0
        per_client_logs  = []

        for _, fit_res in results:
            # Actual bytes received over the network (full float32 arrays)
            client_actual = sum(len(arr) for arr in fit_res.parameters.tensors)
            actual_bytes += client_actual

            # Theoretical bytes if only non-zero values were transmitted (Top-K savings)
            client_compressed = int(fit_res.metrics.get("compressed_bytes", client_actual))
            compressed_bytes += client_compressed

            per_client_logs.append({
                "client_id":        int(fit_res.metrics.get("client_id", -1)),
                "device":           str(fit_res.metrics.get("device", "unknown")),
                "train_loss":       round(float(fit_res.metrics.get("train_loss", 0.0)), 6),
                "local_epochs":     int(fit_res.metrics.get("epochs", 0)),
                "lr":               round(float(fit_res.metrics.get("lr", 0.001)), 6),
                "actual_bytes":     client_actual,
                "compressed_bytes": client_compressed,
            })

        savings_pct = (1.0 - compressed_bytes / max(actual_bytes, 1)) * 100

        log_round(server_round, {
            "phase":             "fit",
            "num_clients":       len(results),
            "num_failures":      len(failures),
            "actual_bytes":      actual_bytes,
            "compressed_bytes":  compressed_bytes,
            "savings_pct":       round(savings_pct, 1),
            "clients":           per_client_logs,
        }, self.log_path)

        print(f"[Server] Round {server_round} | "
              f"Clients: {len(results)} | Failures: {len(failures)} | "
              f"Comm savings: {savings_pct:.1f}%")
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if not results:
            return aggregated

        # results is List[Tuple[ClientProxy, EvaluateRes]] — use .num_examples / .metrics
        total_samples = sum(r.num_examples for _, r in results)
        avg_loss = float(np.mean([r.loss for _, r in results]))
        rmse     = sum(r.metrics.get("rmse", 0.0) * r.num_examples for _, r in results) / total_samples
        mae      = sum(r.metrics.get("mae",  0.0) * r.num_examples for _, r in results) / total_samples

        log_round(server_round, {
            "phase":    "evaluate",
            "avg_loss": round(avg_loss, 6),
            "rmse":     round(float(rmse), 6),
            "mae":      round(float(mae),  6),
        }, self.log_path)

        # Early stopping: check if loss improved by at least 1e-4
        if avg_loss < self.best_loss - 1e-4:
            self.best_loss      = avg_loss
            self.best_round     = server_round
            self.patience_count = 0
        else:
            self.patience_count += 1
            remaining = self.patience - self.patience_count
            print(f"  [EarlyStopping] No improvement. Patience: {self.patience_count}/{self.patience} "
                  f"(best loss: {self.best_loss:.6f} @ round {self.best_round})")
            if self.patience_count >= self.patience:
                print(f"[Server] Early stopping triggered at round {server_round}. "
                      f"Best round: {self.best_round} | Best loss: {self.best_loss:.6f}")
                self.stop_training = True

        return aggregated


def build_strategy(num_rounds, min_clients, log_path, patience=10):
    return LoggingFedAvg(
        log_path=log_path,
        patience=patience,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )


def start_server(num_rounds=50, min_clients=3, server_address="0.0.0.0:8080",
                 experiment_name="baseline", patience=10):
    log_path = os.path.join(LOG_DIR, f"{RUN_ID}_{experiment_name}.jsonl")
    print(f"[Server] Starting  | Address: {server_address} | "
          f"Rounds: {num_rounds} | Min clients: {min_clients} | "
          f"Early-stop patience: {patience}")
    print(f"[Server] Log file: {log_path}")

    strategy = build_strategy(
        num_rounds=num_rounds,
        min_clients=min_clients,
        log_path=log_path,
        patience=patience,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
