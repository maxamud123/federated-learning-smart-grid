"""
Evaluation Utilities - RMSE, MAE, Memory, Communication Metrics
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025
"""

import numpy as np
import torch
import psutil
import os
import json
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.lstm_model import get_model_size_mb


# ── Forecasting accuracy metrics ─────────────────────────────────────────────
def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def compute_mape(y_true, y_pred, epsilon=1e-8):
    """Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def evaluate_model(model, test_loader, scaler=None):
    """
    Run full evaluation on test set.
    Returns dict of metrics: rmse, mae, mape, predictions, targets.
    """
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            all_preds.extend(preds.numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

    # Inverse-transform if scaler provided (get real kWh values)
    if scaler is not None:
        all_preds   = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
        all_targets = scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

    return {
        "rmse":        compute_rmse(all_targets, all_preds),
        "mae":         compute_mae(all_targets, all_preds),
        "mape":        compute_mape(all_targets, all_preds),
        "predictions": all_preds,
        "targets":     all_targets,
    }


# ── System resource metrics ───────────────────────────────────────────────────
def get_memory_usage_mb():
    """Return current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def measure_training_time(fn, *args, **kwargs):
    """Wrap any function and return (result, elapsed_seconds)."""
    start  = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start


def estimate_communication_bytes(parameters, compressed=False, k=0.1):
    """
    Estimate bytes transferred per round.
    Full precision: 4 bytes per float32 parameter.
    Compressed (Top-K): only k fraction of parameters sent.
    """
    total_params = sum(p.size for p in parameters)
    if compressed:
        total_params = int(total_params * k)
    return total_params * 4   # float32 = 4 bytes


# ── Results saving and plotting ───────────────────────────────────────────────
def save_results(results_dict, output_dir, experiment_name):
    """Save experiment results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = os.path.join(output_dir, f"{timestamp}_{experiment_name}.json")

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)):     return int(obj)
        return obj

    serializable = {k: convert(v) for k, v in results_dict.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Results saved to: {path}")
    return path


def plot_predictions(targets, predictions, title="Energy Forecasting", save_path=None):
    """Plot actual vs predicted energy consumption."""
    plt.figure(figsize=(14, 5))
    plt.plot(targets[:200],     label="Actual",    color="black",  linewidth=1.5)
    plt.plot(predictions[:200], label="Predicted", color="grey",   linewidth=1.5, linestyle="--")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Time (hours)", fontsize=12)
    plt.ylabel("Global Active Power (kW)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    plt.show()


def print_comparison_table(baseline, optimized):
    """
    Print a side-by-side comparison table of baseline vs optimized results.
    Used for Chapter 4 results section.
    """
    print("\n" + "="*70)
    print(f"{'METRIC':<30} {'BASELINE':>15} {'OPTIMIZED':>15} {'CHANGE':>10}")
    print("="*70)

    metrics = ["rmse", "mae", "memory_mb", "comm_bytes", "train_time_s"]
    labels  = ["RMSE", "MAE", "Memory (MB)", "Comm. Bytes", "Train Time (s)"]

    for metric, label in zip(metrics, labels):
        b = baseline.get(metric, 0)
        o = optimized.get(metric, 0)
        if b > 0:
            change = f"{((o - b) / b) * 100:+.1f}%"
        else:
            change = "N/A"
        print(f"  {label:<28} {b:>15.4f} {o:>15.4f} {change:>10}")

    print("="*70 + "\n")
