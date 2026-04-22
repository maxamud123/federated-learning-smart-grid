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
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.lstm_model import get_model_size_mb


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def compute_mape(y_true, y_pred, epsilon=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def evaluate_model(model, test_loader, scaler=None):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            all_preds.extend(preds.numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

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


def get_memory_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def measure_training_time(fn, *args, **kwargs):
    start  = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start


def estimate_communication_bytes(parameters, compressed=False, k=0.1):
    total_params = sum(p.size for p in parameters)
    if compressed:
        total_params = int(total_params * k)
    return total_params * 4


def save_results(results_dict, output_dir, experiment_name):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path      = os.path.join(output_dir, f"{timestamp}_{experiment_name}.json")

    def convert(obj):
        if isinstance(obj, np.ndarray):                  return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):    return float(obj)
        if isinstance(obj, (np.int32, np.int64)):        return int(obj)
        return obj

    with open(path, "w") as f:
        json.dump({k: convert(v) for k, v in results_dict.items()}, f, indent=2)

    print(f"Results saved to: {path}")
    return path


def plot_predictions(targets, predictions, title="Energy Forecasting", save_path=None):
    plt.figure(figsize=(14, 5))
    plt.plot(targets[:200],     label="Actual",    color="black", linewidth=1.5)
    plt.plot(predictions[:200], label="Predicted", color="grey",  linewidth=1.5, linestyle="--")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Time (hours)", fontsize=12)
    plt.ylabel("Global Active Power (kW)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    plt.show()


def _parse_log(log_path):
    """Parse a .jsonl log file into separate fit and evaluate record lists."""
    fit_rounds  = []
    eval_rounds = []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("phase") == "fit":
                fit_rounds.append(entry)
            elif entry.get("phase") == "evaluate":
                eval_rounds.append(entry)
    return fit_rounds, eval_rounds


def print_comparison_table(baseline, optimized):
    """Side-by-side comparison of baseline vs optimized result dicts."""
    print("\n" + "="*70)
    print(f"{'METRIC':<30} {'BASELINE':>15} {'OPTIMIZED':>15} {'CHANGE':>10}")
    print("="*70)
    metrics = ["rmse", "mae", "memory_mb", "comm_bytes", "train_time_s"]
    labels  = ["RMSE", "MAE", "Memory (MB)", "Comm. Bytes", "Train Time (s)"]
    for metric, label in zip(metrics, labels):
        b = baseline.get(metric, 0)
        o = optimized.get(metric, 0)
        change = f"{((o - b) / b) * 100:+.1f}%" if b > 0 else "N/A"
        print(f"  {label:<28} {b:>15.4f} {o:>15.4f} {change:>10}")
    print("="*70 + "\n")


def compare_logs(baseline_log_path, optimized_log_path):
    """
    Automated comparison of two experiment log files.
    Prints a summary table and plots convergence curves side by side.
    """
    b_fit,  b_eval  = _parse_log(baseline_log_path)
    o_fit,  o_eval  = _parse_log(optimized_log_path)

    # ── Summary stats ──────────────────────────────────────────────────────────
    b_final_loss = b_eval[-1]["avg_loss"] if b_eval else float("nan")
    o_final_loss = o_eval[-1]["avg_loss"] if o_eval else float("nan")

    def _best_loss_and_round(eval_rows):
        if not eval_rows:
            return float("nan"), "N/A"
        valid_rows = [e for e in eval_rows if "avg_loss" in e]
        if not valid_rows:
            return float("nan"), "N/A"
        best_row = min(valid_rows, key=lambda e: e["avg_loss"])
        return best_row["avg_loss"], best_row.get("round", "N/A")

    b_best_loss, b_best_round = _best_loss_and_round(b_eval)
    o_best_loss, o_best_round = _best_loss_and_round(o_eval)

    b_final_rmse = b_eval[-1].get("rmse", float("nan")) if b_eval else float("nan")
    o_final_rmse = o_eval[-1].get("rmse", float("nan")) if o_eval else float("nan")
    b_final_mae  = b_eval[-1].get("mae",  float("nan")) if b_eval else float("nan")
    o_final_mae  = o_eval[-1].get("mae",  float("nan")) if o_eval else float("nan")

    # Communication bytes (average per round)
    b_avg_actual     = np.mean([r["actual_bytes"]     for r in b_fit if "actual_bytes"     in r]) if b_fit else 0
    o_avg_actual     = np.mean([r["actual_bytes"]     for r in o_fit if "actual_bytes"     in r]) if o_fit else 0
    o_avg_compressed = np.mean([r["compressed_bytes"] for r in o_fit if "compressed_bytes" in r]) if o_fit else 0
    o_avg_savings    = np.mean([r["savings_pct"]      for r in o_fit if "savings_pct"      in r]) if o_fit else 0

    print("\n" + "="*70)
    print(f"{'METRIC':<35} {'BASELINE':>15} {'OPTIMIZED':>15}")
    print("="*70)
    rows = [
        ("Final loss",              f"{b_final_loss:.6f}",  f"{o_final_loss:.6f}"),
        ("Best loss",               f"{b_best_loss:.6f}",   f"{o_best_loss:.6f}"),
        ("Best round",              f"{b_best_round}",      f"{o_best_round}"),
        ("Final RMSE",              f"{b_final_rmse:.6f}",  f"{o_final_rmse:.6f}"),
        ("Final MAE",               f"{b_final_mae:.6f}",   f"{o_final_mae:.6f}"),
        ("Avg actual bytes/round",  f"{b_avg_actual:.0f}",  f"{o_avg_actual:.0f}"),
        ("Avg compressed bytes",    "—",                    f"{o_avg_compressed:.0f}"),
        ("Avg comm savings",        "—",                    f"{o_avg_savings:.1f}%"),
        ("Total rounds logged",     f"{len(b_eval)}",       f"{len(o_eval)}"),
    ]
    for label, b_val, o_val in rows:
        print(f"  {label:<33} {b_val:>15} {o_val:>15}")
    print("="*70)

    # ── Convergence plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss convergence
    ax = axes[0]
    if b_eval:
        ax.plot([e["round"] for e in b_eval],
                [e["avg_loss"] for e in b_eval],
                label="Baseline", color="steelblue", linewidth=1.5)
    if o_eval:
        ax.plot([e["round"] for e in o_eval],
                [e["avg_loss"] for e in o_eval],
                label="Optimized", color="darkorange", linewidth=1.5)
    ax.set_title("Loss Convergence", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg Loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE per round
    ax = axes[1]
    b_rmse_rounds = [e for e in b_eval if "rmse" in e]
    o_rmse_rounds = [e for e in o_eval if "rmse" in e]
    if b_rmse_rounds:
        ax.plot([e["round"] for e in b_rmse_rounds],
                [e["rmse"]  for e in b_rmse_rounds],
                label="Baseline RMSE", color="steelblue", linewidth=1.5)
    if o_rmse_rounds:
        ax.plot([e["round"] for e in o_rmse_rounds],
                [e["rmse"]  for e in o_rmse_rounds],
                label="Optimized RMSE", color="darkorange", linewidth=1.5)
    ax.set_title("RMSE per Round", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join("experiments", "results", "comparison_plot.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nComparison plot saved to: {out_path}")
    plt.show()
