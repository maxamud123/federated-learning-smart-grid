"""
Centralized Baseline Training - Non-Federated Reference Model
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025

Trains a single LSTM on the POOLED training data of all federated clients
(i.e. the federated split is undone) and evaluates it on the same held-out
test set the clients use. Provides the centralized RMSE/MAE reference that
hypothesis H5 compares the federated system against.

Usage:
  python train_centralized.py
  python train_centralized.py --epochs 50 --batch_size 32
"""

import argparse
import glob
import json
import os
import pickle
import random
import re

import numpy as np
import torch
import torch.nn as nn

from src.models.lstm_model import LSTMModel
from src.preprocessing.data_loader import get_dataloaders
from src.client.fl_client import BASE_LR, LR_DECAY, LR_STEP, CLIP_NORM
from src.evaluation.metrics import get_memory_usage_mb

DATA_SPLITS = "data/splits"
RESULTS_DIR = "experiments/results"
OUTPUT_JSON = os.path.join(RESULTS_DIR, "centralized_baseline.json")


def set_seed(seed):
    """Identical seeding to main.py so the comparison is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pooled_data(splits_dir):
    """Undo the federated split: concatenate every client's training data."""
    paths = sorted(
        glob.glob(os.path.join(splits_dir, "client_*_X.npy")),
        key=lambda p: int(re.search(r"client_(\d+)_X", p).group(1)),
    )
    if not paths:
        raise FileNotFoundError(
            "No client splits found in {}. Run: python main.py --mode preprocess "
            "--data data/raw/household_power_consumption.txt".format(splits_dir)
        )

    X_parts, y_parts = [], []
    for x_path in paths:
        y_path = x_path.replace("_X.npy", "_y.npy")
        X_c, y_c = np.load(x_path), np.load(y_path)
        X_parts.append(X_c)
        y_parts.append(y_c)
        print("  {:22s} {:6d} samples".format(os.path.basename(x_path), X_c.shape[0]))

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    X_test  = np.load(os.path.join(splits_dir, "test_X.npy"))
    y_test  = np.load(os.path.join(splits_dir, "test_y.npy"))
    return X_train, y_train, X_test, y_test, len(paths)


def evaluate(model, test_loader, criterion):
    """Mirrors EnergyFLClient.evaluate() exactly, so the numbers are comparable."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output      = model(X_batch)
            total_loss += criterion(output, y_batch).item()
            all_preds.extend(output.numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    metrics = {
        "avg_loss": float(total_loss / len(test_loader)),
        "rmse":     float(np.sqrt(np.mean((preds - targets) ** 2))),
        "mae":      float(np.mean(np.abs(preds - targets))),
    }
    return metrics, preds, targets


def denormalized_metrics(preds, targets, splits_dir):
    """Secondary metrics in real units (kW) via the saved StandardScaler."""
    scaler_path = os.path.join(splits_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    p = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    t = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    return {
        "rmse_kw": float(np.sqrt(np.mean((p - t) ** 2))),
        "mae_kw":  float(np.mean(np.abs(p - t))),
    }


def convert(obj):
    """Same coercion save_results() in metrics.py applies before json.dump."""
    if isinstance(obj, np.ndarray):               return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)):     return int(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Centralized (non-federated) LSTM baseline")
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Training epochs (default 50, mirroring the 50 FL rounds)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default 32, the pi4 client profile)")
    parser.add_argument("--lr",         type=float, default=BASE_LR)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--splits",     default=DATA_SPLITS)
    parser.add_argument("--output",     default=OUTPUT_JSON)
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 60)
    print("CENTRALIZED BASELINE - pooled training (H5 reference)")
    print("=" * 60)
    print("[Seed] {} | Epochs: {} | Batch: {} | LR: {}".format(
        args.seed, args.epochs, args.batch_size, args.lr))

    print("\nPooling client training data:")
    X_train, y_train, X_test, y_test, num_clients = load_pooled_data(args.splits)
    print("  {:22s} {:6d} samples".format("POOLED TOTAL", X_train.shape[0]))
    print("  {:22s} {:6d} samples".format("test set", X_test.shape[0]))

    train_loader, test_loader = get_dataloaders(
        X_train, y_train, X_test, y_test, args.batch_size
    )

    model     = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mem_before = get_memory_usage_mb()
    print("\nRAM before training: {:.1f} MB\n".format(mem_before))

    for epoch in range(args.epochs):
        # Same schedule as the FL clients, with epoch standing in for round.
        lr = args.lr * (LR_DECAY ** (epoch // LR_STEP))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            epoch_loss += loss.item()

        print("  [Centralized] Epoch {}/{} | Loss: {:.6f} | LR: {:.6f}".format(
            epoch + 1, args.epochs, epoch_loss / len(train_loader), lr), flush=True)

    mem_after = get_memory_usage_mb()
    metrics, preds, targets = evaluate(model, test_loader, criterion)
    real_units = denormalized_metrics(preds, targets, args.splits)

    results = {
        "experiment":         "centralized_baseline",
        "mode":               "centralized",
        "num_clients_pooled": num_clients,
        "train_samples":      int(X_train.shape[0]),
        "test_samples":       int(X_test.shape[0]),
        "epochs":             args.epochs,
        "batch_size":         args.batch_size,
        "base_lr":            args.lr,
        "lr_decay":           LR_DECAY,
        "lr_step":            LR_STEP,
        "clip_norm":          CLIP_NORM,
        "seed":               args.seed,
        "avg_loss":           round(metrics["avg_loss"], 6),
        "rmse":               round(metrics["rmse"], 6),
        "mae":                round(metrics["mae"], 6),
        "memory_before_mb":   round(mem_before, 2),
        "memory_after_mb":    round(mem_after, 2),
        "metric_space":       "normalized (StandardScaler) - matches federated client evaluate()",
    }
    if real_units:
        results["rmse_kw"] = round(real_units["rmse_kw"], 6)
        results["mae_kw"]  = round(real_units["mae_kw"], 6)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({k: convert(v) for k, v in results.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("CENTRALIZED BASELINE RESULTS")
    print("=" * 60)
    print("  RMSE      : {:.6f}  (normalized)".format(results["rmse"]))
    print("  MAE       : {:.6f}  (normalized)".format(results["mae"]))
    if real_units:
        print("  RMSE (kW) : {:.6f}".format(results["rmse_kw"]))
        print("  MAE  (kW) : {:.6f}".format(results["mae_kw"]))
    print("  Avg loss  : {:.6f}".format(results["avg_loss"]))
    print("\nResults saved to: {}".format(args.output))
    print("=" * 60)


if __name__ == "__main__":
    main()
