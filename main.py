"""
Main Experiment Runner - End-to-End Federated Learning
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025

Usage:
  # Step 1: Preprocess data (run once)
  python main.py --mode preprocess --data data/raw/household_power_consumption.txt

  # Step 2: Run baseline (unoptimized FedAvg)
  python main.py --mode server --experiment baseline
  python main.py --mode client --client_id 1 --device pi4
  python main.py --mode client --client_id 2 --device pi_zero
  python main.py --mode client --client_id 3 --device esp32

  # Step 3: Run optimized system
  python main.py --mode server --experiment optimized
  python main.py --mode client --client_id 1 --device pi4    --compress
  python main.py --mode client --client_id 2 --device pi_zero --compress
  python main.py --mode client --client_id 3 --device esp32   --compress
"""

import argparse
import numpy as np
import os
import torch

from src.preprocessing.data_loader import prepare_and_save_splits, get_dataloaders
from src.server.fl_server import start_server
from src.client.fl_client import start_client
from src.models.lstm_model import LSTMModel, apply_int8_quantization, get_model_size_mb
from src.evaluation.metrics import evaluate_model, get_memory_usage_mb, save_results, print_comparison_table


# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_ADDRESS = "127.0.0.1:8080"
DATA_SPLITS    = "data/splits"
RESULTS_DIR    = "experiments/results"
NUM_ROUNDS     = 50
NUM_CLIENTS    = 3


def run_preprocess(raw_data_path, num_clients=NUM_CLIENTS, iid=True):
    """Prepare and save all client data splits."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    prepare_and_save_splits(
        raw_path=raw_data_path,
        output_dir=DATA_SPLITS,
        num_clients=num_clients,
        iid=iid
    )
    print("Preprocessing complete.")


def run_server(experiment_name="baseline", num_rounds=NUM_ROUNDS, min_clients=NUM_CLIENTS):
    """Start the FL aggregation server."""
    print("\n" + "="*60)
    print(f"FL SERVER STARTING - Experiment: {experiment_name}")
    print("="*60)
    start_server(
        num_rounds=num_rounds,
        min_clients=min_clients,
        server_address=SERVER_ADDRESS,
        experiment_name=experiment_name
    )


def run_client(client_id, device_type, compress=True):
    """Start a single FL client."""
    print("\n" + "="*60)
    print(f"FL CLIENT {client_id} STARTING - Device: {device_type}")
    print("="*60)

    # Load this client's data
    X_train = np.load(os.path.join(DATA_SPLITS, f"client_{client_id}_X.npy"))
    y_train = np.load(os.path.join(DATA_SPLITS, f"client_{client_id}_y.npy"))
    X_test  = np.load(os.path.join(DATA_SPLITS, "test_X.npy"))
    y_test  = np.load(os.path.join(DATA_SPLITS, "test_y.npy"))

    compression_k = 0.1 if compress else 1.0

    start_client(
        server_address=SERVER_ADDRESS,
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device_type=device_type,
        compression_k=compression_k
    )


def run_local_test():
    """
    Quick local test: train one model locally (no FL) to verify pipeline works.
    Useful for debugging before running full FL experiment.
    """
    print("\n" + "="*60)
    print("LOCAL PIPELINE TEST")
    print("="*60)

    # Check data splits exist
    if not os.path.exists(os.path.join(DATA_SPLITS, "client_1_X.npy")):
        print("ERROR: Data splits not found. Run --mode preprocess first.")
        return

    X_train = np.load(os.path.join(DATA_SPLITS, "client_1_X.npy"))
    y_train = np.load(os.path.join(DATA_SPLITS, "client_1_y.npy"))
    X_test  = np.load(os.path.join(DATA_SPLITS, "test_X.npy"))
    y_test  = np.load(os.path.join(DATA_SPLITS, "test_y.npy"))

    from src.preprocessing.data_loader import EnergyDataset, get_dataloaders
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)

    # Build and train model locally (5 epochs)
    model     = LSTMModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Model size (FP32): {get_model_size_mb(model):.2f} MB")
    print(f"RAM before training: {get_memory_usage_mb():.1f} MB")

    model.train()
    for epoch in range(5):
        total_loss = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/5 - Loss: {total_loss/len(train_loader):.6f}")

    print(f"RAM after training: {get_memory_usage_mb():.1f} MB")

    # Evaluate
    results_fp32 = evaluate_model(model, test_loader)
    print(f"\nFP32 Results -> RMSE: {results_fp32['rmse']:.6f} | MAE: {results_fp32['mae']:.6f}")

    # Apply INT8 quantization
    print("\nApplying INT8 quantization...")
    model_q = apply_int8_quantization(model, calibration_loader=train_loader)
    print(f"Model size (INT8): {get_model_size_mb(model_q):.2f} MB")
    print(f"RAM after quantization: {get_memory_usage_mb():.1f} MB")

    print("\nLocal pipeline test complete. FL system is ready.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Learning for Smart Grid Energy Forecasting"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess", "server", "client", "test"],
                        help="Run mode")
    parser.add_argument("--data",        type=str, default="data/raw/household_power_consumption.txt")
    parser.add_argument("--client_id",   type=int, default=1)
    parser.add_argument("--device",      type=str, default="pi4",
                        choices=["pi4", "pi_zero", "esp32", "laptop"])
    parser.add_argument("--experiment",  type=str, default="baseline")
    parser.add_argument("--compress",    action="store_true",
                        help="Enable Top-K gradient compression (k=0.1)")
    parser.add_argument("--num_rounds",  type=int, default=NUM_ROUNDS)
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--iid",         action="store_true", default=True)

    args = parser.parse_args()

    if args.mode == "preprocess":
        run_preprocess(args.data, args.num_clients, args.iid)

    elif args.mode == "server":
        run_server(args.experiment, args.num_rounds, args.num_clients)

    elif args.mode == "client":
        run_client(args.client_id, args.device, args.compress)

    elif args.mode == "test":
        run_local_test()
