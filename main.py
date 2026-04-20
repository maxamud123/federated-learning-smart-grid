"""
Main Experiment Runner - End-to-End Federated Learning
Thesis: Optimizing Federated Learning for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025

Usage:
  # Step 1: Preprocess (IID or Non-IID)
  python main.py --mode preprocess --data data/raw/household_power_consumption.txt
  python main.py --mode preprocess --data data/raw/household_power_consumption.txt --non_iid

  # Step 2: Local pipeline test
  python main.py --mode test

  # Step 3: Run FL experiment — localhost simulation (4 terminals)
  python main.py --mode server --experiment baseline --num_rounds 50
  python main.py --mode client --client_id 1 --device pi4
  python main.py --mode client --client_id 2 --device pi_zero
  python main.py --mode client --client_id 3 --device esp32

  # Step 4: Run optimized experiment
  python main.py --mode server --experiment optimized --num_rounds 50
  python main.py --mode client --client_id 1 --device pi4    --compress
  python main.py --mode client --client_id 2 --device pi_zero --compress
  python main.py --mode client --client_id 3 --device esp32   --compress

  # Step 5: Compare results
  python main.py --mode compare \
    --baseline_log experiments/logs/<timestamp>_baseline.jsonl \
    --optimized_log experiments/logs/<timestamp>_optimized.jsonl

  # Real hardware over WiFi: add --server_ip 192.168.1.10 to all client commands
"""

import argparse
import random
import numpy as np
import os
import torch

from src.preprocessing.data_loader import prepare_and_save_splits, get_dataloaders
from src.server.fl_server import start_server
from src.client.fl_client import start_client
from src.models.lstm_model import LSTMModel, apply_int8_quantization, get_model_size_mb
from src.evaluation.metrics import (
    evaluate_model, get_memory_usage_mb, save_results,
    print_comparison_table, compare_logs,
)


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
DATA_SPLITS = "data/splits"
RESULTS_DIR = "experiments/results"
NUM_ROUNDS  = 50
NUM_CLIENTS = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_preprocess(raw_data_path, num_clients=NUM_CLIENTS, iid=True):
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    prepare_and_save_splits(
        raw_path=raw_data_path,
        output_dir=DATA_SPLITS,
        num_clients=num_clients,
        iid=iid,
    )
    print("Preprocessing complete.")


def run_server(experiment_name="baseline", num_rounds=NUM_ROUNDS,
               min_clients=NUM_CLIENTS, host=SERVER_HOST, port=SERVER_PORT,
               patience=10):
    server_address = f"{host}:{port}"
    print("\n" + "="*60)
    print(f"FL SERVER — Experiment: {experiment_name}")
    print(f"Listening on {server_address} | Patience: {patience}")
    print("="*60)
    start_server(
        num_rounds=num_rounds,
        min_clients=min_clients,
        server_address=server_address,
        experiment_name=experiment_name,
        patience=patience,
    )


def run_client(client_id, device_type, compress=False,
               server_ip=SERVER_HOST, server_port=SERVER_PORT):
    server_address = f"{server_ip}:{server_port}"
    print("\n" + "="*60)
    print(f"FL CLIENT {client_id} — Device: {device_type}")
    print(f"Connecting to {server_address}")
    print("="*60)

    X_train = np.load(os.path.join(DATA_SPLITS, f"client_{client_id}_X.npy"))
    y_train = np.load(os.path.join(DATA_SPLITS, f"client_{client_id}_y.npy"))
    X_test  = np.load(os.path.join(DATA_SPLITS, "test_X.npy"))
    y_test  = np.load(os.path.join(DATA_SPLITS, "test_y.npy"))

    start_client(
        server_address=server_address,
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device_type=device_type,
        compression_k=0.1 if compress else 1.0,
    )


def run_local_test():
    print("\n" + "="*60)
    print("LOCAL PIPELINE TEST")
    print("="*60)

    if not os.path.exists(os.path.join(DATA_SPLITS, "client_1_X.npy")):
        print("ERROR: Data splits not found. Run --mode preprocess first.")
        return

    X_train = np.load(os.path.join(DATA_SPLITS, "client_1_X.npy"))
    y_train = np.load(os.path.join(DATA_SPLITS, "client_1_y.npy"))
    X_test  = np.load(os.path.join(DATA_SPLITS, "test_X.npy"))
    y_test  = np.load(os.path.join(DATA_SPLITS, "test_y.npy"))

    from src.preprocessing.data_loader import get_dataloaders
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/5 - Loss: {total_loss/len(train_loader):.6f}")

    print(f"RAM after training: {get_memory_usage_mb():.1f} MB")
    results_fp32 = evaluate_model(model, test_loader)
    print(f"\nFP32  -> RMSE: {results_fp32['rmse']:.6f} | MAE: {results_fp32['mae']:.6f}")

    print("\nApplying INT8 quantization...")
    model_q = apply_int8_quantization(model, calibration_loader=train_loader)
    print(f"Model size (INT8): {get_model_size_mb(model_q):.2f} MB")
    print(f"RAM after quantization: {get_memory_usage_mb():.1f} MB")
    print("\nLocal pipeline test complete. FL system is ready.")


def run_compare(baseline_log, optimized_log):
    print("\n" + "="*60)
    print("EXPERIMENT COMPARISON")
    print("="*60)
    compare_logs(baseline_log, optimized_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Learning for Smart Grid Energy Forecasting"
    )
    parser.add_argument("--mode", required=True,
                        choices=["preprocess", "server", "client", "test", "compare"])
    parser.add_argument("--data",          default="data/raw/household_power_consumption.txt")
    parser.add_argument("--client_id",     type=int,   default=1)
    parser.add_argument("--device",        default="pi4",
                        choices=["pi4", "pi_zero", "esp32", "laptop"])
    parser.add_argument("--experiment",    default="baseline")
    parser.add_argument("--compress",      action="store_true",
                        help="Enable Top-K gradient compression (k=0.1)")
    parser.add_argument("--non_iid",       action="store_true",
                        help="Use Non-IID Dirichlet data split instead of IID")
    parser.add_argument("--num_rounds",    type=int,   default=NUM_ROUNDS)
    parser.add_argument("--num_clients",   type=int,   default=NUM_CLIENTS)
    parser.add_argument("--patience",      type=int,   default=10,
                        help="Early stopping patience (rounds without improvement)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--server_ip",     default=SERVER_HOST,
                        help="Server IP (e.g. 192.168.1.10 for real hardware)")
    parser.add_argument("--server_port",   type=int,   default=SERVER_PORT)
    parser.add_argument("--baseline_log",  default=None,
                        help="Path to baseline .jsonl log (for --mode compare)")
    parser.add_argument("--optimized_log", default=None,
                        help="Path to optimized .jsonl log (for --mode compare)")

    args = parser.parse_args()

    set_seed(args.seed)
    print(f"[Seed] {args.seed} | Non-IID: {args.non_iid}")

    if args.mode == "preprocess":
        run_preprocess(args.data, args.num_clients, iid=not args.non_iid)

    elif args.mode == "server":
        run_server(args.experiment, args.num_rounds, args.num_clients,
                   host=args.server_ip, port=args.server_port,
                   patience=args.patience)

    elif args.mode == "client":
        run_client(args.client_id, args.device, args.compress,
                   server_ip=args.server_ip, server_port=args.server_port)

    elif args.mode == "test":
        run_local_test()

    elif args.mode == "compare":
        if not args.baseline_log or not args.optimized_log:
            print("ERROR: --mode compare requires --baseline_log and --optimized_log")
        else:
            run_compare(args.baseline_log, args.optimized_log)
