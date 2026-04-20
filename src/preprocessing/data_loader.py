"""
Data Preprocessing Pipeline - UCI Household Power Consumption
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import os
import pickle


# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN      = 24     # 24-hour sliding window
TRAIN_RATIO  = 0.80   # 80% train, 20% test
BATCH_SIZE   = 32
TARGET_COL   = "Global_active_power"


class EnergyDataset(Dataset):
    """
    PyTorch Dataset for sliding-window energy time series.
    Each sample: (seq_len, 1) input -> (1,) target (next hour value)
    """

    def __init__(self, sequences, targets):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets,   dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_uci_dataset(filepath):
    """
    Load and clean the raw UCI Household Power Consumption CSV.
    Returns a cleaned hourly DataFrame.
    """
    print(f"Loading UCI dataset from: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=";",
        low_memory=False,
        na_values=["?"]
    )

    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df.drop(columns=["Date", "Time"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime",  inplace=True)

    # Keep only the target column
    df = df[[TARGET_COL]].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Drop missing values
    df.dropna(inplace=True)

    # Resample from 1-minute to hourly (mean)
    df = df.resample("h").mean()
    df.dropna(inplace=True)

    print(f"Dataset loaded: {len(df)} hourly records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df


def create_sliding_windows(series, seq_len=SEQ_LEN):
    """
    Convert a 1D time series into (X, y) sliding window pairs.
    X shape: (n_samples, seq_len, 1)
    y shape: (n_samples, 1)
    """
    values = series.values.flatten()
    X, y   = [], []

    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def normalize_data(train_data, test_data):
    """
    Fit StandardScaler on train data only, apply to both train and test.
    Returns (X_train, y_train, X_test, y_test, scaler)
    """
    scaler = StandardScaler()

    X_train, y_train = train_data
    X_test,  y_test  = test_data

    # Reshape for scaler (needs 2D), then reshape back
    n_train, seq, feat = X_train.shape
    X_train_2d = X_train.reshape(-1, feat)
    X_test_2d  = X_test.reshape(-1, feat)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, seq, feat)
    X_test_scaled  = scaler.transform(X_test_2d).reshape(X_test.shape)

    y_train_scaled = scaler.transform(y_train)
    y_test_scaled  = scaler.transform(y_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler


def split_for_clients(X, y, num_clients=5, iid=True):
    """
    Split data across FL clients.
    iid=True  : random shuffle then split equally (IID)
    iid=False : split by time order (Non-IID, each client has different period)
    """
    n = len(X)

    if iid:
        indices = np.random.permutation(n)
        X, y    = X[indices], y[indices]

    split_size  = n // num_clients
    client_data = []

    for i in range(num_clients):
        start = i * split_size
        end   = start + split_size if i < num_clients - 1 else n
        client_data.append((X[start:end], y[start:end]))
        print(f"  Client {i+1}: {end - start} samples")

    return client_data


def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE):
    """Return PyTorch DataLoaders for train and test splits."""
    train_ds     = EnergyDataset(X_train, y_train)
    test_ds      = EnergyDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def prepare_and_save_splits(raw_path, output_dir, num_clients=5, iid=True):
    """
    Full pipeline: load, clean, window, normalize, split, save.
    Run this once before starting FL training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and resample
    df = load_uci_dataset(raw_path)

    # 2. Train/test split (time-based, no shuffle)
    split_idx  = int(len(df) * TRAIN_RATIO)
    train_df   = df.iloc[:split_idx]
    test_df    = df.iloc[split_idx:]

    # 3. Create sliding windows
    X_train_raw, y_train_raw = create_sliding_windows(train_df[TARGET_COL])
    X_test_raw,  y_test_raw  = create_sliding_windows(test_df[TARGET_COL])

    # 4. Normalize
    X_train, y_train, X_test, y_test, scaler = normalize_data(
        (X_train_raw, y_train_raw),
        (X_test_raw,  y_test_raw)
    )

    # 5. Split into client partitions
    print(f"\nSplitting into {num_clients} clients (iid={iid}):")
    client_splits = split_for_clients(X_train, y_train, num_clients, iid)

    # 6. Save everything
    for i, (X_c, y_c) in enumerate(client_splits):
        np.save(os.path.join(output_dir, f"client_{i+1}_X.npy"), X_c)
        np.save(os.path.join(output_dir, f"client_{i+1}_y.npy"), y_c)

    np.save(os.path.join(output_dir, "test_X.npy"), X_test)
    np.save(os.path.join(output_dir, "test_y.npy"), y_test)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nAll splits saved to: {output_dir}")
    print(f"Test set: {len(X_test)} samples")
    return client_splits, X_test, y_test, scaler
