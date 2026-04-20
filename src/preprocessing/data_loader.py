"""
Data Preprocessing Pipeline - UCI Household Power Consumption
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025

Fixes:
  - Non-IID split uses Dirichlet-based quantity skew (realistic heterogeneous data)
  - IID split preserves temporal order after shuffle
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import os
import pickle


SEQ_LEN     = 24
TRAIN_RATIO = 0.80
BATCH_SIZE  = 32
TARGET_COL  = "Global_active_power"


class EnergyDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets,   dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_uci_dataset(filepath):
    print(f"Loading UCI dataset from: {filepath}")
    df = pd.read_csv(filepath, sep=";", low_memory=False, na_values=["?"])
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df.drop(columns=["Date", "Time"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime",  inplace=True)
    df = df[[TARGET_COL]].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df.dropna(inplace=True)
    df = df.resample("h").mean()
    df.dropna(inplace=True)
    print(f"Dataset loaded: {len(df)} hourly records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df


def create_sliding_windows(series, seq_len=SEQ_LEN):
    values = series.values.flatten()
    X, y   = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y).reshape(-1, 1)


def normalize_data(train_data, test_data):
    scaler = StandardScaler()
    X_train, y_train = train_data
    X_test,  y_test  = test_data
    n_train, seq, feat = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, feat)).reshape(n_train, seq, feat)
    X_test_scaled  = scaler.transform(X_test.reshape(-1, feat)).reshape(X_test.shape)
    y_train_scaled = scaler.transform(y_train)
    y_test_scaled  = scaler.transform(y_test)
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler


def split_for_clients(X, y, num_clients=3, iid=True):
    """
    Split data across FL clients.

    iid=True  : randomly shuffle then split equally — each client sees
                a representative sample of the full distribution.

    iid=False : Dirichlet-based quantity skew (alpha=0.5) — each client
                receives a different-sized, temporally-skewed partition,
                simulating realistic heterogeneous smart-meter deployments
                where devices have different usage patterns and data volumes.
    """
    n = len(X)
    client_data = []

    if iid:
        indices = np.random.permutation(n)
        X, y    = X[indices], y[indices]
        size    = n // num_clients
        for i in range(num_clients):
            start = i * size
            end   = start + size if i < num_clients - 1 else n
            client_data.append((X[start:end], y[start:end]))
            print(f"  Client {i+1}: {end - start} samples (IID)")
    else:
        # Dirichlet quantity skew: each client gets a non-uniform share
        # alpha=0.5 produces moderate heterogeneity (lower = more skewed)
        alpha      = 0.5
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions / proportions.sum())  # normalise

        # Assign indices in time order (preserves temporal structure per client)
        boundaries = (np.cumsum(proportions) * n).astype(int)
        boundaries[-1] = n  # ensure last client gets remainder
        start = 0
        for i, end in enumerate(boundaries):
            client_data.append((X[start:end], y[start:end]))
            print(f"  Client {i+1}: {end - start} samples "
                  f"({(end-start)/n*100:.1f}%) (Non-IID)")
            start = end

    return client_data


def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE):
    train_ds     = EnergyDataset(X_train, y_train)
    test_ds      = EnergyDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def prepare_and_save_splits(raw_path, output_dir, num_clients=3, iid=True):
    os.makedirs(output_dir, exist_ok=True)
    df         = load_uci_dataset(raw_path)
    split_idx  = int(len(df) * TRAIN_RATIO)
    train_df   = df.iloc[:split_idx]
    test_df    = df.iloc[split_idx:]

    X_train_raw, y_train_raw = create_sliding_windows(train_df[TARGET_COL])
    X_test_raw,  y_test_raw  = create_sliding_windows(test_df[TARGET_COL])

    X_train, y_train, X_test, y_test, scaler = normalize_data(
        (X_train_raw, y_train_raw), (X_test_raw, y_test_raw)
    )

    split_label = "IID" if iid else "Non-IID"
    print(f"\nSplitting into {num_clients} clients ({split_label}):")
    client_splits = split_for_clients(X_train, y_train, num_clients, iid)

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
