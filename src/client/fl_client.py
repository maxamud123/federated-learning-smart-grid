"""
Federated Learning Client - Flower Framework
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025

Optimizations applied here:
  1. Adaptive local training  - epochs adjusted by device capability
  2. Top-K gradient compression - only top 10% of gradients sent to server
"""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from src.models.lstm_model import LSTMModel, get_model_parameters, set_model_parameters
from src.preprocessing.data_loader import EnergyDataset, get_dataloaders, BATCH_SIZE


# ── Device capability profiles (adaptive training) ────────────────────────────
DEVICE_PROFILES = {
    "pi4":    {"local_epochs": 5, "batch_size": 32},   # Raspberry Pi 4 (2GB RAM)
    "pi_zero":{"local_epochs": 2, "batch_size": 16},   # Raspberry Pi Zero (512MB)
    "esp32":  {"local_epochs": 1, "batch_size": 8},    # ESP32 (520KB RAM)
    "laptop": {"local_epochs": 10,"batch_size": 64},   # Development laptop
}


def top_k_compression(gradients, k=0.1):
    """
    Top-K Gradient Compression (k=0.1 means top 10%).
    Keeps only the largest-magnitude gradients; zeroes the rest.
    Reduces communication overhead by ~60-90%.
    Returns compressed gradients (same shape, sparse values).
    """
    compressed = []
    for grad in gradients:
        flat        = grad.flatten()
        num_keep    = max(1, int(len(flat) * k))
        threshold   = np.sort(np.abs(flat))[-num_keep]
        mask        = np.abs(flat) >= threshold
        sparse_grad = flat * mask
        compressed.append(sparse_grad.reshape(grad.shape))
    return compressed


class EnergyFLClient(fl.client.NumPyClient):
    """
    Flower NumPy client for energy forecasting.
    Each physical device (Pi4, Pi Zero, ESP32) runs one instance.
    """

    def __init__(self, client_id, X_train, y_train, X_test, y_test,
                 device_type="pi4", compression_k=0.1):
        self.client_id     = client_id
        self.device_type   = device_type
        self.compression_k = compression_k
        self.device        = torch.device("cpu")   # always CPU for edge devices

        # Adaptive training parameters based on device type
        profile            = DEVICE_PROFILES.get(device_type, DEVICE_PROFILES["pi4"])
        self.local_epochs  = profile["local_epochs"]
        self.batch_size    = profile["batch_size"]

        print(f"[Client {client_id}] Device: {device_type} | "
              f"Epochs: {self.local_epochs} | Batch: {self.batch_size}")

        # Build model
        self.model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)

        # DataLoaders
        self.train_loader, self.test_loader = get_dataloaders(
            X_train, y_train, X_test, y_test, self.batch_size
        )

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    # ── Flower interface ───────────────────────────────────────────────────────

    def get_parameters(self, config):
        """Return current model parameters (compressed for upload)."""
        params = get_model_parameters(self.model)
        if self.compression_k < 1.0:
            params = top_k_compression(params, k=self.compression_k)
        return params

    def set_parameters(self, parameters):
        """Load global model parameters from server."""
        set_model_parameters(self.model, parameters)

    def fit(self, parameters, config):
        """
        Receive global model, train locally, return updated parameters.
        Called every federated round.
        """
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0.0
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss   = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss    = epoch_loss / len(self.train_loader)
            total_loss += avg_loss
            print(f"  [Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs} "
                  f"- Loss: {avg_loss:.6f}")

        # Return compressed parameters
        updated_params = self.get_parameters(config={})
        num_samples    = len(self.train_loader.dataset)

        return updated_params, num_samples, {
            "train_loss": total_loss / self.local_epochs,
            "client_id":  self.client_id,
            "device":     self.device_type,
            "epochs":     self.local_epochs,
        }

    def evaluate(self, parameters, config):
        """
        Evaluate global model on local test data.
        Returns loss and metrics to the server.
        """
        self.set_parameters(parameters)
        self.model.eval()

        total_loss  = 0.0
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                output      = self.model(X_batch)
                loss        = self.criterion(output, y_batch)
                total_loss += loss.item()
                all_preds.extend(output.numpy().flatten())
                all_targets.extend(y_batch.numpy().flatten())

        avg_loss = total_loss / len(self.test_loader)
        rmse     = float(np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)))
        mae      = float(np.mean(np.abs(np.array(all_preds) - np.array(all_targets))))

        return avg_loss, len(self.test_loader.dataset), {
            "rmse":      rmse,
            "mae":       mae,
            "client_id": self.client_id,
        }


def start_client(server_address, client_id, X_train, y_train, X_test, y_test,
                 device_type="pi4", compression_k=0.1):
    """Launch a Flower client and connect to the FL server."""
    client = EnergyFLClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device_type=device_type,
        compression_k=compression_k
    )
    fl.client.start_numpy_client(server_address=server_address, client=client)
