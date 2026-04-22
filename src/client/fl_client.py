"""
Federated Learning Client - Flower Framework
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025
"""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from src.models.lstm_model import LSTMModel, get_model_parameters, set_model_parameters
from src.preprocessing.data_loader import get_dataloaders


DEVICE_PROFILES = {
    "pi4":     {"local_epochs": 5,  "batch_size": 32},
    "pi_zero": {"local_epochs": 2,  "batch_size": 16},
    "esp32":   {"local_epochs": 1,  "batch_size": 8},
    "laptop":  {"local_epochs": 10, "batch_size": 64},
}

BASE_LR    = 0.001
LR_DECAY   = 0.5     # multiply LR by this factor
LR_STEP    = 10      # decay every N rounds
CLIP_NORM  = 1.0     # max gradient norm for clipping


def top_k_compression(gradients, k=0.1):
    """Keep top-k fraction of gradient values by magnitude; zero out the rest."""
    compressed = []
    for grad in gradients:
        flat      = grad.flatten()
        num_keep  = max(1, int(len(flat) * k))
        threshold = np.sort(np.abs(flat))[-num_keep]
        mask      = np.abs(flat) >= threshold
        compressed.append((flat * mask).reshape(grad.shape))
    return compressed


def count_compressed_bytes(params):
    return int(sum(np.count_nonzero(p) for p in params)) * 4


class EnergyFLClient(fl.client.NumPyClient):

    def __init__(self, client_id, X_train, y_train, X_test, y_test,
                 device_type="pi4", compression_k=0.1):
        self.client_id     = client_id
        self.device_type   = device_type
        self.compression_k = compression_k

        profile           = DEVICE_PROFILES.get(device_type, DEVICE_PROFILES["pi4"])
        self.local_epochs = profile["local_epochs"]
        self.batch_size   = profile["batch_size"]

        print(f"[Client {client_id}] Device: {device_type} | "
              f"Epochs: {self.local_epochs} | Batch: {self.batch_size} | "
              f"Compression k={compression_k}")

        self.model     = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=BASE_LR)

        self.train_loader, self.test_loader = get_dataloaders(
            X_train, y_train, X_test, y_test, self.batch_size
        )

    def _get_lr(self, current_round):
        return BASE_LR * (LR_DECAY ** (current_round // LR_STEP))

    def get_parameters(self, config):
        params = get_model_parameters(self.model)
        if self.compression_k < 1.0:
            params = top_k_compression(params, k=self.compression_k)
        return params

    def set_parameters(self, parameters):
        set_model_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        current_round = int(config.get("round", 1))
        lr = self._get_lr(current_round)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.model.train()
        total_loss = 0.0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss   = self.criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_NORM)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            total_loss    += avg_epoch_loss
            print(f"  [Client {self.client_id}] Round {current_round} | "
                  f"Epoch {epoch+1}/{self.local_epochs} | "
                  f"Loss: {avg_epoch_loss:.6f} | LR: {lr:.6f}")

        updated_params   = self.get_parameters(config={})
        compressed_bytes = count_compressed_bytes(updated_params)
        num_samples      = len(self.train_loader.dataset)

        return updated_params, num_samples, {
            "train_loss":       float(total_loss / self.local_epochs),
            "client_id":        self.client_id,
            "device":           self.device_type,
            "epochs":           self.local_epochs,
            "lr":               float(lr),
            "compressed_bytes": compressed_bytes,
        }

    def evaluate(self, parameters, config):
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
        preds    = np.array(all_preds)
        targets  = np.array(all_targets)
        rmse     = float(np.sqrt(np.mean((preds - targets) ** 2)))
        mae      = float(np.mean(np.abs(preds - targets)))

        return avg_loss, len(self.test_loader.dataset), {
            "rmse":      rmse,
            "mae":       mae,
            "client_id": self.client_id,
        }


def start_client(server_address, client_id, X_train, y_train, X_test, y_test,
                 device_type="pi4", compression_k=0.1):
    client = EnergyFLClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device_type=device_type,
        compression_k=compression_k,
    )
    fl.client.start_numpy_client(server_address=server_address, client=client)
