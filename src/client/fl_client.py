"""
Federated Learning Client - Flower Framework
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025
"""

import copy
import io
import time

import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from src.models.lstm_model import (
    LSTMModel, get_model_parameters, set_model_parameters, apply_int8_quantization,
)
from src.preprocessing.data_loader import get_dataloaders
from src.evaluation.metrics import get_memory_usage_mb


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
    """Keep top-k fraction of weight values by magnitude; zero out the rest."""
    compressed = []
    for grad in gradients:
        flat      = grad.flatten()
        num_keep  = max(1, int(len(flat) * k))
        threshold = np.sort(np.abs(flat))[-num_keep]
        mask      = np.abs(flat) >= threshold
        compressed.append((flat * mask).reshape(grad.shape))
    return compressed


def serialized_size_mb(model):
    """Size of the model's state_dict as actually serialized.

    get_model_size_mb() walks .parameters(), which is EMPTY for quantized
    modules (their weights live in _packed_params), so it reports 0.0 MB for
    an INT8 model. Serializing the state_dict counts the packed weights.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def count_compressed_bytes(params):
    return int(sum(np.count_nonzero(p) for p in params)) * 4


class EnergyFLClient(fl.client.NumPyClient):

    def __init__(self, client_id, X_train, y_train, X_test, y_test,
                 device_type="pi4", compression_k=0.1, quantize=False,
                 fixed_epochs=None):
        self.client_id     = client_id
        self.device_type   = device_type
        self.compression_k = compression_k
        # Measurement only: quantization runs on a throwaway copy each round.
        # The FP32 parameters are what get returned and aggregated (see fit()).
        self.quantize      = quantize

        profile           = DEVICE_PROFILES.get(device_type, DEVICE_PROFILES["pi4"])
        self.local_epochs = profile["local_epochs"]
        self.batch_size   = profile["batch_size"]

        # H3 control arm: identical epoch count on every device, overriding the
        # adaptive 5/2/1 profile. Batch size still follows the device profile.
        self.adaptive_epochs = fixed_epochs is None
        if fixed_epochs is not None:
            self.local_epochs = fixed_epochs

        print(f"[Client {client_id}] Device: {device_type} | "
              f"Epochs: {self.local_epochs} | Batch: {self.batch_size} | "
              f"Compression k={compression_k} | "
              f"Epoch mode: {'adaptive' if self.adaptive_epochs else 'fixed'}")

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
        memory_before_mb = get_memory_usage_mb()
        self.set_parameters(parameters)

        current_round = int(config.get("round", 1))
        lr = self._get_lr(current_round)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.model.train()
        total_loss = 0.0
        train_start = time.perf_counter()

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

        # Local training only - excludes the INT8 measurement block below.
        train_time_s = time.perf_counter() - train_start

        memory_fp32_mb     = 0.0
        memory_int8_mb     = 0.0
        model_size_fp32_mb = 0.0
        model_size_int8_mb = 0.0
        if self.quantize:
            # FP32 point: training done, nothing quantized yet.
            memory_fp32_mb     = get_memory_usage_mb()
            model_size_fp32_mb = serialized_size_mb(self.model)
            # deepcopy is required: apply_int8_quantization() mutates in place
            # (prepare/convert both use inplace=True). self.model must stay FP32.
            q_model            = apply_int8_quantization(
                copy.deepcopy(self.model), calibration_loader=self.train_loader
            )
            memory_int8_mb     = get_memory_usage_mb()
            model_size_int8_mb = serialized_size_mb(q_model)
            del q_model
            print(f"  [Client {self.client_id}] Round {current_round} | "
                  f"INT8 measure | RSS {memory_fp32_mb:.1f} -> {memory_int8_mb:.1f} MB | "
                  f"model {model_size_fp32_mb:.4f} -> {model_size_int8_mb:.4f} MB "
                  f"({(1 - model_size_int8_mb / max(model_size_fp32_mb, 1e-9)) * 100:.1f}% smaller)")

        # Always the untouched FP32 model - never the quantized copy.
        updated_params   = self.get_parameters(config={})
        compressed_bytes = count_compressed_bytes(updated_params)
        num_samples      = len(self.train_loader.dataset)
        memory_after_mb  = get_memory_usage_mb()

        return updated_params, num_samples, {
            "train_loss":       float(total_loss / self.local_epochs),
            "client_id":        self.client_id,
            "device":           self.device_type,
            "epochs":           self.local_epochs,
            "lr":               float(lr),
            "compressed_bytes": compressed_bytes,
            "train_time_s":     float(train_time_s),
            "memory_before_mb": float(memory_before_mb),
            "memory_after_mb":  float(memory_after_mb),
            "memory_fp32_mb":     float(memory_fp32_mb),
            "memory_int8_mb":     float(memory_int8_mb),
            "model_size_fp32_mb": float(model_size_fp32_mb),
            "model_size_int8_mb": float(model_size_int8_mb),
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
                 device_type="pi4", compression_k=0.1, quantize=False,
                 fixed_epochs=None):
    client = EnergyFLClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        device_type=device_type,
        compression_k=compression_k,
        quantize=quantize,
        fixed_epochs=fixed_epochs,
    )
    fl.client.start_numpy_client(server_address=server_address, client=client)
