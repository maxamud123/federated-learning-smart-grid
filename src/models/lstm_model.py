"""
LSTM Model for Energy Consumption Forecasting
Thesis: Optimizing FL for Resource-Constrained Edge Devices in Smart Grids
Author: Mohamoud Abukar | Supervisor: Dr. KAMUHANDA Danny | ULK 2024-2025
"""

import torch
import torch.nn as nn
import torch.quantization


class LSTMModel(nn.Module):
    """
    LSTM-based energy forecasting model.
    Input:  (batch, seq_len=24, features=1)
    Output: (batch, 1) - next hour prediction
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])   # last time step only
        return out


def get_model_size_mb(model):
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


def apply_int8_quantization(model, calibration_loader=None):
    """Post-training static INT8 quantization. CPU-only (fbgemm backend)."""
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    if calibration_loader is not None:
        with torch.no_grad():
            for inputs, _ in calibration_loader:
                model(inputs)

    torch.quantization.convert(model, inplace=True)
    return model


def get_model_parameters(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
