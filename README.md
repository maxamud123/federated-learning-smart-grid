# Federated Learning for Smart Grid Energy Forecasting
### A Case Study in Somalia

**Thesis:** Optimizing Federated Learning for Resource-Constrained Edge Devices in Smart Grids
**Author:** Mohamoud Abukar | Reg No: 202413001
**Supervisor:** Dr. KAMUHANDA Danny
**University:** Kigali Independent University (ULK) | MSc Internet Systems | 2024-2025

---

## Project Structure

```
federated-learning-smart-grid/
├── data/
│   ├── raw/              ← Place UCI dataset here
│   ├── processed/        ← Cleaned hourly data
│   └── splits/           ← Per-client numpy arrays
├── src/
│   ├── models/
│   │   └── lstm_model.py         ← LSTM + INT8 quantization
│   ├── preprocessing/
│   │   └── data_loader.py        ← UCI loading, windowing, normalization
│   ├── client/
│   │   └── fl_client.py          ← Flower client + Top-K compression
│   ├── server/
│   │   └── fl_server.py          ← Flower server + FedAvg
│   └── evaluation/
│       └── metrics.py            ← RMSE, MAE, memory, communication metrics
├── experiments/
│   ├── results/          ← JSON result files
│   └── logs/             ← Per-round training logs
├── main.py               ← Entry point for all modes
├── requirements.txt
└── README.md
```

---

## Three Optimizations

| Optimization | Technique | Target Reduction |
|---|---|---|
| Memory | INT8 Quantization | ~50% RAM reduction |
| Communication | Top-K Compression (k=0.1) | ~60% bandwidth saving |
| Training Time | Adaptive Local Epochs | ~40% time reduction |

---

## How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download UCI dataset
Download from: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
Place the file at: `data/raw/household_power_consumption.txt`

### Step 3: Preprocess data
```bash
python main.py --mode preprocess --data data/raw/household_power_consumption.txt
```

### Step 4: Test pipeline locally
```bash
python main.py --mode test
```

### Step 5: Run FL experiment (open 4 terminals)

**Terminal 1 - Server:**
```bash
python main.py --mode server --experiment baseline --num_rounds 50
```

**Terminal 2 - Client 1 (Pi 4):**
```bash
python main.py --mode client --client_id 1 --device pi4 --compress
```

**Terminal 3 - Client 2 (Pi Zero):**
```bash
python main.py --mode client --client_id 2 --device pi_zero --compress
```

**Terminal 4 - Client 3 (ESP32):**
```bash
python main.py --mode client --client_id 3 --device esp32 --compress
```

---

## Hardware Targets

| Device | RAM | Local Epochs |
|---|---|---|
| Raspberry Pi 4 | 2GB | 5 |
| Raspberry Pi Zero | 512MB | 2 |
| ESP32 | 520KB | 1 |
