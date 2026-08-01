# Optimizing Federated Learning for Resource-Constrained Edge Devices in Smart Grids

## A Case Study in Somalia

**Thesis:** Optimizing Federated Learning for Resource-Constrained Edge Devices in Smart Grids
**Author:** Mohamoud Abukar | Reg No: 202413001
**Supervisor:** Dr. KAMUHANDA Danny
**University:** Kigali Independent University (ULK) | MSc Internet Systems | 2024-2025

<!-- cspell:words Mohamoud Abukar KAMUHANDA numpy lstm LSTM RMSE hotspot -->

---

## Project Structure

```text
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
├── DEVICE_SETUP.md       ← Hardware setup guide (Pi 4 emulated on laptop, Raspberry Pi Zero 2 W, ESP32)
├── main.py               ← Entry point for all modes
├── requirements.txt
└── README.md
```

---

## Three Optimizations

| Optimization  | Technique                  | Target Reduction      |
| ------------- | -------------------------- | --------------------- |
| Memory        | INT8 Quantization          | ~50% RAM reduction    |
| Communication | Top-K Compression (k=0.1)  | ~60% bandwidth saving |
| Training Time | Adaptive Local Epochs      | ~40% time reduction   |

---

## Physical Setup (No Router Needed)

The Raspberry Pi 4 client runs on the laptop/host machine using the `pi4` device profile — it is not run on physical Pi 4 hardware. Physical hardware validation is performed on the Raspberry Pi Zero 2 W and the ESP32.

All devices connect through **home WiFi or a phone hotspot**. No dedicated router required.

| Device | Role | IP | RAM | Local Epochs |
| --- | --- | --- | --- | --- |
| Laptop | FL Server | 192.168.1.10 | — | — |
| Raspberry Pi 4 (emulated on laptop) | Client 1 | local (laptop) | 2GB | 5 |
| Raspberry Pi Zero 2 W | Client 2 | 192.168.1.12 | 512MB | 2 |
| ESP32 | Client 3 | 192.168.1.13 | 520KB | 1 |

> See [DEVICE_SETUP.md](DEVICE_SETUP.md) for step-by-step hardware configuration.

---

## How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download UCI dataset

Download from: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
Place the file at: `data/raw/household_power_consumption.txt`

### Step 3: Preprocess data

```bash
python main.py --mode preprocess --data data/raw/household_power_consumption.txt
```

### Step 4: Test pipeline locally

```bash
python main.py --mode test
```

### Step 5: Run FL experiment

#### Option A — Single machine (simulation, localhost)

Open 4 terminals:

```bash
# Terminal 1 – Server
python main.py --mode server --experiment baseline --num_rounds 50

# Terminal 2 – Client 1
python main.py --mode client --client_id 1 --device pi4 --compress

# Terminal 3 – Client 2
python main.py --mode client --client_id 2 --device pi_zero --compress

# Terminal 4 – Client 3
python main.py --mode client --client_id 3 --device esp32 --compress
```

#### Option B — Physical hardware validation over WiFi / hotspot

First, find your laptop's IP on the shared network:

```bash
# Windows
ipconfig
# Linux / Pi
hostname -I
```

**Laptop (FL Server) — Terminal 1:**

```bash
python main.py --mode server --server_ip 0.0.0.0 --experiment baseline --num_rounds 50
```

**Raspberry Pi 4 (emulated on laptop) — Terminal 2, run locally on the host machine:**

```bash
python main.py --mode client --client_id 1 --device pi4 --compress --server_ip 192.168.1.10
```

**Raspberry Pi Zero 2 W — run via SSH (physical hardware validation):**

```bash
python main.py --mode client --client_id 2 --device pi_zero --compress --server_ip 192.168.1.10
```

**ESP32 — run via SSH or serial (physical hardware validation):**

```bash
python main.py --mode client --client_id 3 --device esp32 --compress --server_ip 192.168.1.10
```

> Replace `192.168.1.10` with your laptop's actual IP on the shared network.

---

## Hardware Targets

| Device | RAM | Local Epochs | Power |
| --- | --- | --- | --- |
| Raspberry Pi 4 (emulated on laptop) | 2GB | 5 | n/a (runs on laptop) |
| Raspberry Pi Zero 2 W | 512MB | 2 | micro-USB 5V |
| ESP32 | 520KB | 1 | USB 5V |
