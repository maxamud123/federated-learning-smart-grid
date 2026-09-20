# Optimizing Federated Learning for Resource-Constrained Edge Devices in Smart Grids

## A Case Study in Somalia

**Thesis:** Optimizing Federated Learning for Resource-Constrained Edge Devices in Smart Grids
**Author:** Mohamoud Abukar | Reg No: 202413001
**Supervisor:** Dr. KAMUHANDA Danny
**University:** Kigali Independent University (ULK) | MSc Internet Systems | 2024-2025

<!-- cspell:words Mohamoud Abukar KAMUHANDA numpy lstm LSTM RMSE hotspot sparsification sparsified sparsifies WROOM -->

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
│   │   └── fl_client.py          ← Flower client + Top-K weight sparsification
│   ├── server/
│   │   └── fl_server.py          ← Flower server + FedAvg
│   └── evaluation/
│       └── metrics.py            ← RMSE, MAE, memory, communication metrics
├── experiments/
│   ├── results/          ← JSON result files
│   └── logs/             ← Per-round training logs
├── DEVICE_SETUP.md       ← Device profiles and per-device hardware status (all clients run on the host)
├── main.py               ← Entry point for all modes
├── requirements.txt
└── README.md
```

---

## Three Optimizations

| Optimization  | Technique                 | Flag               | Target                | Measured 2026-09-10              |
| ------------- | ------------------------- | ------------------ | --------------------- | -------------------------------- |
| Memory        | INT8 Quantization         | `--quantize`       | ~50% RAM reduction    | **56.9%** model-size reduction † |
| Communication | Top-K weight sparsification (k=0.1) | `--compress` | ~60% bandwidth saving | **90.1%** payload saving |
| Training Time | Adaptive Local Epochs | on by default | ~40% time reduction (not evaluable on this host) | **11.1%** lower RMSE than fixed epochs ‡ |

† Measured as serialized `state_dict` size (FP32 0.1959 MB -> INT8 0.0844 MB), **not**
process RAM. Process RSS does not fall: the model is ~0.19 MB against ~400 MB of
Python interpreter and PyTorch runtime, so the saving is invisible at process level.
Cite the `model_size_*` log fields, never `memory_*`. See [DEVICE_SETUP.md](DEVICE_SETUP.md).

‡ **No timing claim is made for this optimization.** The same-workload control run
showed that `train_time_s` on this host measures process contention between the three
co-located client processes, not per-device training cost, so no wall-clock reduction can
be attributed to adaptive epochs from these logs. Establishing one requires physical
hardware, which this project does not have (see
[Execution Environment](#execution-environment-what-ran-where)).

What the H3 arms do support is an accuracy comparison, same round, round 33: adaptive
(5/2/1 epochs) RMSE **0.543578** against `--fixed_epochs 5` RMSE **0.611595**, an 11.1%
reduction.

Caveat to state when reporting H3: each arm's *best* RMSE is nearly equal - adaptive
0.537647 at round 14 against fixed 0.538918 at round 5, a 0.24% gap. The round-33
advantage comes from the fixed arm degrading after its early minimum, not from adaptive
reaching a better floor.

---

## Execution Environment: What Ran Where

**Every experiment log in `experiments/` was produced on one host laptop, with the server
and all three clients communicating over localhost. No result in this repository was
produced on physical edge hardware.**

The three clients differ only by *device profile* — the local epoch count and batch size in
`DEVICE_PROFILES` in [`src/client/fl_client.py`](src/client/fl_client.py) — not by the
machine they run on.

| Client | Device profile | Epochs | Batch | Ran on | Hardware status |
| --- | --- | --- | --- | --- | --- |
| 1 | `pi4` — Raspberry Pi 4, 2GB | 5 | 32 | Host laptop | Emulated by choice; no Pi 4 was purchased |
| 2 | `pi_zero` — Raspberry Pi Zero W, 512MB, single-core | 2 | 16 | Host laptop | Board purchased; physical run **not achieved, unresolved** |
| 3 | `esp32` — ESP32-WROOM, 520KB | 1 | 8 | Host laptop | Emulated of necessity; the board cannot run this code |

### Raspberry Pi 4 — emulated, by methodological choice

No Raspberry Pi 4 was purchased. The `pi4` profile is deliberately emulated on the host
machine, and the thesis states this as a methodological choice rather than reporting it as
a gap. What the profile reproduces is the Pi 4's per-round work budget (5 local epochs,
batch 32) — not its wall-clock speed and not its 2GB memory ceiling.

### Raspberry Pi Zero W — physical board obtained, no run achieved

This is the one physical board purchased for the project, and it never stayed on a network
long enough to run a client. Connectivity attempts failed across four separate networks;
the WiFi credentials were verified in the board's boot configuration before each attempt.
The board would associate briefly, then drop the connection without recovering.

**No federated training run was attempted on the physical Pi Zero W.** The problem is
unresolved as of the latest commit. The client-2 numbers in `experiments/` are the
`pi_zero` profile running on the host laptop, like the other two.

### ESP32 — emulated of necessity; the board cannot run this code

The ESP32 has never executed the federated client code, and cannot: 520KB of RAM cannot
hold the PyTorch tensors or the Flower runtime. The logs bear this out — the client-3
process reports roughly **470 MB RSS** (`memory_after_mb` in any `--quantize` run), about
**900x the entire memory of the device it stands for**. The ESP32 is represented solely by
its device profile (1 local epoch, batch size 8) running on the host laptop.

This has always been true of the implementation. Only earlier versions of this
documentation overstated it.

> See [DEVICE_SETUP.md](DEVICE_SETUP.md) for the device profiles and the per-device
> hardware status.

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

Open 4 terminals. The clean baseline takes **no optimization flags** - adding any
of them makes it an optimized run, not a baseline:

```bash
# Terminal 1 - Server
python main.py --mode server --experiment baseline --num_rounds 50

# Terminal 2 - Client 1
python main.py --mode client --client_id 1 --device pi4

# Terminal 3 - Client 2
python main.py --mode client --client_id 2 --device pi_zero

# Terminal 4 - Client 3
python main.py --mode client --client_id 3 --device esp32
```

To run the **optimized** experiment instead, add `--compress` to every client and
change the experiment name:

```bash
python main.py --mode server --experiment optimized --num_rounds 50
python main.py --mode client --client_id 1 --device pi4     --compress
python main.py --mode client --client_id 2 --device pi_zero --compress
python main.py --mode client --client_id 3 --device esp32   --compress
```

#### Optional client flags

Each flag is independent and off by default. Combine them as needed.

| Flag | Effect |
| --- | --- |
| `--compress` | Top-K sparsification of the transmitted **model weights** (k=0.1): `get_parameters()` keeps the largest 10% of each weight tensor by magnitude and zeros the rest. Not gradient compression — gradients are never sparsified. Changes what is transmitted. |
| `--quantize` | Records INT8 vs FP32 model size each round, measured on a local copy. Measurement only - aggregation stays FP32. See [DEVICE_SETUP.md](DEVICE_SETUP.md). |
| `--fixed_epochs N` | Forces N local epochs on every client, overriding the adaptive 5/2/1 profile. Used as the H3 control arm. |

```bash
# INT8 memory-footprint evidence (H1)
python main.py --mode client --client_id 1 --device pi4     --quantize
python main.py --mode client --client_id 2 --device pi_zero --quantize
python main.py --mode client --client_id 3 --device esp32   --quantize

# Fixed-epoch control arm (H3) - every client does 5 epochs
python main.py --mode client --client_id 1 --device pi4     --fixed_epochs 5
python main.py --mode client --client_id 2 --device pi_zero --fixed_epochs 5
python main.py --mode client --client_id 3 --device esp32   --fixed_epochs 5
```

#### Option B — Multi-device over WiFi / hotspot (illustrative; never executed)

**No experiment in this repository was run this way.** The commands below document how a
multi-device deployment *would* be addressed, and the IP addresses in them are
placeholders. Client 2 is the only one of the three with physical hardware, and that board
has not joined a network — see
[Execution Environment](#execution-environment-what-ran-where).

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

**Client 1, `pi4` profile — Terminal 2, on the host machine. This profile is always
emulated; there is no Pi 4 to run it on:**

```bash
python main.py --mode client --client_id 1 --device pi4 --compress --server_ip 192.168.1.10
```

**Client 2, `pi_zero` profile — intended to run over SSH on the physical Pi Zero W. Never
achieved: the board has never held a network connection, so this command has never been run
on the board. Every logged client-2 result comes from this profile on the host:**

```bash
python main.py --mode client --client_id 2 --device pi_zero --compress --server_ip 192.168.1.10
```

**Client 3, `esp32` profile — on the host machine. There is no on-device variant of this
command: the ESP32 cannot run the client code (520KB RAM against ~470 MB process RSS):**

```bash
python main.py --mode client --client_id 3 --device esp32 --compress --server_ip 192.168.1.10
```

> Replace `192.168.1.10` with your laptop's actual IP on the shared network.

---

## Hardware Targets

These are the devices the profiles *represent*. None of them ran the client code — see
[Execution Environment](#execution-environment-what-ran-where).

| Target device | RAM | Local epochs | Power | In hand? | Ran the client? |
| --- | --- | --- | --- | --- | --- |
| Raspberry Pi 4 | 2GB | 5 | n/a | No — never purchased | No — emulated on the host by choice |
| Raspberry Pi Zero W (single-core) | 512MB | 2 | micro-USB 5V | Yes | No — never joined a network |
| ESP32-WROOM | 520KB | 1 | USB 5V | Yes | No — cannot hold the runtime |
