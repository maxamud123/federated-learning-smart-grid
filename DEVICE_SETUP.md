# Device Setup Guide

The intended per-device configuration for the FL experiment, and the actual status of each
device.

**Read this first: every experiment log in this repository was produced with the server and
all three clients on one host laptop, communicating over localhost. No federated training
run was executed on physical edge hardware.** The sections below record what each target
device *is* and where its setup actually stands.

| Client | Device profile | Epochs | Batch | Ran on | Hardware status |
| --- | --- | --- | --- | --- | --- |
| 1 | `pi4` — Raspberry Pi 4, 2GB | 5 | 32 | Host laptop | Emulated by choice; never purchased |
| 2 | `pi_zero` — Raspberry Pi Zero W, 512MB, single-core | 2 | 16 | Host laptop | Purchased; no network connection achieved |
| 3 | `esp32` — ESP32-WROOM, 520KB | 1 | 8 | Host laptop | Emulated of necessity; cannot run this code |

---

## Network Overview

```text
Phone Hotspot / Home WiFi (192.168.1.x)
    ├── Laptop                 192.168.1.10   FL Server + Client 1 (Pi 4 profile, emulated)
    ├── Raspberry Pi Zero W    192.168.1.12   Client 2 (board in hand; never connected)
    └── ESP32                  192.168.1.13   Client 3 (profile only; cannot run this code)
```

> **The addresses above are illustrative placeholders showing how a multi-device setup
> would be addressed; every experiment log in this repository ran on localhost.**

> Your actual IPs may differ. Find them with `hostname -I` (Linux/Pi) or `ipconfig` (Windows).

---

## 1. Laptop (FL Server)

### Requirements

- Python 3.10+
- All packages from `requirements.txt`

### Steps

```bash
# Install dependencies
pip install -r requirements.txt

# Find your laptop IP on the shared network (use this as --server_ip on clients)
ipconfig        # Windows
hostname -I     # Linux

# Start the server (0.0.0.0 means accept connections from any device on the network)
python main.py --mode server --server_ip 0.0.0.0 --experiment baseline --num_rounds 50
```

---

## 2. Raspberry Pi 4 profile (Client 1 — emulated on the host, by choice)

No Raspberry Pi 4 was purchased. Client 1 runs on the laptop/host machine using the `pi4`
device profile (5 epochs, batch 32). Emulating it there is a stated methodological choice
in the thesis, not a gap. The profile reproduces the Pi 4's per-round work budget; it does
not reproduce the Pi 4's wall-clock speed or its 2GB memory ceiling.

### Requirements

- Same laptop/host machine used for the FL server (Python 3.10+, `requirements.txt` installed)

### Run client

Open a new terminal on the laptop:

```bash
python main.py --mode client --client_id 1 --device pi4 --compress --server_ip 192.168.1.10
```

---

## 3. Raspberry Pi Zero W (Client 2)

### Status: board in hand, no run achieved — unresolved

This is the one physical board purchased for the project. It never stayed on a network long
enough to run a client:

- Connectivity attempts failed across **four separate networks**.
- The WiFi credentials were verified in the board's boot configuration before each attempt.
- The board would associate briefly, then drop the connection without recovering.

**No federated training run was attempted on the physical Pi Zero W.** The problem is
unresolved as of the latest commit, and the client-2 results in `experiments/` are the
`pi_zero` profile running on the host laptop. The steps below are the intended setup
procedure; they were not completed past flashing the card.

### Hardware needed (Pi Zero W)

- Raspberry Pi Zero W (single-core, 512MB)
- MicroSD card (32GB)
- micro-USB power supply (5V 2A)

### Setup steps (Pi Zero W) — intended, not completed

**Flash OS:**

1. Flash **Raspberry Pi OS Lite (32-bit)** — the Pi Zero W's ARMv6 CPU requires 32-bit
2. Enable SSH and WiFi in Imager settings

**First boot:**

```bash
ssh pi@192.168.1.12

sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip git -y

git clone https://github.com/maxamud123/federated-learning-smart-grid.git
cd federated-learning-smart-grid
pip3 install -r requirements.txt

# Copy data splits from laptop
# scp -r data/splits pi@192.168.1.12:~/federated-learning-smart-grid/data/
```

> Pi Zero W is slow — `pip install` may take 10–20 minutes. Be patient. This step was
> never reached: the board did not hold a network connection long enough to clone the repo.

**Run client:**

```bash
python3 main.py --mode client --client_id 2 --device pi_zero --compress --server_ip 192.168.1.10
```

---

## 4. ESP32 (Client 3)

### Hardware needed (ESP32)

- ESP32-WROOM development board
- USB cable (for power and serial)

### Status: the ESP32 has never run this code, and cannot

The ESP32 has 520KB of RAM, which cannot hold the PyTorch tensors or the Flower runtime.
The federated client code has never executed on the board. The logs show the client-3
process at roughly **470 MB RSS** (`memory_after_mb` in any `--quantize` run) — about
**900x the device's entire memory**. This has always been true of the implementation; only
earlier versions of this documentation overstated it.

**What actually runs, in every logged experiment:** the ESP32 *device profile* — 1 local
epoch, batch size 8 — on the host laptop.

```bash
python main.py --mode client --client_id 3 --device esp32 --compress --server_ip 192.168.1.10
```

That constrains the per-round work to the profile's epoch and batch budget. It does not
reproduce the ESP32's 520KB memory ceiling, its CPU or its radio, and it is not evidence
that the model fits on an ESP32.

**Not attempted — real ESP32 with MicroPython:** flashing MicroPython and writing a
lightweight on-device client would mean re-implementing training without PyTorch or Flower.
That is outside the scope of this implementation and was not done.

---

## Firewall / Port

Make sure port **8080** is open on the laptop (server):

```bash
# Windows — allow port 8080 in Windows Firewall
netsh advfirewall firewall add rule name="FL Server" dir=in action=allow protocol=TCP localport=8080

# Linux
sudo ufw allow 8080
```

---

## Quick Checklist

> This checklist covers the multi-device deployment described above, which was never
> carried out. For the runs that produced the logs in `experiments/`, only the last two
> items apply.

- [ ] All devices connected to the same WiFi / hotspot
- [ ] Laptop IP noted (e.g. `192.168.1.10`)
- [ ] Port 8080 open on laptop firewall
- [ ] Data splits copied to each Pi (`data/splits/`)
- [ ] Dependencies installed on each device
- [ ] Start server **before** starting clients

---

## Top-K Weight Sparsification (`--compress`)

```bash
python main.py --mode client --client_id 1 --device pi4 --compress
```

**This sparsifies model weights, not gradients.** `--compress` sets `compression_k=0.1`,
and `get_parameters()` in `src/client/fl_client.py` applies `top_k_compression()` to the
model's **weight tensors** on the way out: it keeps the largest 10% of each tensor by
magnitude and zeros the rest. Gradients are never sparsified — they are used in full during
local training, and the reduction applies only to the parameters transmitted to the server.

Earlier versions of this documentation and of the `--compress` help text called this
"Top-K gradient compression," which named the wrong quantity. The behaviour of the code has
not changed.

---

## INT8 Quantization Measurement (`--quantize`)

```bash
python main.py --mode client --client_id 1 --device pi4 --quantize
```

**Methodological limitation — state this explicitly when reporting H1.**

`--quantize` is a *measurement* flag, not an optimization flag. Each round, after
local training finishes, the client quantizes a **throwaway deep copy** of the
trained model and records its memory footprint. The copy is discarded
immediately afterwards.

The parameters sent to the server and used in aggregation remain the **original
FP32 weights**, unchanged from the non-quantized path. Federated averaging in
this system operates on FP32 tensors only; mixing INT8 and FP32 parameters in
`FedAvg` is not supported and is deliberately not attempted here.

So the recorded figures answer "how much smaller *would* the model be on an edge
device after INT8 quantization?" They do **not** show a federated system that
trains or aggregates in INT8. That remains future work.

Fields written per client per round:

| Field | Meaning |
|---|---|
| `memory_fp32_mb` | Process RSS after training, before quantization |
| `memory_int8_mb` | Process RSS after quantizing the copy |
| `model_size_fp32_mb` | Serialized FP32 `state_dict` size |
| `model_size_int8_mb` | Serialized INT8 `state_dict` size |

**Read `model_size_*`, not `memory_*`, for H1.** Process RSS is dominated by the
Python interpreter and PyTorch itself (~300–400 MB) while the model is ~0.19 MB,
so RSS *rises* when the quantized copy is allocated. `get_model_size_mb()` is
also unusable here: it walks `.parameters()`, which is empty for quantized
modules (weights move to `_packed_params`), so it reports 0.0 MB and an apparent
100% reduction. The `model_size_*` fields serialize the `state_dict` instead,
which counts packed INT8 weights correctly.

Without `--quantize`, none of this runs and all four fields are `0.0`.
