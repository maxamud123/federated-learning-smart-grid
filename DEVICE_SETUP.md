# Device Setup Guide

Step-by-step instructions to configure each hardware device for the FL experiment.
All devices connect via **home WiFi or phone hotspot** — no dedicated router needed.

<!-- cspell:words hotspot Hotspot Imager venv WROOM netsh advfirewall localport -->

---

## Network Overview

```text
Phone Hotspot / Home WiFi (192.168.1.x)
    ├── Laptop                 192.168.1.10   FL Server + Client 1 (Pi 4 profile, emulated)
    ├── Raspberry Pi Zero 2 W  192.168.1.12   Client 2 (physical hardware)
    └── ESP32                  192.168.1.13   Client 3 (physical hardware)
```

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

## 2. Raspberry Pi 4 profile (Client 1 — emulated on laptop)

The Raspberry Pi 4 client is **not run on physical Pi 4 hardware**. It runs on the laptop/host machine using the existing `pi4` device profile (5 epochs, batch 32) to reproduce the Pi 4's resource constraints. Physical hardware validation for this project is performed only on the Raspberry Pi Zero 2 W and the ESP32.

### Requirements

- Same laptop/host machine used for the FL server (Python 3.10+, `requirements.txt` installed)

### Run client

Open a new terminal on the laptop:

```bash
python main.py --mode client --client_id 1 --device pi4 --compress --server_ip 192.168.1.10
```

---

## 3. Raspberry Pi Zero 2 W (Client 2)

### Hardware needed (Pi Zero 2 W)

- Raspberry Pi Zero 2 W
- MicroSD card (32GB)
- micro-USB power supply (5V 2A)

### Setup steps (Pi Zero 2 W)

**Flash OS:**

1. Flash **Raspberry Pi OS Lite (32-bit)** — use 32-bit for Pi Zero 2 W compatibility
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

> Pi Zero 2 W is slow — `pip install` may take 10–20 minutes. Be patient.

**Run client:**

```bash
python3 main.py --mode client --client_id 2 --device pi_zero --compress --server_ip 192.168.1.10
```

---

## 4. ESP32 (Client 3)

### Hardware needed (ESP32)

- ESP32-WROOM development board
- USB cable (for power and serial)

### Note on ESP32

The ESP32 has only 520KB RAM — it cannot run Python directly. Two options:

**Option A (Recommended for thesis simulation):** Run on laptop as a simulated ESP32:

```bash
python main.py --mode client --client_id 3 --device esp32 --compress --server_ip 192.168.1.10
```

This uses the ESP32 device profile (1 epoch, minimal memory) on your laptop, accurately simulating the constraints.

**Option B (Real ESP32 with MicroPython):** Flash MicroPython and use a lightweight client script. This requires significant adaptation of the FL client code and is outside the scope of the current implementation.

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

- [ ] All devices connected to the same WiFi / hotspot
- [ ] Laptop IP noted (e.g. `192.168.1.10`)
- [ ] Port 8080 open on laptop firewall
- [ ] Data splits copied to each Pi (`data/splits/`)
- [ ] Dependencies installed on each device
- [ ] Start server **before** starting clients
