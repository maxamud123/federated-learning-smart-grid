# Device Setup Guide

Step-by-step instructions to configure each hardware device for the FL experiment.
All devices connect via **home WiFi or phone hotspot** — no dedicated router needed.

<!-- cspell:words hotspot Hotspot Imager venv WROOM netsh advfirewall localport -->

---

## Network Overview

```text
Phone Hotspot / Home WiFi (192.168.1.x)
    ├── Laptop          192.168.1.10   FL Server  :8080
    ├── Raspberry Pi 4  192.168.1.11   Client 1
    ├── Raspberry Pi 0  192.168.1.12   Client 2
    └── ESP32           192.168.1.13   Client 3
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

## 2. Raspberry Pi 4 (Client 1)

### Hardware needed

- Raspberry Pi 4 (2GB RAM)
- MicroSD card (32GB, Class 10)
- USB-C power supply (5V 3A)

### Setup steps

**Flash OS:**

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Flash **Raspberry Pi OS Lite (64-bit)** to the MicroSD
3. In Imager settings, enable SSH and set WiFi credentials before flashing

**First boot:**

```bash
# SSH into the Pi (replace with your Pi's IP)
ssh pi@192.168.1.11

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
sudo apt install python3-pip python3-venv git -y

# Clone the project
git clone https://github.com/maxamud123/federated-learning-smart-grid.git
cd federated-learning-smart-grid

# Install Python packages
pip3 install -r requirements.txt

# Copy data splits from laptop (run this on your laptop)
# scp -r data/splits pi@192.168.1.11:~/federated-learning-smart-grid/data/
```

**Run client:**

```bash
python3 main.py --mode client --client_id 1 --device pi4 --compress --server_ip 192.168.1.10
```

---

## 3. Raspberry Pi Zero (Client 2)

### Hardware needed (Pi Zero)

- Raspberry Pi Zero W (or Zero 2 W)
- MicroSD card (32GB)
- micro-USB power supply (5V 2A)

### Setup steps (Pi Zero)

**Flash OS:**

1. Flash **Raspberry Pi OS Lite (32-bit)** — use 32-bit for Pi Zero compatibility
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

> Pi Zero is slow — `pip install` may take 10–20 minutes. Be patient.

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
