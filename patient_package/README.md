# 🖥️ Patient Telemetry Agent — Quick Setup Guide

## What This Does
This folder turns **any Windows/Linux machine** into a monitored "patient" in the Edge-AI Predictive Maintenance system. Once started, it collects hardware telemetry (CPU, GPU, RAM, Disk, PSU, Fans) and streams it to the RPi4 Doctor node via MQTT.

## Setup (3 Steps)

### Step 1: Install Python
- **Windows:** Download from [python.org](https://www.python.org/downloads/). Check ☑️ **"Add Python to PATH"** during install.
- **Linux:** `sudo apt install python3 python3-pip`

### Step 2: Install Dependencies
- **Windows:** Double-click `INSTALL_DEPS.bat`
- **Linux:** `pip3 install -r requirements.txt`

### Step 3: Configure & Start
1. Open `START_PATIENT.bat` (Windows) or `start_patient.sh` (Linux) in a text editor
2. Change the `BROKER_IP` to your RPi4's hostname or IP (default: `rpi4-edge.tail1ab139.ts.net` via Tailscale)
3. Optionally set `SERVER_NAME` to a friendly name (e.g., `"Lab-PC-01"`)
4. Save and double-click `START_PATIENT.bat` (or `bash start_patient.sh` on Linux)

That's it! The machine will appear on the RPi4 dashboard within 5 seconds.

## Files in This Package

| File | Purpose |
|------|---------|
| `START_PATIENT.bat` | Windows launcher (edit BROKER_IP here) |
| `start_patient.sh` | Linux/macOS launcher |
| `INSTALL_DEPS.bat` | One-click dependency installer (Windows) |
| `requirements.txt` | Python dependencies list |
| `mqtt_publisher_v2.py` | Main telemetry publisher |
| `collector_v2.py` | Plugin-based hardware collector |
| `auto_discovery.py` | Auto-detects available hardware |
| `server_identity.py` | Generates & persists unique server UUID |
| `plugins/` | Hardware collection plugins (CPU, GPU, RAM, Disk, PSU/Fan) |

## How It Works
1. On first run, the agent auto-discovers your hardware and generates a unique UUID
2. Every 5 seconds, it collects metrics from all available hardware
3. Data is published via MQTT to the RPi4 broker
4. The RPi4 runs AI models on the data and displays results on the dashboard
5. If the RPi4 is unreachable, the agent keeps retrying automatically

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: psutil` | Run `INSTALL_DEPS.bat` or `pip install psutil paho-mqtt` |
| `Connection refused` | Check RPi4 is powered on and MQTT broker is running |
| No GPU metrics | Normal if machine has no NVIDIA GPU. Install `pynvml` if it does. |
| Dashboard doesn't show this machine | Wait 10 seconds. Check `BROKER_IP` matches your RPi4. |
