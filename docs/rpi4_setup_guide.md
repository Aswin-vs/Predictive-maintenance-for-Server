# 🍓 Raspberry Pi 4 Edge Node Setup Guide

> **Last Updated: 2026-09-13 (Phase G3 — RPi4 OS Reinstalled, 7 Edge Models)**

This guide covers setting up the Raspberry Pi 4 (4GB) as the "Doctor" (Edge Node) for the Edge-AI Predictive Maintenance system.

## Quick Setup (Automated)

If you have SSH access to a freshly formatted RPi4, run from the LOQ laptop:

```bash
# Full automated setup: apt → pip → deploy → systemd → verify
python setup_rpi4_fresh.py

# Or if partially set up, just finish remaining steps:
python setup_rpi4_continue.py
```

These scripts handle everything below automatically. The manual steps are documented for reference.

---

## 1. Operating System Installation

1. Download and install [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Insert your **USB Pendrive (32GB+ USB 3.0)** or MicroSD card.
3. In Raspberry Pi Imager:
   - **Device:** Select `Raspberry Pi 4`
   - **OS:** Select `Raspberry Pi OS Lite (64-bit)` — Debian 13 Trixie
   - **Storage:** Select your media
4. Click **Next** → **Edit Settings**:
   - Set hostname to `rpi`
   - Enable SSH (password authentication)
   - Set username: `pi`, password: `edge12345`
   - Configure WiFi or use USB-Ethernet tethering
5. Click **Write** and wait for verification.

## 2. Network — Windows Mobile Hotspot

The RPi4 connects via Windows Mobile Hotspot on the LOQ laptop:

| Device | IP | Role |
|---|---|---|
| **LOQ (Host)** | 192.168.137.1 | Cloud Backend + Hotspot |
| **RPi4** | 192.168.137.100 | Edge Node (Doctor) |
| **IdeaPad** | 192.168.137.137 | Patient Server |

SSH into the RPi4:
```bash
ssh pi@192.168.137.100
# Password: edge12345
```

## 3. System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3 python3-pip python3-venv \
    mosquitto mosquitto-clients \
    i2c-tools libopenblas-dev \
    sqlite3 git curl wget
```

> **Note:** `libatlas-base-dev` is **not available** on Debian 13 Trixie. Use `libopenblas-dev` instead.

Enable I2C and SPI for external sensors:
```bash
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
```

## 4. Mosquitto MQTT Broker

```bash
# Configure for external connections
sudo bash -c 'echo -e "listener 1883\nallow_anonymous true" > /etc/mosquitto/conf.d/external.conf'
sudo systemctl enable mosquitto
sudo systemctl restart mosquitto

# Verify
mosquitto_pub -h 127.0.0.1 -t test -m hello && echo "MQTT OK"
```

## 5. Python Virtual Environment

```bash
python3 -m venv ~/predictive-maintenance/venv
source ~/predictive-maintenance/venv/bin/activate
pip install --upgrade pip

# Core packages
pip install paho-mqtt fastapi 'uvicorn[standard]' sse-starlette aiofiles requests
pip install numpy pandas scikit-learn joblib
pip install ai-edge-litert  # Google LiteRT for Python 3.13 (replaces tflite-runtime)
pip install psutil           # For v2 telemetry plugins
```

> **Note:** `tflite-runtime` is **not available** for Python 3.13. Use `ai-edge-litert` instead. The `edge_inference.py` auto-detects both.

## 6. Deploy Files

From the LOQ laptop:
```bash
python deploy_all.py
```

Or manually, the required files on RPi4 at `~/predictive-maintenance/`:
```
edge/main.py, api_server.py, edge_inference.py, mqtt_handler.py,
      local_store.py, sensor_reader.py, edge_logger.py,
      model_router.py, ensemble_decision.py          ← NEW G3
edge/models/classifier_int8.tflite, classifier_scaler.pkl, iso_forest.pkl
edge/models/tcn_storage_rul_int8.tflite, tcn_storage_scaler.joblib  ← NEW G3
edge/models/fan_bearing_cnn_int8.tflite, fan_bearing_scaler.joblib  ← NEW G3
dashboard/index.html, assets/style.css, assets/app.js
server/training/feature_schema.py, feature_schema_v2.py
configs/edge.env
```

Create `__init__.py` files:
```bash
touch ~/predictive-maintenance/edge/__init__.py
touch ~/predictive-maintenance/server/__init__.py
touch ~/predictive-maintenance/server/training/__init__.py
```

## 7. Systemd Service (Auto-Start on Boot)

```bash
sudo tee /etc/systemd/system/edge-ai.service << 'EOF'
[Unit]
Description=Edge AI Predictive Maintenance Service
After=network.target mosquitto.service
Wants=mosquitto.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/predictive-maintenance
ExecStart=/home/pi/predictive-maintenance/venv/bin/python -m edge.main --broker 127.0.0.1 --port 1883 --dashboard-port 8080
Restart=always
RestartSec=5
Environment=PYTHONPATH=/home/pi/predictive-maintenance
Environment=MQTT_BROKER=127.0.0.1
Environment=MQTT_PORT=1883
Environment=CLOUD_OFFLOAD_URL=http://192.168.137.1:8000/api/offload
Environment=CLOUD_CHAT_URL=http://192.168.137.1:8001/api/chat
Environment=API_KEY=pm-edge-ac0008caeb077588ba481e57e1bde8af

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable edge-ai
sudo systemctl start edge-ai
```

Check status:
```bash
sudo systemctl status edge-ai
sudo journalctl -u edge-ai -f  # Live logs
```

## 8. Sensor Wiring (Phase C — When Sensors Arrive)

| Sensor | Interface | RPi4 Pins |
|---|---|---|
| **DHT22** (Temp/Humidity) | Digital GPIO | VCC→3.3V, GND→GND, DATA→GPIO4, 10kΩ pull-up |
| **ADXL345** (Accelerometer) | I2C | VCC→3.3V, GND→GND, SDA→GPIO2, SCL→GPIO3 |
| **MCP3008+MAX4466** (ADC+Mic) | SPI | VCC→3.3V, GND→GND, CLK→GPIO11, DOUT→GPIO9, DIN→GPIO10, CS→GPIO8 |

Install sensor libraries (after wiring):
```bash
pip install adafruit-circuitpython-dht adafruit-circuitpython-adxl34x adafruit-circuitpython-mcp3xxx adafruit-blinka RPi.GPIO
```

> Without sensors connected, `sensor_reader.py` automatically falls back to simulated values.

## 9. Verifying End-to-End

From the LOQ laptop (or IdeaPad patient):
```bash
# v2 publisher (recommended)
python server/telemetry_agent/mqtt_publisher_v2.py --broker 192.168.137.100 --interval 5

# v1 legacy publisher
python server/telemetry_agent/mqtt_publisher.py --broker 192.168.137.100 --dummy
```

You should see:
1. Publisher printing "Published v2 (XXX B) + v1 (XXX B)" every 5 seconds
2. RPi4 logs showing `[LOCAL] HEALTHY (conf=0.92, TTF=300d)` inference results
3. Dashboard at `http://192.168.137.100:8080/` updating in real-time

## 10. Current RPi4 Environment (2026-09-11)

| Component | Version |
|---|---|
| OS | Debian 13 Trixie (Raspberry Pi OS Lite 64-bit) |
| Kernel | 6.18.34+rpt-rpi-v8 aarch64 |
| Python | 3.13.5 |
| NumPy | 2.5.3 |
| Pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| FastAPI | 0.141.1 |
| AI Edge LiteRT | 2.2.0 |
| Mosquitto | 2.0.21 |
| Storage | 113GB USB (104GB free) |
| RAM | 3.7GB |

## 11. Troubleshooting

| Problem | Solution |
|---|---|
| `tflite-runtime` not found | Use `ai-edge-litert` instead — it's auto-detected by `edge_inference.py` |
| `libatlas-base-dev` not found | Not available on Debian 13 — use `libopenblas-dev` instead |
| MQTT "Connection refused" | `sudo systemctl status mosquitto` — restart if needed |
| Dashboard shows no data | Check that telemetry publisher is running and sending to RPi4 IP |
| Cloud offload fails | Ensure LOQ servers are running: `python server/deep_analysis/inference_server.py` |
| sudo requires terminal | Use `echo 'password' | sudo -S command` when running via SSH scripts |
| sklearn version warning | Harmless — model trained on 1.9.0, loaded on 1.9.1 |
