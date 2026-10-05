# 🎬 Demo Day Presentation Runbook & System Operating Procedure

> **Project:** Edge-AI Predictive Maintenance for Server Hardware  
> **Audience:** Final Year Project Evaluation Committee, Faculty Examiners & Industry Evaluators  
> **Topology:** 3-Tier Edge-Cloud Confidence-Aware Architecture (Patient $\to$ RPi4 Doctor $\to$ LOQ Hospital)  
> **Live Mesh Tailnet:** `tail1ab139.ts.net`  

---

## ⏱️ Quick Summary & Presenter Script Flow

| Stage | Duration | Core Action | Visual Target |
| :--- | :--- | :--- | :--- |
| **1. Boot & Mesh Sync** | 2 mins | Bring up RPi4 Broker, LOQ Docker/TimescaleDB, and Patient Telemetry | Green status LEDs / `tailscale status` |
| **2. Operator View** | 3 mins | Open Operator Console; explain SSE stream, 98% Optimal Health, Subsystem cards | `http://rpi4-edge.tail1ab139.ts.net:8080` |
| **3. Live Fault Injection** | 4 mins | Inject bearing vibration / thermal throttle; show INT8 detection & Cloud Offload | Dashboard Alert Log + Cloud Fusion Tab |
| **4. Cloud Deep Analysis** | 3 mins | Showcase ST-GCN graph attention, Weibull survival TTF curve, and SHAP root cause | Grafana `:3000` / `/api/deep_analysis` |
| **5. AI RAG Assistant** | 2 mins | Query Ollama Qwen 3.5 4B on root cause and mitigation runbook | Operator Chat Drawer |

---

## 1. Power-On & Connection Sequence

Execute in this strict chronological sequence to establish the MQTT broker and database before streaming telemetry:

### Step 1.1: Power On RPi4 Gateway (The Doctor / Edge Node)
* Connect USB-C power to Raspberry Pi 4.
* The systemd service `edge-ai.service` starts automatically upon boot (or launch manually if headless debugging):
  ```bash
  ssh pi@rpi4-edge.tail1ab139.ts.net
  sudo systemctl status edge-ai
  ```
* **Verification:** Verify Mosquitto broker is active on port `1883` and API server on port `8080`:
  ```bash
  ss -tulpn | grep -E '1883|8080'
  ```

### Step 1.2: Launch Cloud Backend Services (The Hospital / Lenovo LOQ)
* On Lenovo LOQ (Windows 11 with NVIDIA RTX 4060):
* Double-click `START_ALL.bat` or run:
  ```powershell
  # Starts Docker (TimescaleDB PG16 + Grafana), Inference Server (:8000), Chat API (:8001), and TSDB Writer
  .\start_loq_cloud.bat
  ```
* **Verification:** Verify TimescaleDB Docker container is healthy:
  ```powershell
  docker ps --filter "name=timescaledb"
  curl -s http://localhost:8000/api/health
  ```

### Step 1.3: Start Patient Telemetry Agent (The Monitored Server / IdeaPad)
* On the Patient Machine (`ideapads145.tail1ab139.ts.net`):
* The Windows Scheduled Task `\PredictiveMaintenancePatient` executes automatically at boot.
* To launch or view live telemetry in a visible console:
  ```cmd
  C:\Users\Aswin\Documents\Predictive_Maintenance_Patient\start_patient_ideapad_local.bat
  ```
* **Verification:** Verify live JSON frames streaming over MQTT topic `server/IdeaPad-Patient/telemetry`.

---

## 2. Opening the Live Dashboard

Open Google Chrome or Chromium and navigate to:
```text
http://rpi4-edge.tail1ab139.ts.net:8080/
```
*(Alternative local fallback if off-mesh: `http://192.168.137.100:8080/`)*

### Key Elements to Highlight to Evaluators:
1. **Header Badge:** Point out the green glowing badge: `Live SSE Active` (sub-50ms latency Server-Sent Events).
2. **Server Identity:** Select `IdeaPad-Patient` in the sidebar; show genuine dual-partition storage (`C:\` and `D:\`), per-core CPU loads, and motherboard temps.
3. **Composite Health Gauge:** High-efficiency radial dial showing health index (e.g. `98% - OPTIMAL`).
4. **Hardware Subsystems:** 6 real-time telemetry component cards (CPU, GPU, RAM, Storage, PSU & Fans, Physical Environment).
5. **Physical Hardware Drivers:** Point out physical ADXL345 vibration ($g$ dynamic norm) and INMP441 acoustic RMS streaming from the RPi4 I2C/ALSA pins.

---

## 3. Live Fault Injection Demonstration

Demonstrate the system's sub-second anomaly detection and confidence-aware edge-to-cloud offloading:

### Scenario A: Fan Bearing Vibration & Cavitation Surge
From the LOQ terminal, run the realistic synthetic failure injector:
```powershell
.\venv_loq_py311\Scripts\python.exe run_dummy_simulator.py --anomaly fan_bearing --duration 30
```
* **What Happens:**
  1. Fan RPM drops 35% below PWM target; vibration RMS spikes to $> 0.25g$; acoustic kurtosis $> 4.2$.
  2. Edge model `fan_bearing_cnn_int8.tflite` flags `FAN_FAILURE` with confidence $0.99$.
  3. Dashboard immediately flashes amber/red alert: `WARNING: Fan Bearing Degradation Detected`.
  4. Composite health drops from $98\%$ to $42\%$.

### Scenario B: Thermal Paste Pump-Out / CPU Thermal Throttle
```powershell
.\venv_loq_py311\Scripts\python.exe run_dummy_simulator.py --anomaly thermal_critical --duration 30
```
* **What Happens:**
  1. CPU thermal resistance slope $\Delta T / \Delta P$ climbs sharply $> 0.85^\circ\text{C}/\text{W}$.
  2. Edge classifier detects thermal runaway; edge confidence drops below 0.85 router threshold.
  3. **Confidence Router Offload:** Edge offloads raw window to Cloud Backend (`http://loq.tail1ab139.ts.net:8000/api/deep_analysis`).
  4. Cloud ST-GCN and Weibull survival models calculate remaining useful life ($TTF = 4.2\text{ hours}$) and pinpoint thermal paste dry-out.

---

## 4. Cloud Deep Analysis & AI Diagnostic Assistant

1. **Cloud Fusion View:** Click on the **Cloud Fusion** tab in the dashboard.
   - Show the ST-GCN graph attention heatmap connecting CPU Package Temp to VRM Phase Voltages.
   - Display the adaptive threshold table dynamically recalibrating based on historical 97th-percentile baselines.
2. **AI Maintenance Assistant:**
   - Click the **AI Assistant** drawer icon on the right.
   - Ask: *"What caused the recent health drop on IdeaPad-Patient?"*
   - Ollama (Qwen 3.5 4B) combines the recent prediction telemetry, SHAP root cause vectors, and hardware logs to respond with actionable instructions:
     > *"Diagnosis: Critical Thermal Throttling. PROCHOT active. Root cause is thermal resistance degradation ($\Delta T/\Delta P > 0.85$). Recommended action: Clean heatsink fins and reapply thermal paste."*

---

## 5. One-Line Rollback & Emergency Troubleshooting Commands

If any network hiccup or evaluator question requires restarting a service:

| Issue | Target Machine | One-Line Fix / Status Command |
| :--- | :--- | :--- |
| **Check Tailscale Mesh** | Any | `tailscale status` |
| **Restart Edge Pipeline** | RPi4 | `sudo systemctl restart edge-ai` |
| **Inspect Edge Live Logs** | RPi4 | `sudo journalctl -u edge-ai -f -n 30` |
| **Restart Mosquitto Broker** | RPi4 | `sudo systemctl restart mosquitto` |
| **Restart TimescaleDB** | LOQ | `docker restart timescaledb` |
| **Check Cloud Inference** | LOQ | `curl -s http://localhost:8000/api/health` |
| **Restart Patient Telemetry** | IdeaPad | `schtasks /run /tn "PredictiveMaintenancePatient"` |
| **Stop Simulated Anomaly** | LOQ | `Ctrl+C` in simulator terminal (auto-restores normal profile) |

---

## 6. Examiner Q&A Cheat Sheet

* **Q: Why hybrid Edge-Cloud instead of pure Cloud?**  
  * *Answer:* Bandwidth & Privacy. Streaming 36 high-frequency channels from hundreds of servers overwhelms WAN links. Edge INT8 models evaluate local health in 4.2ms. Only uncertain predictions ($\text{conf} < 0.85$) or anomalies trigger cloud offload.
* **Q: How does the system handle missing sensors on consumer hardware?**  
  * *Answer:* Tiered graceful degradation with explicit $(N, 36)$ binary missingness masks. If a machine lacks ECC RAM or Super I/O, the collector flags `"ecc_supported": false`, marks the mask bit, and the model maintains mathematical invariance without throwing `NoneType` errors.
* **Q: How is Time-to-Failure (TTF) modeled?**  
  * *Answer:* Asymmetric loss penalizing late failure predictions 5x more heavily ($\alpha = 5.0$) than early maintenance warnings ($\alpha = 1.0$), combined with right-censored parametric Weibull Accelerated Failure Time (AFT) hazard estimation.
