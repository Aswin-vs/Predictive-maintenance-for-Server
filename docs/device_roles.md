# 🖥️ Device Roles — Corrected Architecture

> **⚠️ IMPORTANT:** The RTX 4060 laptop is **NOT** the server being monitored. A **separate device** is the "patient."

---

## The 3 Devices in the Final System

```
  🏥 PATIENT                    🩺 DOCTOR                     🏨 HOSPITAL
  (Separate Server)             (Raspberry Pi 4)               (Lenovo LOQ RTX 4060)
  ┌───────────────────┐         ┌───────────────────┐          ┌───────────────────┐
  │ The machine whose │ telem.  │ Edge AI Engine    │ offload  │ Cloud Backend     │
  │ hardware we are   │────────▶│ ├ TFLite INT8     │─────────▶│ ├ FP32 Models     │
  │ predicting        │  MQTT   │ ├ TTF Estimator   │ (only    │ ├ LSTM-AE Anomaly │
  │ failures for      │         │ ├ Isolation Forest│  when    │ ├ SHAP Explainer  │
  │                   │         │ └ Confidence      │  unsure) │ ├ Model Training  │
  │ Runs:             │         │   Router          │          │ └ Grafana Dashboard│
  │ ├ Telemetry Agent │         │                   │◀─────────│   (every detail)  │
  │ └ (collects own   │         │ Operator Dashboard│ response │                   │
  │    hardware data) │         │ (professional,    │          │                   │
  │                   │         │  essential info)  │          │                   │
  │ Sensors nearby:   │         │                   │          │                   │
  │ ├ DHT22 (temp)    │         │ WiFi Hotspot for  │          │                   │
  │ ├ ADXL345 (vib.)  │         │ demo access       │          │                   │
  │ └ MAX4466 (sound) │         │                   │          │                   │
  └───────────────────┘         └───────────────────┘          └───────────────────┘
       Hardware TBD                  4GB RAM                    Ryzen 7, 24GB RAM
                                     ARM Cortex-A72             RTX 4060 8GB VRAM
```

**Plus:** Lenovo IdeaPad S145 = 💻 Dev machine (write code only — NOT in the final system)

---

## What Each Device Does

### 🏥 Separate Server (Patient) — Hardware TBD

**This is the machine being monitored for hardware failures.**

| What runs on it | Purpose |
|-----------------|---------|
| Telemetry Agent (Python script) | Collects CPU temp, GPU data, SMART, fan RPM, voltages |
| Nothing else AI-related | It's passive — just sends data |

**Data flows OUT from this device → to RPi4 via MQTT**

External sensors (DHT22, ADXL345, MAX4466) are placed **on or near this device** to capture ambient temperature, vibration, and acoustic signatures. These sensors are wired to the RPi4's GPIO (the RPi4 sits physically near the server).

### 🩺 Raspberry Pi 4 (Doctor / Edge)

**This is the brain — makes fast decisions.**

| What runs on it | Purpose |
|-----------------|---------|
| MQTT Broker (Mosquitto) | Central message hub — all devices connect here |
| Edge Inference Engine | TFLite INT8 classifier + TTF estimator |
| Isolation Forest | PSU/Fan anomaly detection |
| Confidence Router | If confident → local decision. If unsure → offload to LOQ |
| Sensor Collector | Reads DHT22, ADXL345, MAX4466 from GPIO (when available) |
| Operator Dashboard | Custom HTML/CSS/JS — professional, clean, essential info only |
| Chat Widget (Rule-Based) | Simple template-based chat for basic questions when LOQ is unavailable |
| WiFi Hotspot | For demo — guide connects phone/laptop to view dashboard |
| Alert Manager | Sends email/Telegram/buzzer alerts |

### 🏨 Lenovo LOQ RTX 4060 (Hospital / Cloud)

**This handles the heavy lifting — NOT for monitoring, ONLY for computation.**

| What runs on it | Purpose |
|-----------------|---------|
| Deep Analysis API (FastAPI) | Receives uncertain predictions from RPi4, runs full FP32 models |
| LSTM-Autoencoder | Anomaly detection (unsupervised, trained on "healthy" data) |
| SHAP Explainer | Why did the model flag this? Feature importance for operators |
| Model Training | Trains all models on GPU (1D-CNN, LSTM-AE, XGBoost) |
| Grafana + InfluxDB | Detailed engineer dashboard — raw telemetry, model metrics, every detail |
| **Ollama LLM (Llama 3.1 8B)** | **AI chatbot with RAG — answers questions using prediction data + SHAP. ~5GB VRAM.** |
| Chat API (/chat endpoint) | FastAPI endpoint that RPi4 dashboard calls to get LLM responses |
| Model Retraining | Periodically retrains with new data from the server |

### 💻 Lenovo IdeaPad S145 (Dev Machine)

**Write code here → push to GitHub → pull on LOQ + RPi4. Not part of the final system.**

- Ryzen 5, Vega 8 (NO CUDA), 8GB RAM
- Can do ~65% of the project: data prep, feature engineering, API code, dashboard, docs
- CANNOT do: GPU model training, real telemetry collection

---

## What CAN Be Done Now (Without the Separate Server)

Since the server hardware is TBD, here's what's independent of it:

| Task | Depends on server HW? | Can do now? |
|------|----------------------|-------------|
| Download & prep public datasets (Backblaze, NASA, AI4I) | No | ✅ |
| Build TTF labels | No | ✅ |
| Feature engineering pipeline | No | ✅ |
| Write telemetry agent code (with dummy data mode) | No — code is generic | ✅ |
| MQTT pipeline | No | ✅ |
| Train Isolation Forest (CPU) | No | ✅ |
| Edge inference + confidence router code | No | ✅ |
| FastAPI servers (edge + cloud APIs) | No | ✅ |
| Dashboard frontend (HTML/CSS/JS) | No | ✅ |
| RPi4 setup guide | No | ✅ |
| Documentation | No | ✅ |
| GitHub repo setup | No | ✅ |
| **Install telemetry agent on actual server** | **YES** | ❌ Wait |
| **Collect real server telemetry** | **YES** | ❌ Wait |
| **Fine-tune models on real data** | **YES** | ❌ Wait |

> **Bottom line:** ~80% of the codebase is server-hardware-independent. The telemetry agent is written generically (uses psutil, pySMART, nvidia-smi which work on any machine). Only deployment + real data collection needs the actual server.

---

## Network Topology (For Demo)

```
  Separate Server              RPi4                    LOQ RTX 4060
  ┌─────────────┐         ┌─────────────┐          ┌─────────────┐
  │             │ Ethernet │             │ Ethernet │             │
  │             │─────────▶│  MQTT Broker│◀─────────│             │
  │  Telemetry  │  or WiFi │  + Edge AI  │  or WiFi │ Cloud APIs  │
  │  Agent      │          │  + Hotspot  │          │ + Grafana   │
  └─────────────┘          └──────┬──────┘          └─────────────┘
                                  │ WiFi Hotspot
                                  │ (SSID: PredMaint-Edge)
                                  ▼
                           ┌─────────────┐
                           │ Guide's     │
                           │ Phone/Laptop│
                           │ → Dashboard │
                           └─────────────┘
```
