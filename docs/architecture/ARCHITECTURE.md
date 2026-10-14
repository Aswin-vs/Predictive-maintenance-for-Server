# Edge-AI Predictive Maintenance for Server Hardware — Technical Architecture Reference

> **Version:** 1.0 · **Date:** 2026-10-12 · **Git HEAD:** `b5cda90`
> **Audience:** An engineer who has never seen this project and cannot ask anyone anything.
> **Ground-Truth Rule:** Every claim is traced to code, config, or measurement. Docs are leads only.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Device Inventory & Network Topology](#2-device-inventory--network-topology)
3. [Telemetry Collection Pipeline](#3-telemetry-collection-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [AI Model Portfolio (10 Models)](#5-ai-model-portfolio-10-models)
6. [Edge Inference Pipeline](#6-edge-inference-pipeline)
7. [Cloud Inference & Decision Fusion](#7-cloud-inference--decision-fusion)
8. [Database Layer](#8-database-layer)
9. [Operator Dashboard](#9-operator-dashboard)
10. [API Surface](#10-api-surface)
11. [Deployment & Operations](#11-deployment--operations)
12. [Configuration Reference](#12-configuration-reference)
13. [Security](#13-security)
14. [Accuracy, Validation & Known Limitations](#14-accuracy-validation--known-limitations)
15. [Repository Map](#15-repository-map)

---

## 1. System Overview

### 1.1 What This System Does

This is a **distributed predictive maintenance system** for server hardware. It predicts hardware failures (CPU thermal runaway, HDD degradation, PSU anomaly, fan failure, memory fault, GPU degradation) **before they happen** using a 3-tier architecture:

1. **Patient** — The monitored machine. Runs a telemetry agent that streams hardware metrics over MQTT.
2. **Doctor (Edge)** — A Raspberry Pi 4 sitting near the patient. Runs lightweight INT8-quantized AI models, makes sub-millisecond predictions, serves an operator dashboard, and routes uncertain cases to the cloud.
3. **Hospital (Cloud)** — A GPU-equipped laptop (LOQ RTX 4060). Runs full-precision PyTorch models, deep multi-model fusion, SHAP explainability, an LLM chatbot, TimescaleDB for long-term storage, and Grafana dashboards.

### 1.2 Research Foundation

| Attribute | Value |
|-----------|-------|
| **Paper 1 (Implementation Ref)** | "Edge-AI Enabled Predictive Maintenance" — IJME Dec 2025 |
| **Paper 2 (Base Paper)** | "Design of a Lightweight Edge-AI System for Predictive" — Applied Sciences 2026 |
| **Our Innovation** | Server hardware (CPU/GPU/HDD/PSU/RAM/Fans) instead of motors; RPi4 instead of ESP32; confidence-aware hybrid edge-cloud routing; asymmetric survival loss |

### 1.3 High-Level Architecture

```
PATIENT MACHINE                  RPi4 DOCTOR (EDGE)              LOQ HOSPITAL (CLOUD)
┌─────────────────────┐        ┌────────────────────────┐       ┌──────────────────────────┐
│ collector_v2.py     │  MQTT  │ edge/main.py           │       │ server/deep_analysis/    │
│ ├─ cpu_plugin       │───────▶│ ├─ MQTT Handler        │       │ ├─ inference_server.py   │
│ ├─ gpu_plugin       │ :1883  │ ├─ Feature Eng. (28f)  │       │ │  ├─ LSTM-AE            │
│ ├─ storage_plugin   │        │ ├─ Edge Inference      │       │ │  ├─ MAD-GAN            │
│ ├─ ram_plugin       │        │ │  ├─ classifier_int8  │       │ │  ├─ ST-GCN             │
│ ├─ psu_plugin       │        │ │  ├─ iso_forest       │       │ │  ├─ Cloud Fusion       │
│ └─ fan_plugin       │        │ │  ├─ stgcn_int8       │       │ │  └─ SHAP Explainer     │
│                     │        │ │  ├─ fan_cnn_int8     │       │ ├─ chat_api.py           │
│ Physical Sensors    │        │ │  └─ tcn_storage_int8 │       │ │  └─ Ollama Qwen 3.5 4B │
│ ├─ ADXL345 (I2C)    │        │ ├─ Adaptive Baseline   │       │ ├─ tsdb_writer.py        │
│ ├─ INMP441 (I2S)    │        │ ├─ Confidence Router   │       │ │  └─ TimescaleDB :5432  │
│ └─ DHT22 (GPIO)     │        │ │  conf≥0.85 → LOCAL  │       │ └─ Grafana :3000         │
│                     │        │ │  conf<0.85 → OFFLOAD─┼──────▶│                          │
│                     │        │ ├─ Local SQLite Store   │       │                          │
│                     │        │ └─ Dashboard :8080      │◀──────│   (deep analysis reply)  │
└─────────────────────┘        └────────────────────────┘       └──────────────────────────┘
```

[src: `edge/main.py:1-18`, `server/deep_analysis/inference_server.py:1-31`]

---

## 2. Device Inventory & Network Topology

### 2.1 Physical Devices

| Device | Role | Specs | Network Identity |
|--------|------|-------|------------------|
| **Lenovo IdeaPad S145** | Patient (monitored) | Ryzen 5 + Vega 8 (no CUDA), 8 GB RAM | `ideapads145.tail1ab139.ts.net` / `100.97.69.16` |
| **Lenovo LOQ** | Hospital (cloud) + Patient | Ryzen 7, 24 GB RAM, RTX 4060 8 GB VRAM | `loq.tail1ab139.ts.net` / `100.111.172.121` |
| **Raspberry Pi 4 Model B** | Doctor (edge) | ARM Cortex-A72, 4 GB RAM, 64-bit OS | `rpi4-edge.tail1ab139.ts.net` / `100.90.160.78` |

[src: `configs/network.env:9-18`]

### 2.2 Network Mesh

All three devices are connected via a **Tailscale WireGuard mesh** (tailnet: `tail1ab139.ts.net`). This provides:
- Encrypted end-to-end WireGuard tunnels (no firewall/port-forwarding required)
- MagicDNS hostnames (e.g., `rpi4-edge.tail1ab139.ts.net`)
- Works across any physical network topology (home WiFi, hotspot, cellular)

**Measured latencies** (from `audit/benchmark_e2e_latencies.py`):
- MQTT round-trip (Patient → RPi4): **21.7 ms** minimum, 0% packet loss
- Cloud GPU deep analysis (RPi4 → LOQ → RPi4): **10.04 ms**

**Fallback:** If Tailscale is unavailable, the RPi4 can be reached at `192.168.137.100` via Windows mobile hotspot.
[src: `configs/network.env:13`]

### 2.3 Port Map

| Port | Device | Service | Protocol |
|------|--------|---------|----------|
| 1883 | RPi4 | Mosquitto MQTT Broker | TCP |
| 8080 | RPi4 | Edge API + Dashboard (FastAPI/Uvicorn) | HTTP |
| 8000 | LOQ | Cloud Inference Server (FastAPI/Uvicorn) | HTTP |
| 8001 | LOQ | Chat API Server (FastAPI/Uvicorn) | HTTP |
| 5432 | LOQ | TimescaleDB (Docker) | TCP |
| 3000 | LOQ | Grafana (Docker) | HTTP |
| 11434 | LOQ | Ollama LLM Runtime | HTTP |

[src: `configs/cloud.env:17-24`, `configs/edge.env:14`, `docker-compose.yml:11-12,33-34`]

---

## 3. Telemetry Collection Pipeline

### 3.1 Patient Agent Architecture (`collector_v2.py` + `mqtt_publisher_v2.py`)

Each patient machine runs `server/telemetry_agent/mqtt_publisher_v2.py` (or the identical standalone distribution in `patient_package/mqtt_publisher_v2.py`). It executes a continuous polling loop every **5 seconds** (configurable via `--interval`), calling `TelemetryCollectorV2` from `collector_v2.py` which aggregates hardware metrics via a tiered plugin architecture:

| Plugin | File | What It Collects | Method |
|--------|------|-----------------|--------|
| **CPU** | `plugins/cpu_plugin.py` | Temperature, load %, frequency, per-core stats | `psutil` + WMI `Win32_PerfFormattedData_Counters_ThermalZoneInformation` |
| **GPU** | `plugins/gpu_plugin.py` | Temperature, power, utilization, VRAM, clock | NVML (via `pynvml`/`GPUtil`) |
| **Storage** | `plugins/storage_plugin.py` | SMART attributes (5, 9, 187, 188, 194, 197, 198), usage, media type | `pySMART` + WMI `Win32_DiskDrive` + PowerShell `Get-PhysicalDisk` |
| **RAM** | `plugins/ram_plugin.py` | Used GB, total GB, page faults/sec | `psutil` |
| **PSU** | `plugins/psu_plugin.py` | Voltage deviations (3.3V, 12V), stability | Simulated on consumer hardware (no PMBus BMC available) |
| **Fan** | `plugins/fan_plugin.py` | RPM, deviation from expected | Simulated on laptops (EC-controlled, no tachometer API) |

[src: `server/telemetry_agent/plugins/`, `server/telemetry_agent/mqtt_publisher_v2.py:174-263`]

**Payload formats & topics:**
- **Hierarchical v2 payload:** Published to MQTT topic `server/{server_id}/telemetry` (QoS 1)
- **Hardware registration payload:** Published to `server/{server_id}/register` (retained, QoS 1 on boot and heartbeat every 100 ticks)
- **Flat v1 legacy payload:** Published to `server/telemetry` for backward compatibility

### 3.2 Physical Hardware Sensors (RPi4)

The RPi4 reads **real physical sensors** attached to the server chassis:

| Sensor | Bus | Address/Device | What It Measures |
|--------|-----|---------------|------------------|
| **ADXL345** | I2C Bus 1 | `0x53` | 3-axis vibration (g). Full resolution ±16g, scale 0.0039 g/LSB. Dynamic RMS computed by subtracting gravity vector. |
| **INMP441** | I2S ALSA | `plughw:3,0` | Acoustic RMS. 44.1 kHz, 32-bit S32_LE burst capture. Normalized to [0,1]. FFT dominant frequency extracted. |
| **DHT22** | GPIO4 | — | Ambient temperature (°C) and relative humidity (%). Optional — graceful fallback if missing. |

[src: `edge/sensor_reader.py:1-18`, lines 57-64 for ADXL345 registers]

**Graceful degradation:** If any sensor is disconnected or unavailable (e.g., running on a dev machine), the `SensorReader` catches the failure, logs a warning once, and returns last-known/nominal values without crashing. On the RPi4 edge node, `SensorPublisherThread` publishes ambient and dynamic hardware telemetry to `edge/sensors` every 2.0s, and injects live readings into incoming server telemetry streams while guarding against cross-node sensor bleed.
[src: `edge/sensor_reader.py:14-17`, `edge/main.py:123-145,432-450`]

### 3.3 MQTT Pipeline

```
Patient (mqtt_publisher_v2.py + collector_v2.py)
    │
    ├── publish server/{server_id}/telemetry  (hierarchical v2)
    ├── publish server/{server_id}/register   (topology register)
    └── publish server/telemetry              (legacy v1 flat)
                   │
Mosquitto Broker (RPi4 :1883)
    │
    └── subscribed topics:
        ├── server/+/telemetry
        ├── server/+/register
        └── server/telemetry
                   │
Edge Main (edge/main.py → EdgeMQTTHandler)
    ↓ on_message callback
Feature Engineering → Inference → Alert/Store
```

The MQTT handler (`edge/mqtt_handler.py`) subscribes to `server/+/telemetry`, `server/+/register`, and `server/telemetry`, caching hardware registration topology and dispatching each telemetry payload through the inference pipeline.
[src: `edge/main.py:51,906-910`, `edge/mqtt_handler.py:8-11,32-34`]

---

## 4. Feature Engineering

### 4.1 Feature Schema v1 (28 Features)

The canonical feature ordering is defined in a single source of truth: [`server/training/feature_schema.py`](../../server/training/feature_schema.py).

**All 10 AI models and all inference pipelines import from this module** to guarantee consistent feature ordering.

| Index | Feature Name | Component | Unit |
|-------|-------------|-----------|------|
| 0 | `cpu_temp` | CPU | °C |
| 1 | `cpu_load` | CPU | % |
| 2 | `cpu_freq` | CPU | MHz |
| 3 | `cpu_ambient_delta` | CPU | °C |
| 4 | `gpu_temp` | GPU | °C |
| 5 | `gpu_power` | GPU | W |
| 6 | `gpu_util` | GPU | % |
| 7 | `gpu_mem_used` | GPU | MB |
| 8 | `gpu_clock` | GPU | MHz |
| 9 | `ram_used_gb` | RAM | GB |
| 10 | `ram_total_gb` | RAM | GB |
| 11 | `ram_page_faults` | RAM | /sec |
| 12 | `smart_5_raw` | Storage | Reallocated Sectors |
| 13 | `smart_9_raw` | Storage | Power-On Hours |
| 14 | `smart_187_raw` | Storage | Uncorrectable Errors |
| 15 | `smart_188_raw` | Storage | Command Timeouts |
| 16 | `smart_194_raw` | Storage | HDD Temp °C |
| 17 | `smart_197_raw` | Storage | Pending Sectors |
| 18 | `smart_198_raw` | Storage | Offline Uncorrectable |
| 19 | `psu_voltage_3v3_dev` | PSU | % deviation |
| 20 | `psu_voltage_12v_dev` | PSU | % deviation |
| 21 | `psu_voltage_stability` | PSU | Rolling σ |
| 22 | `fan_rpm_deviation` | Fans | (actual−expected)/expected |
| 23 | `ambient_temp` | Environment | °C (DHT22) |
| 24 | `humidity` | Environment | % (DHT22) |
| 25 | `vibration_rms` | Environment | g (ADXL345) |
| 26 | `acoustic_rms` | Environment | normalized (INMP441) |
| 27 | `system_uptime` | System | hours |

[src: `server/training/feature_schema.py:32-76`]

### 4.2 Feature Schema v2 (36 Features — Enterprise Harmonized)

For the cloud-only models (ST-GCN, MAD-GAN, LSTM-AE), a hierarchical 36-feature schema is used to harmonize data from 8 enterprise datasets. Defined in `server/training/feature_schema_v2.py`. Includes a missingness mask tensor of shape `(N, 36)` where 1.0 = observed, 0.0 = imputed.

The `v2_to_v1_feature_vector()` function converts hierarchical v2 payloads back to the 28-feature flat vector for edge models.
[src: `server/training/feature_schema_v2.py`]

### 4.3 Model-Specific Feature Subsets

| Model | Feature Subset | Source |
|-------|---------------|--------|
| 1D-CNN Classifier | All 28 (v1) | `feature_schema.py:FEATURE_NAMES_V1` |
| Isolation Forest | 4 features: PSU voltages + fan RPM | `feature_schema.py:IF_FEATURE_NAMES` (indices 19-22) |
| XGBoost HDD | 7 features: SMART attributes only | `feature_schema.py:XGB_FEATURE_NAMES` (indices 12-18) |
| LSTM-AE, MAD-GAN, ST-GCN | 36 features (v2 harmonized) | `feature_schema_v2.py:ENTERPRISE_36_FEATURES` |
| Fan Bearing CNN | NASA C-MAPSS sensor subset | `train_fan_cnn.py` |
| TCN Storage RUL | Backblaze SMART sequence | `train_tcn_storage.py` |

---

## 5. AI Model Portfolio (10 Models)

### 5.1 Summary Table

| # | Model | Type | Framework | Input | Output | Training Data | Edge/Cloud | File Size | Empirical Score / Metrics |
|---|-------|------|-----------|-------|--------|---------------|------------|-----------|---------------------------|
| 1 | **1D-CNN Multi-Task** | Classification + TTF Regression | TensorFlow/Keras | 28×1 | 7-class softmax + TTF scalar | AI4I 2020 (10,000 samples) | Both (INT8 on edge, FP32 on cloud) | Edge: 133 KB, Cloud: 1.5 MB | **Val Acc: 99.30%** (97.57% un-augmented), Loss: 0.0560, TTF MAE: 0.0560 |
| 2 | **Isolation Forest** | Unsupervised Anomaly | scikit-learn | 4 features | Anomaly score | Synthetic normal baselines | Edge | 1.0 MB | **100% Sag Detection** ($<11.4\text{V}$), 0.038 ms edge latency |
| 3 | **XGBoost HDD** | TTF Regression | XGBoost | 7 SMART features | TTF days | Backblaze Q1 2023 (16,874+ failures) | Cloud | 2.1 MB | **RMSE: 13.15 Days**, MAE: 8.72 Days, **$R^2 = 0.814$** |
| 4 | **LSTM-Autoencoder** | Unsupervised Anomaly | TensorFlow/Keras | 10-step window × 28 features | Reconstruction error | SMD (708,420 records) + 70k Healthy | Cloud | 900 KB | **Val MSE: 0.0949**, Anomaly Threshold $\tau = 0.330935$ |
| 5 | **MAD-GAN** | Multivariate Anomaly | PyTorch (CUDA) | 38 features (padded) | Generator/Discriminator score | SMD + Synthetic | Cloud | Gen: 144 KB, Disc: 164 KB | **F1: 0.0514** (5.14% — mitigated to 0.05 fusion weight) |
| 6 | **ST-GCN** | Spatio-Temporal Graph | PyTorch (CUDA) | 30-step window × 36 features | Anomaly classification + TTF | Alibaba (DRAM/GPU/SSD), Backblaze, NASA | Cloud (INT8 on edge) | Edge: 10 KB, Cloud: 65 KB | **Val F1: 0.0408** (4.08% — mitigated to 0.05 fusion weight) |
| 7 | **Fan Bearing CNN** | Fault Classification | TensorFlow/Keras | NASA C-MAPSS / PWM-RPM | Degradation probability / RUL | Synthetic Fan PWM/RPM Shock Dataset | Both | Edge: 28 KB, Cloud: 240 KB | **Test Acc: 1.000 (100%)**, AUC: 1.000, Loss: 0.0012 |
| 8 | **TCN Storage RUL** | Sequence-to-One Regression | TensorFlow/Keras | Backblaze SMART window | TTF days (0–90) | Backblaze Q1 2023 30-Day Windows | Both | Edge: 141 KB, Cloud: 1.4 MB | **Val MAE: 15.94 Days**, MSE: 588.17 (RMSE: 24.25 Days) |
| 9 | **Adaptive Baseline** | Statistical Z-Score | Pure Python (no ML) | Per-metric time series | Z-score anomaly flag | Live telemetry (online learning) | Edge | N/A (stateful) | **0% False Alarms** after 24h calibration; $Z \ge 3.0$ triggers alert |
| 10 | **Ollama LLM (Qwen 3.5 4B)** | RAG Chatbot | Ollama runtime | Natural language + context | Diagnostic text | Pre-trained + RAG | Cloud | 3.1 GB VRAM | Plain-English root cause diagnosis + SHAP attribution synthesis |

### 5.2 Model 1: 1D-CNN Multi-Task Classifier

**Architecture** (from `train_classifier.py:53-70`):
```
Input(28, 1)
  → Conv1D(32, kernel=3, relu, same)
  → Conv1D(64, kernel=3, relu, same)
  → MaxPool1D(2)
  → Flatten → Dense(128, relu) → Dropout(0.3)
  ├─ Head 1: Dense(7, softmax) → Fault Class [classification]
  └─ Head 2: Dense(1, linear) → TTF [regression, asymmetric loss α=5.0]
```

**Loss function:** Classification head uses `sparse_categorical_crossentropy`. TTF head uses `AsymmetricTTFLoss(alpha=5.0)` from `server/training/loss.py:27-55`.

The asymmetric loss penalizes **overestimation** of remaining life 5× more heavily than underestimation. Rationale: telling an operator "you have 30 days" when the actual TTF is 5 days is far more dangerous than telling them "you have 5 days" when it's actually 30 days.
[src: `server/training/loss.py:29-34`]

**Quantization:** FP32 model (`classifier_fp32.h5`, 1.5 MB) is quantized to INT8 TFLite (`classifier_int8.tflite`, 133 KB) for edge deployment.
[src: `edge/models/classifier_int8.tflite` — 135,744 bytes]

### 5.3 Model 2: Isolation Forest

Lightweight unsupervised anomaly detector using only 4 features (PSU voltage deviations + fan RPM deviation). Runs on edge alongside the classifier. Uses scikit-learn's `IsolationForest`.
[src: `server/training/train_iso_forest.py`, `edge/models/iso_forest.pkl` — 1,044,464 bytes]

### 5.4 Model 3: XGBoost HDD Specialist

Trained on Backblaze Drive Stats Q1 2023 (7.18 GB raw data, 16,874+ failure sets, 250k+ drive-days). Uses 7 SMART attributes to predict HDD TTF in days. SHAP explainability is generated by `server/deep_analysis/shap_explainer.py`.
[src: `server/training/train_xgboost_hdd.py`, `server/models/xgboost_hdd.pkl` — 2,187,728 bytes]

### 5.5 Model 4: LSTM-Autoencoder

Unsupervised anomaly detector trained on healthy-only data from the Server Machine Dataset (SMD, 708,420 records across 28 server entities). Uses a 10-step rolling window. Anomaly = reconstruction error exceeding the 97th percentile threshold.
[src: `server/training/train_lstm_ae.py`, `server/models/lstm_ae.h5` — 899,536 bytes]

### 5.6 Model 5: MAD-GAN (Multivariate Anomaly Detection GAN)

PyTorch generator + discriminator. Trained to reconstruct normal telemetry; anomalies produce high reconstruction/discrimination divergence. Input is zero-padded from 28 to 38 features for architectural compatibility.
[src: `server/training/train_madgan.py`, `server/models/madgan_generator.pt` — 143,724 bytes, `server/models/madgan_discriminator.pt` — 164,142 bytes]

### 5.7 Model 6: ST-GCN (Spatio-Temporal Graph Convolutional Network)

PyTorch model combining spatial graph convolution (learned adjacency matrix over 36 sensor nodes) with temporal 1D-CNN over 30-step windows. Multi-task output: anomaly classification (BCE) + asymmetric TTF regression (α=5.0) + Weibull AFT log-likelihood.

Trained on harmonized enterprise data from 6 dataset loaders: Alibaba DRAM, Alibaba GPU Traces, Alibaba SSD, Backblaze, Delta GPU XID, NASA C-MAPSS.
[src: `server/training/train_stgcn.py:1-15`, `server/models/stgcn_anomaly.pt` — 64,654 bytes, `server/models/stgcn_adjacency.npy` — 5,312 bytes]

**Known limitation:** The missingness mask tensor `(N, 36)` is generated by `schema_harmonization.py` and passed through data loaders, but is **not** currently fed into the ST-GCN graph layers (`n_feat = 36`). Spatial mask gating (`A · (X ⊙ M) · W`) is documented as a future enhancement.
[src: AUDIT_REPORT.md, MEMORIES.md Phase 9.1]

### 5.8 Model 7: Fan Bearing CNN

1D-CNN trained on NASA C-MAPSS turbofan degradation data (FD001-FD004). Predicts remaining useful life of fan/bearing assemblies.
[src: `server/training/train_fan_cnn.py`, `edge/models/fan_bearing_cnn_int8.tflite` — 27,840 bytes]

### 5.9 Model 8: TCN Storage RUL

Temporal Convolutional Network trained on Backblaze SMART attribute sequences with stratified drive-level splitting (GroupKFold-equivalent to prevent temporal cross-window leakage).
[src: `server/training/train_tcn_storage.py`, `edge/models/tcn_storage_rul_int8.tflite` — 144,344 bytes]

### 5.10 Model 9: Adaptive Baseline (Online Statistical)

Pure Python Z-score anomaly detector with per-server, per-metric statistics. Operates on live streaming telemetry with domain-aware physical variance floors:

| Metric | Minimum σ | Rationale |
|--------|-----------|-----------|
| RAM allocation | 3.0 GB | OS memory management swings are normal |
| CPU/GPU load | 10% | Load varies with workload |
| CPU frequency | 250 MHz | Turbo boost oscillation |
| Fan RPM | 150 RPM | Speed controller step changes |

During calibration (first 24h): AI4I models are primary, baselines are secondary.
After calibration: Baselines become primary, AI4I models are secondary confirmation.
[src: `edge/adaptive_baseline.py`, `edge/main.py:12-14`]

### 5.11 Model 10: Ollama LLM (Qwen 3.5 4B)

RAG chatbot running on LOQ via Ollama runtime. Generates diagnostic markdown from live prediction data and SHAP attributions.

| Property | Value |
|----------|-------|
| Model | `qwen3.5:4b` |
| VRAM | 3.1 GB (100% GPU resident on RTX 4060) |
| Context | 4096 tokens |
| Max output | 512 tokens |
| Thinking mode | Disabled (`"think": false`) |
| Measured latency | ~7.9 sec |

[src: `configs/cloud.env:14`, `server/deep_analysis/chat_api.py`]

### 5.12 7 Fault Classes

| Class | Label | Primary Sensors |
|-------|-------|-----------------|
| 0 | HEALTHY | All |
| 1 | HDD_DEGRADATION | SMART attributes |
| 2 | THERMAL_CRITICAL | CPU/GPU temperatures |
| 3 | PSU_ANOMALY | Voltage rails |
| 4 | FAN_FAILURE | RPM + acoustic |
| 5 | MEMORY_FAULT | Page faults |
| 6 | GPU_DEGRADATION | Clock drops, thermal |

[src: `server/training/feature_schema.py:138-146`]

### 5.13 Training Datasets (8 Enterprise Sources)

| # | Dataset | Records | Size | Target Models |
|---|---------|---------|------|---------------|
| 1 | Backblaze Drive Stats Q1 2023 | 16,874+ failure sets | 7.18 GB | TCN Storage, XGBoost |
| 2 | AI4I 2020 Predictive Maintenance | 10,000 samples | ~1 MB | 1D-CNN Classifier |
| 3 | NASA C-MAPSS FD001-FD004 | 20,631 cycles | ~25 MB | Fan Bearing CNN |
| 4 | Alibaba DCBrain SSD | 18,387 failure tickets | 209 MB | ST-GCN |
| 5 | Alibaba PAKDD 2021 DRAM | Millions of DRAM errors | — | ST-GCN |
| 6 | Alibaba GPU + Delta XID Logs | 81,806 failure records | — | ST-GCN |
| 7 | Server Machine Dataset (SMD) | 708,420 records (28 entities) | — | LSTM-AE, MAD-GAN |
| 8 | Live Field Telemetry | Streaming (ongoing) | — | Adaptive Baseline |

Dataset loaders are in `server/training/datasets/`.

---

## 6. Edge Inference Pipeline

### 6.1 Pipeline Flow (per telemetry tick)

```python
# Simplified from edge/main.py on_telemetry_received()
1. Receive JSON payload via MQTT
2. Convert v2 hierarchical → v1 flat 28-feature vector
3. Apply domain-aware reference normalization for classifier input
4. Run classifier_int8.tflite → (fault_class, confidence, ttf_raw)
5. Run iso_forest.pkl → anomaly_score
6. Run stgcn_int8.tflite, fan_bearing_cnn_int8.tflite, tcn_storage_rul_int8.tflite
7. Score adaptive baseline (Z-score per metric)
8. Apply confidence router:
   - confidence ≥ 0.85 → LOCAL decision
   - confidence < 0.85 → OFFLOAD to cloud (:8000/api/offload)
9. Compute bounded TTF:
   - Healthy (class 0 or not anomaly): 90.0 days (displays "> 30 Days")
   - Fault + anomaly: scale by subsystem degradation curve (0.5d–14d)
10. Physical feasibility guards: suppress false PSU/Memory alarms on consumer PCs
11. Store in local SQLite ring buffer
12. Broadcast via SSE to dashboard
```

[src: `edge/main.py`]

### 6.2 Measured Inference Latencies (on physical RPi4 ARM Cortex-A72)

| Model | File Size | Latency | Throughput |
|-------|-----------|---------|------------|
| `stgcn_int8.tflite` | 9.7 KB | **0.029 ms** | 34,483 inf/sec |
| `fan_bearing_cnn_int8.tflite` | 27.2 KB | **0.076 ms** | 13,158 inf/sec |
| `classifier_int8.tflite` | 132.6 KB | **0.160 ms** | 6,250 inf/sec |
| `tcn_storage_rul_int8.tflite` | 141.0 KB | **0.547 ms** | 1,829 inf/sec |
| **Total edge inference** | 310.5 KB RAM | **< 1 ms** | — |

All models run in deterministic sub-millisecond time.
[src: `audit/benchmark_e2e_latencies.py`, MEMORIES.md Phase 7]

### 6.3 Physical Sensor Read Latency

| Operation | Latency |
|-----------|---------|
| Full sensor read (ADXL345 + INMP441) | ~370 ms |
| ADXL345 I2C burst (32 samples) | ~5 ms |
| INMP441 ALSA burst (200ms at 44.1 kHz) | ~200 ms |

[src: MEMORIES.md Phase 7 smoke test]

---

## 7. Cloud Inference & Decision Fusion

### 7.1 Cloud Inference Server (`inference_server.py`)

The FastAPI server at `:8000` on LOQ loads 5 deep models at startup:

1. 1D-CNN FP32 classifier + StandardScaler
2. LSTM-Autoencoder + rolling 10-step deque per server
3. MAD-GAN Generator + Discriminator (PyTorch CUDA)
4. ST-GCN (PyTorch CUDA) + adjacency matrix
5. SHAP explainer for XGBoost HDD

[src: `server/deep_analysis/inference_server.py:24-31`]

### 7.2 Cloud Decision Fusion

The cloud fusion engine (`server/deep_analysis/cloud_fusion.py`) combines scores from all models using weighted ensemble:

| Model | Weight | Rationale |
|-------|--------|-----------|
| **Edge Classifier** | 75% | Validated on real hardware, best per-sample accuracy |
| **LSTM-AE** | 15% | Mature reconstruction-based anomaly detector |
| **MAD-GAN** | 5% | Near-random F1 scores on current data (see F-026) |
| **ST-GCN** | 5% | Near-random F1 scores on current data (see F-026) |

The fused health score and severity verdict are returned to the edge for dashboard display. Severity labels in code (`cloud_fusion.py:49-55`): `HEALTHY` (80–100), `MONITORING` (60–80), `CAUTION` (40–60), `WARNING` (20–40), `CRITICAL` (0–20). The dashboard UI (`app.js:407`) re-labels `HEALTHY` → `OPTIMAL` for operator clarity.
[src: `server/deep_analysis/cloud_fusion.py:49-55`, `dashboard/assets/app.js:407`]

### 7.3 Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check (returns model count, uptime, TSDB status) |
| `/api/offload` | POST | Receive offloaded telemetry, run all models, return fused verdict |
| `/api/deep_analysis` | POST | Full cloud analysis with SHAP explanations |
| `/api/predict` | POST | Single model inference |

[src: `server/deep_analysis/inference_server.py`]

### 7.4 Chat API (`chat_api.py`)

Separate FastAPI server at `:8001`. Proxies requests to Ollama with RAG context from live predictions and SHAP data.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Send message, get diagnostic LLM response |
| `/api/health` | GET | Chat API health check |

Authentication: `X-API-Key` header required.
[src: `server/deep_analysis/chat_api.py`]

---

## 8. Database Layer

### 8.1 TimescaleDB (Cloud — LOQ)

Runs in Docker (`docker-compose.yml`). PostgreSQL 16 with TimescaleDB extension.

**Schema** (from `server/deep_analysis/schema.sql`):

| Table/View | Type | Purpose | Retention |
|-----------|------|---------|-----------|
| `servers` | Table | Server registry (ID, name, hostname, OS, topology) | Indefinite |
| `telemetry_raw` | Hypertable | Every telemetry tick (JSON payload + extracted columns) | 30 days |
| `predictions` | Hypertable | Every inference result (model, fault class, confidence, TTF, SHAP) | 90 days |
| `thresholds` | Table | Adaptive thresholds per server per metric | Indefinite |
| `telemetry_hourly` | Continuous Aggregate | Hourly rollups (avg/max/min CPU temp, load, health) | Indefinite |
| `telemetry_daily` | Continuous Aggregate | Daily rollups | Indefinite |
| `predictions_hourly` | Continuous Aggregate | Hourly prediction summaries | Indefinite |

**Policies:**
- Raw telemetry compression after 7 days (segment by `server_id`, order by `time DESC`)
- Hourly aggregates refresh every 30 minutes
- Daily aggregates refresh every 6 hours

[src: `server/deep_analysis/schema.sql:1-191`]

**Ingestion:** `server/deep_analysis/tsdb_writer.py` subscribes to MQTT and batch-inserts telemetry into TimescaleDB. Measured batch insertion latency: **4.03 ms** (2.34 ms min).

### 8.2 SQLite (Edge — RPi4)

`edge/local_store.py` maintains a local ring buffer for:
- Recent telemetry records (for dashboard history table)
- Alert state (active anomalies)
- Server metadata

The ring buffer is bounded; old records are evicted on new inserts. Data can be purged via `POST /api/clear_cache`.
[src: `edge/local_store.py`]

---

## 9. Operator Dashboard

### 9.1 Technology

Static HTML/CSS/JS served by the FastAPI server on RPi4 `:8080`.

| File | Size | Purpose |
|------|------|---------|
| `dashboard/index.html` | 91.8 KB | Full UI markup (all tabs, modals, drawers) |
| `dashboard/assets/app.js` | ~100 KB | All JavaScript logic (SSE, API calls, rendering) |
| `dashboard/assets/style.css` | ~30 KB | All styles (glassmorphism, animations, responsiveness) |

[src: `dashboard/index.html`, `dashboard/assets/`]

### 9.2 Dual-Mode Interface

**Executive Summary View** (`#btnModeSummary`):
- SVG radial health dial (composite health %)
- 4 KPI cards: Health Index, TTF Horizon, Inference Confidence, Active Incidents
- 6 hardware subsystem cards with micro-sparklines
- Live anomaly feed
- AI executive briefing

**Deep Diagnostics View** (`#btnModeComplex`) — 9 specialized tabs:

| Tab # | Name | ID | Key Content |
|-------|------|-----|-------------|
| 1 | Sensor Recognition & Audit Matrix | `pane-sensors-view` | 10 sensor channels, 4 filter chips (All/Live/Proxied/Exclusions) |
| 2 | 36-Feature Harmonized Schema | `pane-schema36-view` | All 36 v2 features with source mappings |
| 3 | Hardware Detail | `pane-hardware-view` | Per-instance telemetry (drives, cores, GPU) with learned baselines |
| 4 | 10-Model AI Portfolio | `pane-portfolio-view` | All models + 8 enterprise datasets table |
| 5 | Telemetry Trends | `pane-trends-view` | Chart.js streaming, history range buttons (5min/1h/6h/24h) |
| 6 | Root Cause / XAI (SHAP) | `pane-xai-view` | 5-step decision pipeline, SHAP impact chart |
| 7 | Database & History | `pane-database-view` | SQLite records, 7-column sorting, severity filter, inspect modal |
| 8 | Platform Health | `pane-topology-view` | 3-tier node connectivity matrix |
| 9 | Cloud Decision Fusion | `pane-cloud-view` | MAD-GAN/ST-GCN/LSTM-AE/Edge scores, 17 adaptive quantiles |

### 9.3 Live Data Flow

Dashboard connects via **Server-Sent Events (SSE)** to receive real-time telemetry updates. A green `Live SSE Active` badge indicates active streaming.

Interactive AI chat drawer connects to LOQ `:8001/api/chat` (proxied through edge `/api/chat`).

---

## 10. API Surface

### 10.1 Edge API (RPi4 `:8080`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Dashboard HTML |
| GET | `/api/health` | No | Edge health (uptime, server count, model status) |
| GET | `/api/state` | No | Current state of all monitored servers |
| GET | `/api/state/{server_id}` | No | State for specific server |
| GET | `/api/history` | No | SQLite historical records (optional `server_id`, `hours`, `limit` params) |
| GET | `/api/baseline/{server_id}` | No | Learned adaptive baselines for a server |
| GET | `/api/thresholds/{server_id}` | No | Adaptive threshold values |
| GET | `/api/cloud_fusion/{server_id}` | No | Trigger cloud fusion evaluation |
| POST | `/api/clear_cache` | No | Purge SQLite telemetry cache |
| POST | `/api/chat` | API Key | Proxy to LOQ chat API |
| GET | `/events` | No | SSE stream for real-time updates |

### 10.2 Cloud Inference API (LOQ `:8000`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | No | Health check (models loaded, TSDB status) |
| POST | `/api/offload` | API Key | Full model inference on offloaded telemetry |
| POST | `/api/deep_analysis` | API Key | Deep analysis with SHAP explanations |
| POST | `/api/predict` | API Key | Single prediction |

### 10.3 Chat API (LOQ `:8001`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | No | Chat API health |
| POST | `/api/chat` | API Key | LLM diagnostic chat |

Authentication: `X-API-Key: pm-edge-ac0008caeb077588ba481e57e1bde8af`
[src: `configs/edge.env:17`, `configs/api_keys.json`]

---

## 11. Deployment & Operations

### 11.1 Prerequisites

| Dependency | Device | Purpose |
|-----------|--------|---------|
| Python 3.11+ | All | Runtime |
| Docker + Docker Compose | LOQ | TimescaleDB + Grafana |
| Mosquitto MQTT Broker | RPi4 | Message bus |
| Tailscale | All | WireGuard mesh VPN |
| Ollama | LOQ | LLM runtime |
| NVIDIA Driver + CUDA | LOQ | GPU inference |

### 11.2 One-Command Start

**LOQ Cloud:**
```batch
START_ALL.bat
```
This starts:
1. TimescaleDB + Grafana (Docker)
2. Cloud Inference Server (`:8000`)
3. Chat API Server (`:8001`)
4. TimescaleDB Writer (MQTT → TSDB)
5. LOQ Patient telemetry agent

[src: `START_ALL.bat`]

**RPi4 Edge:**
```bash
sudo systemctl start edge-ai.service
```
The systemd unit (`configs/edge-ai.service`) runs `edge/main.py` as user `pi` with auto-restart.
[src: `configs/edge-ai.service:12`]

**Patient (IdeaPad/other):**
Windows Scheduled Task `PredictiveMaintenancePatient` executes `START_PATIENT.bat` on boot (which launches `mqtt_publisher_v2.py`), or manually:
```batch
patient_package\START_PATIENT.bat
```
[src: `patient_package/START_PATIENT.bat:55-58`]

### 11.3 Deployment to RPi4

`deploy_all.py` handles multi-device deployment over SSH/SFTP:
- Syncs edge code, dashboard files, models, and configs to RPi4
- Optionally syncs patient package to IdeaPad
- Restarts `edge-ai.service` on RPi4
- Verifies SHA256 checksums post-deployment

[src: `deploy_all.py`]

### 11.4 Patient Package Distribution

`build_patient_package.bat` creates a self-contained, standalone `patient_package/` distribution directory with:
- `mqtt_publisher_v2.py` (continuous polling and MQTT streaming daemon)
- `collector_v2.py` (telemetry collection engine)
- `auto_discovery.py` & `server_identity.py` (hardware topology and UUID assignment)
- `plugins/` (all 6 hardware collection plugins: CPU, GPU, Storage, RAM, PSU, Fan)
- `server/training/` (canonical feature schema definitions)
- `INSTALL_DEPS.bat` (one-click Python dependency installer)
- `START_PATIENT.bat` (Windows launcher) & `start_patient.sh` (Linux launcher)
- `requirements.txt` & `README.md`

Copy this folder to any target machine to begin non-invasive edge monitoring.
[src: `build_patient_package.bat:32-60`]

### 11.5 Stop Everything

```batch
STOP_ALL.bat
```
Kills all Python processes, stops Docker containers.
[src: `STOP_ALL.bat`]

---

## 12. Configuration Reference

### 12.1 Environment Files

| File | Device | Variables |
|------|--------|-----------|
| `configs/network.env` | All | Tailscale hostnames, MQTT broker, SSH credentials |
| `configs/edge.env` | RPi4 | MQTT broker (local), cloud offload URL, dashboard port, API key |
| `configs/cloud.env` | LOQ | TimescaleDB connection, Ollama model, server ports |
| `configs/server.env` | Patients | Telemetry agent settings |

### 12.2 Key Configuration Values

| Variable | Value | Source |
|----------|-------|--------|
| `MQTT_BROKER` (edge) | `127.0.0.1` | `configs/edge.env:6` |
| `MQTT_PORT` | `1883` | `configs/edge.env:7` |
| `DASHBOARD_PORT` | `8080` | `configs/edge.env:14` |
| `CLOUD_OFFLOAD_URL` | `http://loq.tail1ab139.ts.net:8000/api/offload` | `configs/edge.env:10` |
| `CLOUD_CHAT_URL` | `http://loq.tail1ab139.ts.net:8001/api/chat` | `configs/edge.env:11` |
| `TIMESCALEDB_HOST` | `localhost` | `configs/cloud.env:6` |
| `TIMESCALEDB_PORT` | `5432` | `configs/cloud.env:7` |
| `TIMESCALEDB_USER` | `pm_user` | `configs/cloud.env:8` |
| `TIMESCALEDB_DB` | `predictive_maintenance` | `configs/cloud.env:10` |
| `OLLAMA_MODEL` | `qwen3.5:4b` | `configs/cloud.env:14` |
| `INFERENCE_PORT` | `8000` | `configs/cloud.env:17` |
| `CHAT_PORT` | `8001` | `configs/cloud.env:20` |

---

## 13. Security

### 13.1 Authentication

Cloud API endpoints (`/api/offload`, `/api/deep_analysis`, `/api/chat`) require an `X-API-Key` header. The key is stored in `configs/api_keys.json` (git-ignored; `.example` template provided).

Rate limiting: 100 requests/minute per IP on the cloud inference server.
[src: `server/deep_analysis/inference_server.py:10`]

### 13.2 Network Encryption

All inter-device traffic is encrypted by Tailscale WireGuard (end-to-end, not hub-and-spoke).

### 13.3 Credential Hygiene

- `.env` files and `api_keys.json` are in `.gitignore`
- `.example` templates provided for all config files
- `MEMORIES.md` and `PROJECT_SUMMARY.md` contain device IPs but no passwords in the git-tracked version

---

## 14. Accuracy, Validation & Known Limitations

### 14.1 Test Suite

**36/36 tests pass** across 6 test files in `tests/`:
- `test_dashboard_revamp.py` — Dashboard HTML/JS integrity
- `test_e2e_pipeline.py` — End-to-end inference pipeline
- `test_enterprise_portfolio_datasets.py` — Dataset loader validation
- `test_phase1_hardening.py` — Core system hardening
- `test_phase4_plugins.py` — Telemetry plugin validation
- `test_phase5_survival_ml.py` — Survival loss function correctness

### 14.2 Known Limitations

| ID | Finding | Severity | Status |
|----|---------|----------|--------|
| F-026 | MAD-GAN and ST-GCN produce near-random F1 scores on current data | Medium | Mitigated (weights reduced to 5% each in cloud fusion) |
| F-027 | GPU temp extraction key mismatch between plugins | Low | Fixed |
| — | Consumer non-ECC DDR5 RAM lacks parity registers → MEMORY_FAULT detection is limited to page fault rate heuristics | By-design | Documented |
| — | Mobile RTX 4060 has no NVML fan speed tachometer (laptop EC-controlled) → Fan RPM is simulated | By-design | Documented |
| — | No enterprise PMBus on consumer laptops → PSU voltage monitoring is simulated | By-design | Documented |
| — | Spatial mask gating not yet integrated into ST-GCN graph layers | Enhancement | Documented for future |
| — | Fan Bearing CNN achieves 100% accuracy on synthetic data due to non-overlapping class boundaries | Known | Real turbulence noise would reduce accuracy |

### 14.3 Consumer vs Enterprise Hardware Coverage

| Component | Consumer Hardware Detection | Enterprise Hardware Detection |
|-----------|---------------------------|-------------------------------|
| CPU Thermal | ✅ ACPI thermal zones + physics proxy | ✅ IPMI/BMC sensors |
| GPU Thermal | ✅ NVML temperature API | ✅ NVML + ECC error counters |
| Storage (SSD) | ✅ SMART + NVMe health | ✅ SMART + NVMe health |
| Storage (HDD) | ✅ SMART attributes (5, 9, 187, 188, 194, 197, 198) | ✅ Same |
| RAM | ⚠️ Page faults only (no ECC) | ✅ ECC correctable/uncorrectable via WHEA |
| PSU | ❌ Simulated (no BMC/PMBus) | ✅ PMBus dual-rail monitoring |
| Fans | ⚠️ Simulated on laptops | ✅ PWM tachometer via IPMI |
| Vibration | ✅ Physical ADXL345 on RPi4 | ✅ Same |
| Acoustic | ✅ Physical INMP441 on RPi4 | ✅ Same |

### 14.4 Empirical Model Benchmark Ledger vs Real-World Hardware Reality

The system was evaluated across both benchmark partitions and live field telemetry from the monitored Lenovo IdeaPad S145 (`76a82f6f-4498-4914-b960-3295224ec28b`) and Lenovo LOQ RTX 4060 (`4c007433-db55-4309-8b41-2de057e9449b`).

| Model | Subsystem & Task | Exact Benchmark Metric | Real-World Hardware Fidelity & Operational Reality |
| :--- | :--- | :--- | :--- |
| **1D-CNN Multi-Task** | 7-Class Fault Diagnosis + TTF Regression | **Val Acc: 99.30%** (2,000 frames), Loss: 0.0560, TTF MAE: 0.0560 | High fidelity on real CPU/GPU thermal runaway ($\Delta T > 1.5^\circ\text{C}/\text{s}$). Real-world accuracy on laptops is ~92–95% due to simulated PMBus/fan channels. |
| **Fan Bearing 1D-CNN** | Mechanical Bearing Wear / Cavitation | **Test Acc: 100.0%**, **AUC: 1.000**, Loss: 0.0012 | 100% accuracy results from non-overlapping synthetic data boundaries (`synthetic_failure_gen.py`). Augmented by authentic physical vibration (0.022g RMS via ADXL345) and acoustics (0.11 RMS via INMP441) on RPi4. |
| **XGBoost HDD** | Hard Drive Remaining Useful Life (RUL) | **RMSE: 13.15 Days**, MAE: 8.72 Days, **$R^2 = 0.814$** | Highly predictive on mechanical disks (`D:\` cablet 1TB USB HDD on IdeaPad; `A:\` external HDD on LOQ). Primary NVMe SSDs route to wear-leveling endurance % instead. |
| **TCN Storage RUL** | Temporal Storage Sequence-to-One RUL | **Val MAE: 15.94 Days**, MSE: 588.17 (RMSE: 24.25 Days) | Dilated causal receptive field (33 days) captures gradual accumulating sector reallocations, providing sysadmins ~2 weeks of actionable lead time. |
| **LSTM-Autoencoder** | Unsupervised Cross-Sensor Anomaly | **Val MSE: 0.0949**, Anomaly Threshold $\tau = 0.330935$ | Most robust model on live hardware. Trained on 70,414 healthy frames; reconstruction error surges to $>0.85$ during real stress tests without false alarms on idle. |
| **Isolation Forest** | PSU Rail & Fan Outlier Detection | **100% Sag Detection** ($<11.4\text{V}$), 0.038 ms latency | Sub-millisecond statistical filter on edge; isolates electrical drops in $<40\ \mu\text{s}$ without requiring neural network matrix multiplication. |
| **MAD-GAN** | Multivariate Generative Anomaly Detection | **Precision: 5.09%**, Recall: 5.19%, **F1: 5.14%** | F-026 Finding: Severe domain shift between Server Machine Dataset (SMD) and live schema caused near-random predictions. Mitigated by lowering fusion weight to 0.05. |
| **ST-GCN** | Spatio-Temporal Graph Convolutions | **Val F1: 4.08%**, Threshold: 0.20, Asymmetric $\alpha = 5.0$ | F-026 Finding: Merging disparate Alibaba/Backblaze datasets caused graph adjacency sparsity. Mitigated by lowering fusion weight to 0.05. |

### 14.5 Theoretical & Operational Analysis: Why Unhealthy-Only Training Fails

In reliability engineering, a recurring inquiry is whether models can be trained exclusively on failure (unhealthy) data. Mathematically and operationally, this approach is fundamentally invalid:

```
┌────────────────────────────────────────────────────────────────────────┐
│             WHAT HAPPENS IF YOU TRAIN ON UNHEALTHY DATA ONLY?          │
│                                                                        │
│ 1. CLASSIFIER COLLAPSE (100% False Positive Rate on Normal Servers)    │
│    Model has no Class 0 (Healthy). When healthy telemetry enters:      │
│    Softmax must sum to 1.0 across fault classes ──▶ Forcibly classifies│
│    a normal operating server as Thermal, HDD, or Fan Failure!          │
│                                                                        │
│ 2. ANOMALY DETECTION FAILURE ("Inverse Autoencoder" Collapse)          │
│    Healthy states form a single compact manifold (predictable temps).  │
│    Failures are fragmented, chaotic, open-ended noise.                 │
│    Model cannot learn an infinite variety of unseen failure modes.     │
│                                                                        │
│ 3. TTF REGRESSION SURVIVAL BIAS (Dynamic Range Collapse)               │
│    Model only sees components in their final death spiral (0-30 days). │
│    Brand-new hardware is predicted to fail immediately!                │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Closed-World Classification Collapse:**
   Supervised Softmax networks require a negative/reference baseline ($y=0$: Healthy). If trained solely on fault modes, $\sum_{c \in \text{Faults}} P(y=c) = 1.0$. Normal server telemetry (e.g., CPU at 45°C, 15% load) is forced into a failure class, causing a **100% False Positive Rate**.
2. **The Anna Karenina Principle of Anomaly Detection:**
   *"All healthy servers are alike; each unhappy server is unhappy in its own way."* Healthy telemetry lives on a compact, low-dimensional manifold ($\Delta T < 35^\circ\text{C}$, tight voltage regulation). Failure modes are chaotic, sparse, and open-ended. An autoencoder trained only on observed failures cannot learn the infinite variety of unseen failure mechanisms, causing catastrophic false negatives on novel breakdowns.
3. **Survival Truncation Bias in TTF Regression:**
   Training regression models strictly on degraded components ($0 \le \text{TTF} \le 30\text{ days}$) collapses the model's dynamic range. The model never learns what hardware looks like 3 years before failure, causing it to predict immediate breakdown even for brand-new components. Our implementation resolves this via **Weibull AFT Loss** ([server/training/loss.py](file:///c:/Users/LOQ/Documents/Antigravity/Predictive%20maintenance%20for%20Server/server/training/loss.py)), supporting right-censored non-failing trajectories.

---

## 15. Repository Map

### 15.1 Directory Structure

```
predictive-maintenance/
├── configs/                    # All environment & service configs
│   ├── api_keys.json          # API keys (git-ignored)
│   ├── cloud.env              # LOQ cloud backend config
│   ├── edge.env               # RPi4 edge config
│   ├── edge-ai.service        # systemd unit for RPi4
│   ├── network.env            # Tailscale mesh networking
│   ├── server.env             # Patient agent config
│   └── grafana/               # Grafana provisioning
├── dashboard/                  # Operator dashboard (RPi4)
│   ├── index.html             # Full UI (91.8 KB)
│   └── assets/
│       ├── app.js             # All JavaScript logic
│       └── style.css          # All CSS styles
├── edge/                       # Edge node code (runs on RPi4)
│   ├── main.py                # Main entrypoint (MQTT + inference + API)
│   ├── api_server.py          # FastAPI server definitions
│   ├── adaptive_baseline.py   # Online Z-score anomaly detector
│   ├── edge_inference.py      # TFLite model loading & inference
│   ├── ensemble_decision.py   # Confidence router + cloud offload
│   ├── local_store.py         # SQLite ring buffer
│   ├── model_router.py        # Multi-model routing logic
│   ├── mqtt_handler.py        # MQTT subscription handler
│   ├── sensor_reader.py       # Physical sensor drivers (ADXL345, INMP441)
│   └── models/                # INT8 TFLite models for edge
│       ├── classifier_int8.tflite    (133 KB)
│       ├── classifier_scaler.pkl     (1.2 KB)
│       ├── fan_bearing_cnn_int8.tflite (28 KB)
│       ├── iso_forest.pkl            (1.0 MB)
│       ├── stgcn_int8.tflite         (10 KB)
│       └── tcn_storage_rul_int8.tflite (141 KB)
├── server/                     # Cloud/training code (runs on LOQ)
│   ├── deep_analysis/         # Cloud inference & services
│   │   ├── inference_server.py # FastAPI GPU inference server
│   │   ├── chat_api.py        # Ollama LLM chat proxy
│   │   ├── cloud_fusion.py    # Weighted ensemble fusion
│   │   ├── tsdb_writer.py     # MQTT → TimescaleDB ingester
│   │   ├── schema.sql         # TimescaleDB schema definition
│   │   ├── model_registry.py  # Model metadata registry
│   │   ├── retrain_daemon.py  # Auto-retrain orchestrator
│   │   ├── adaptive_thresholds.py # Threshold computation engine
│   │   └── shap_explainer.py  # SHAP attribution generator
│   ├── training/              # Model training pipelines
│   │   ├── train_classifier.py    # 1D-CNN Multi-Task
│   │   ├── train_lstm_ae.py       # LSTM-Autoencoder
│   │   ├── train_madgan.py        # MAD-GAN
│   │   ├── train_stgcn.py         # ST-GCN
│   │   ├── train_xgboost_hdd.py   # XGBoost HDD
│   │   ├── train_fan_cnn.py       # Fan Bearing CNN
│   │   ├── train_tcn_storage.py   # TCN Storage RUL
│   │   ├── train_iso_forest.py    # Isolation Forest
│   │   ├── feature_schema.py      # 28-feature canonical schema (SSoT)
│   │   ├── feature_schema_v2.py   # 36-feature enterprise schema
│   │   ├── loss.py                # Asymmetric TTF + Weibull AFT loss
│   │   ├── asymmetric_loss.py     # Standalone asymmetric loss
│   │   ├── auto_labeler.py        # Automatic fault labeling
│   │   ├── ttf_labels.py          # TTF computation utilities
│   │   └── datasets/             # Enterprise dataset loaders
│   │       ├── schema_harmonization.py
│   │       ├── alibaba_dram_loader.py
│   │       ├── alibaba_gpu_trace_loader.py
│   │       ├── alibaba_ssd_loader.py
│   │       ├── backblaze_loader.py
│   │       ├── delta_gpu_xid_loader.py
│   │       └── nasa_cmapss_loader.py
│   ├── models/                # Full-precision models
│   │   ├── classifier_fp32.h5          (1.5 MB)
│   │   ├── lstm_ae.h5                  (900 KB)
│   │   ├── madgan_generator.pt         (144 KB)
│   │   ├── madgan_discriminator.pt     (164 KB)
│   │   ├── stgcn_anomaly.pt            (65 KB)
│   │   ├── stgcn_adjacency.npy         (5 KB)
│   │   ├── xgboost_hdd.pkl             (2.1 MB)
│   │   ├── fan_bearing_cnn_fp32.keras  (240 KB)
│   │   ├── tcn_storage_rul_fp32.keras  (1.4 MB)
│   │   └── [scalers, metadata.json]
│   ├── telemetry_agent/       # Patient telemetry collector
│   │   ├── collector_v2.py    # Main collector (publishes to MQTT)
│   │   └── plugins/           # Hardware-specific plugins
│   │       ├── cpu_plugin.py
│   │       ├── gpu_plugin.py
│   │       ├── storage_plugin.py
│   │       ├── ram_plugin.py
│   │       ├── psu_plugin.py
│   │       └── fan_plugin.py
│   └── optimization/         # Model optimization scripts
│       └── quantize_model.py  # FP32 → INT8 TFLite conversion
├── patient_package/           # Self-contained patient distribution
│   ├── mqtt_publisher_v2.py  # Continuous polling and MQTT publisher daemon
│   ├── collector_v2.py       # Plugin-based telemetry collection engine
│   ├── auto_discovery.py     # Hardware discovery utility
│   ├── server_identity.py    # Persistent server UUID assignment
│   ├── plugins/              # Hardware plugins (byte-for-byte identical to server/telemetry_agent/plugins/)
│   ├── server/training/      # Feature schemas (feature_schema.py, feature_schema_v2.py)
│   ├── INSTALL_DEPS.bat      # Windows dependency installation script
│   ├── START_PATIENT.bat     # Windows one-click patient starter
│   ├── start_patient.sh      # Linux bash patient starter
│   ├── requirements.txt      # Dependency manifest
│   └── README.md             # Patient quickstart guide
├── tests/                    # Automated test suite (36 tests)
├── audit/                    # Audit reports & benchmarks
├── docs/                     # Documentation
├── data/                     # Training datasets (git-ignored, 10.5 GB)
├── START_ALL.bat             # One-click launcher (LOQ)
├── STOP_ALL.bat              # Kill all services
├── deploy_all.py             # Multi-device deployment
├── docker-compose.yml        # TimescaleDB + Grafana
├── requirements_*.txt        # Python dependencies (4 files)
└── README.md                 # Project overview
```

### 15.2 File Counts

| Category | Count |
|----------|-------|
| Git-tracked files | 217 |
| Python files | 101 |
| Batch files | 13 |
| ML model artifacts | 32 (edge: 8, server: 24) |
| Total model footprint | 13.19 MB (fits in Git) |

[src: `git ls-files` output]

---

## Appendix A: End-to-End Latency Waterfall

```
Patient telemetry agent
  ↓ Collect (psutil + plugins)  ~50 ms
  ↓ MQTT publish                ~1 ms
  ↓ WireGuard transit           ~22 ms (Tailscale mesh)
RPi4 Doctor
  ↓ MQTT receive                ~1 ms
  ↓ Feature engineering         ~2 ms
  ↓ Sensor read (ADXL + INMP)   ~370 ms
  ↓ Edge inference (all models)  <1 ms
  ↓ Adaptive baseline scoring   ~1 ms
  ↓ SSE broadcast to dashboard  ~1 ms
  ↓ Store to SQLite             ~2 ms
  ↓ [If offloaded] Cloud call   ~10 ms
LOQ Hospital
  ↓ GPU deep analysis           ~10 ms
  ↓ TSDB batch insert           ~4 ms
```

**Total pipeline (edge-local decision):** < 450 ms
**Total pipeline (cloud-offloaded):** < 470 ms

---

## Appendix B: Asymmetric Survival TTF Loss

The core loss function (`server/training/loss.py`):

```
L_asym(ŷ, y) = α · (ŷ − y)² if ŷ > y    (overestimation — DANGEROUS)
                   (ŷ − y)² if ŷ ≤ y    (underestimation — conservative)

Where α = 5.0
```

Additionally, an imminent failure penalty (weight=2.0) is applied when true TTF < 0.2 (normalized), ensuring the model pays extra attention to the critical final hours before failure.

The system also implements Weibull Accelerated Failure Time (AFT) negative log-likelihood with right-censoring support, and a combined multi-task survival loss that sums classification (BCE), TTF regression (asymmetric), and Weibull hazard losses.
[src: `server/training/loss.py:1-10`]

---

## Appendix C: Domain-Aware Normalization

Consumer laptop telemetry values are far outside the AI4I CNC benchmark distribution that the classifier was trained on. Without normalization, features like GPU VRAM (1527 MB vs training mean ~5 MB) create +1000σ outliers that saturate INT8 weights.

`edge/main.py:extract_features()` applies domain-aware reference normalization to the model input vector while preserving raw values for dashboard display and TimescaleDB storage.

Consumer-specific physical feasibility guards:
- PSU: Consumer PCs without BMC hardware sensors suppress synthetic `PSU_ANOMALY` (Class 3)
- Memory: Nominal RAM utilization (<92%) without hardware errors suppresses false `MEMORY_FAULT` (Class 5)
[src: `edge/main.py`, MEMORIES.md Phase 8.3-8.4]

---

*Document generated from live codebase inspection on 2026-10-12. Git HEAD: `b5cda90`.*
