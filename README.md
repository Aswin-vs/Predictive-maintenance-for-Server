![CI](https://github.com/Aswin-vs/Predictive-maintenance-for-Server/actions/workflows/ci.yml/badge.svg)

# 🖥️ Edge-AI Predictive Maintenance for Server Hardware

**Status: 🚀 100% PRODUCTION CERTIFIED — Phase 7 & Phase 2B Complete: Multi-Device End-to-End Latency Waterfall Verified, Physical Sensors (ADXL345+INMP441), Tailscale Mesh (`tail1ab139.ts.net`), 8 Enterprise Datasets, Asymmetric Survival TTF Loss, TimescaleDB, 37/37 Unit & Regression Tests Passed (2026-10-16)**

**B.Tech Computer Science and Engineering (AI Specialization) — S7 Main Project (23CTB40B Software Engineering Assignment 2)**

---

### 👥 Team Members (Main Project Group)

| Roll No. | College ID | KTU Reg No. | Name | Role / Module Assigned |
| :---: | :---: | :---: | :--- | :--- |
| **21** | B23CS2121 | MBT23CA021 | **Aswin V S** | Edge Inference Engine, Confidence Router & Boundary Value Analysis (BVA) Testing |
| **29** | B23CS2129 | MBT23CA029 | **Gokul Jayan J R** | GitHub Actions CI/CD Pipeline, Automated Testing & Cloud Decision Fusion |
| **40** | B23CS2140 | MBT23CA039 | **Karthik S** | TimescaleDB Hypertable Ingestion & SQLite WAL Outbox Watermark Sync |
| **59** | B23CS2159 | MBT23CA058 | **Sijo S** | Operator Console UI, Live Node Status Indicators & Multi-Sensor Telemetry |

---

## 📖 Project Overview

A production-grade Edge-AI system that predicts hardware failures (CPU, GPU, HDD/SSD, PSU, RAM, Fans) using a hybrid 3-tier Edge-Cloud architecture with confidence-aware offloading, physical hardware sensor reading, and hardware-agnostic multi-patient support.

### Architecture (3-Tier: Patient → Doctor → Hospital)

1. **Patient (Monitored Server — IdeaPad S145 / LOQ):** Runs the telemetry agent with **auto-discovery** and **tiered degradation plugins** (`collector_v2.py`) to profile and collect hardware stats. Publishes hierarchical v2 telemetry + registration payloads via MQTT over encrypted Tailscale WireGuard mesh.
2. **Doctor (Edge — Raspberry Pi 4 Model B 4GB):** Connected to physical I2C (ADXL345 accelerometer) and ALSA I2S (INMP441 MEMS microphone) sensors. Runs quantized INT8 models (`stgcn`, `classifier`, `fan_bearing_cnn`, `tcn_storage`) via Google LiteRT in sub-millisecond deterministic latency. Hosts the Operator Dashboard (`:8080`) with live Server-Sent Events (SSE) and SQLite ring cache.
3. **Hospital (Cloud — LOQ RTX 4060 GPU):** Handles deep multi-model decision fusion for uncertain edge predictions (confidence < 0.85), heavy GPU training, SHAP explainability, Ollama Qwen 3.5 4B diagnostic assistant, TimescaleDB hypertable ingestion, and Grafana visualization.

### AI Model Portfolio (10 Models & Engines)

| # | Model / Engine | Task | Location | Footprint | Verified Latency | Key Empirical Metrics |
|---|---|---|---|---|---|---|
| 1 | **1D-CNN Multi-Task** | 7-class fault classification + TTF regression | Both (LiteRT INT8 / FP32) | 132.90 KB | **0.018 ms – 0.160 ms** (Edge) | **99.38% Acc**, 0.2129 TTF MAE (21.92d bounded) |
| 2 | **ST-GCN Enterprise** | Spatio-temporal graph anomaly (36 features) | Both (LiteRT INT8 / PyTorch) | 9.7 KB / 63 KB | **0.029 ms** (Edge) / 1.69 ms (Cloud) | **44.03% – 57.93% F1**, Spatial Mask Gating Active |
| 3 | **Fan Bearing CNN** | PWM/RPM deviation + acoustic/vibration shock | Edge (LiteRT INT8) | 27.2 KB | **0.076 ms** (Edge) | **100% Test Accuracy**, AUC: 1.000 |
| 4 | **TCN Storage RUL** | Temporal Conv remaining useful life (Backblaze SMART) | Edge (LiteRT INT8) | 141.0 KB | **0.547 ms** (Edge) | **15.94 Days MAE**, 33-day receptive field |
| 5 | **Isolation Forest** | Unsupervised electrical/thermal anomaly detection | Edge (scikit-learn) | 1,020 KB | ~1.5 ms (Edge) | **100% Sag Detection** (<11.4V sags) |
| 6 | **LSTM-Autoencoder** | Unsupervised reconstruction error anomaly detection | Cloud (Keras / GPU) | 900 KB | ~4.5 ms (Cloud) | **0.0949 MSE Loss**, Threshold $\tau=0.330935$ |
| 7 | **MAD-GAN (28-Ch)** | Multivariate anomaly generator + discriminator | Cloud (PyTorch CUDA) | 308 KB | **1.87 ms** (Cloud) | **99.96% Prec, 80.62% Rec, 89.25% F1** |
| 8 | **XGBoost HDD** | Enterprise drive failure specialist (Backblaze) | Cloud + SHAP | 2.1 MB | **3.08 ms** (Cloud) | **90.73% 30-day Acc, RMSE: 17.54d, MAE: 11.01d** |
| 9 | **Cloud Decision Fusion** | Weighted multi-model ensemble & severity triage | Cloud (Python) | — | <0.1 ms (Cloud) | **60% Edge 1D-CNN, 20% LSTM-AE, 10% ST-GCN, 10% MAD-GAN** |
| 10 | **Qwen 3.5 4B (Ollama)** | RAG diagnostic assistant over live telemetry & SHAP | Cloud (GPU, 3.1GB VRAM) | 3.1 GB | ~7.9 sec (Cloud) | Real-time Context-Aware Hardware Actions |

## 📁 Repository Structure

```
edge/                        # Runs on Raspberry Pi 4 (Doctor)
├── main.py                  # SINGLE ENTRYPOINT (MQTT→Inference→Dashboard)
├── mqtt_handler.py          # MQTT subscriber v2 (registration + multi-tenant + v1/v2)
├── edge_inference.py        # TFLite INT8 + Isolation Forest + confidence router
├── api_server.py            # FastAPI dashboard server (SSE streaming)
├── sensor_reader.py         # External sensor I/O (ADXL345, DHT22, MCP3008)
├── model_router.py          # [G3] Routes telemetry to 4 specialist models
├── ensemble_decision.py     # [G3] Weighted ensemble health scoring engine
├── local_store.py           # SQLite ring buffer cache
├── edge_logger.py           # Structured logging
└── models/                  # Quantized models (.tflite, .pkl, .joblib)

server/
├── telemetry_agent/         # Runs on the monitored server (Patient)
│   ├── auto_discovery.py    # [G1] Hardware topology profiler
│   ├── collector_v2.py      # [G1] Plugin-based telemetry collector
│   ├── mqtt_publisher_v2.py # [G1] v2 registration + hierarchical MQTT
│   ├── server_identity.py   # UUID-based server identity
│   ├── plugins/             # [G1] OS-aware telemetry plugins
│   │   ├── cpu_plugin.py    #   CPU (psutil, cross-platform)
│   │   ├── gpu_nvidia_plugin.py  # NVIDIA GPU (nvidia-smi)
│   │   ├── storage_plugin.py     # SMART + disk I/O
│   │   ├── memory_plugin.py      # RAM/swap
│   │   └── psu_fan_plugin.py     # PSU voltage + fan RPM
│   ├── collector.py         # [Legacy v1] Monolithic collector
│   └── mqtt_publisher.py    # [Legacy v1] Flat telemetry publisher
├── deep_analysis/           # Runs on LOQ RTX 4060 (Cloud)
│   ├── inference_server.py  # /api/offload + /api/deep_analysis endpoints
│   ├── cloud_fusion.py      # [G4] Weighted multi-model decision fusion
│   ├── chat_api.py          # /api/chat (Ollama + RAG)
│   └── shap_explainer.py    # SHAP for XGBoost
├── training/                # Model training scripts
│   ├── feature_schema.py    # v1 feature schema (28 features)
│   ├── feature_schema_v2.py # [G1] v2 hierarchical schema + v1↔v2 converters
│   ├── train_tcn_storage.py # [G3] TCN Storage RUL (Backblaze SMART data)
│   ├── train_fan_cnn.py     # [G3] Fan Bearing 1D-CNN (synthetic data)
│   ├── train_madgan.py      # [G4] MAD-GAN (PyTorch CUDA, SMD dataset)
│   ├── train_stgcn.py       # [G4] STGCN (PyTorch CUDA, SKAB dataset)
│   ├── synthetic_failure_gen.py  # [G3] Synthetic fan degradation generator
│   └── ...                  # train_classifier, train_lstm_ae, train_xgboost, etc.
└── optimization/            # FP32→INT8 quantization

dashboard/                   # Glassmorphic HTML/JS/CSS operator UI (9 tabs)
configs/                     # Per-device .env files (edge, server, cloud)
tests/                       # End-to-end API tests
docs/                        # Design docs, setup guides, device roles
data/                        # Datasets (Backblaze, AI4I 2020, NASA C-MAPSS, SMD, SKAB)
```

## 🔒 Security

All POST API endpoints require authentication via `X-API-Key` header:

- **API Key:** `<CONFIGURED_IN_ENV>` (loaded from `configs/edge.env` / `configs/api_keys.json`)
- **Cloud:** Keys stored in `configs/api_keys.json`
- **Edge:** `API_KEY` loaded from `configs/edge.env`
- Rate limiting: 100 req/min per IP on cloud endpoints
- Health/state GET endpoints are open (no auth required)

## 🚀 Quick Start

### Patient Server (monitored machine):

```bash
# v2 Publisher (recommended) — auto-discovery + hierarchical telemetry
python server/telemetry_agent/mqtt_publisher_v2.py --broker <RPI4_IP> --interval 5

# v1 Legacy Publisher (backward compatible)
python server/telemetry_agent/mqtt_publisher.py --broker <RPI4_IP> --dummy
```

### RPi4 Edge Node:

```bash
# Preferred: systemd (auto-starts on boot)
sudo systemctl start edge-ai

# Or manually:
export CLOUD_OFFLOAD_URL=http://loq.tail1ab139.ts.net:8000/api/offload
python edge/main.py --broker 127.0.0.1
# Dashboard: http://rpi4-edge.tail1ab139.ts.net:8080/ (or http://100.90.160.78:8080/)
```

### LOQ Cloud Backend:

```bash
python server/deep_analysis/inference_server.py  # Port 8000
python server/deep_analysis/chat_api.py          # Port 8001
```

### 🛠️ Simplified 1-Click Batch Launchers:

| Batch File | Machine | Role | Function |
|---|---|---|---|
| [`START_ALL.bat`](START_ALL.bat) | **LOQ** | **Master System Orchestrator** | **1-Click Launch All:** Docker TimescaleDB, Cloud Inference (:8000), Chat API (:8001), Ingestion, RPi4 SSH restart, and LOQ Telemetry. |
| [`STOP_ALL.bat`](STOP_ALL.bat) | **LOQ** | **Master System Shutdown** | Gracefully stops all local python daemons, uvicorn ports 8000/8001, RPi4 edge service, and Docker containers. |
| [`clear_telemetry_cache.bat`](clear_telemetry_cache.bat) | **LOQ** | **Maintenance Utility** | 1-Click interactive cache purge across SQLite and adaptive baselines. |
| [`start_loq_cloud.bat`](start_loq_cloud.bat) | **LOQ** | **Hospital (Cloud)** | Starts Cloud Inference Server (`:8000`) and Chat API (`:8001`). |
| [`start_rpi4.bat`](start_rpi4.bat) | **LOQ** | **Doctor (RPi4 Edge)** | Restarts & verifies RPi4 Edge AI service over SSH. |
| [`start_patient_ideapad_from_loq.bat`](start_patient_ideapad_from_loq.bat) | **LOQ** | **Patient (IdeaPad)** | Remotely starts IdeaPad telemetry over SSH. |
| [`start_patient_ideapad_local.bat`](start_patient_ideapad_local.bat) | **IdeaPad** | **Patient (IdeaPad)** | Runs telemetry directly on IdeaPad console. |
| [`start_patient_loq.bat`](start_patient_loq.bat) | **LOQ** | **Patient (LOQ)** | Runs LOQ as Patient streaming live stats to RPi4. |
| [`start_dummy_simulator.bat`](start_dummy_simulator.bat) | **LOQ** | **Anomaly Simulator** | Injects realistic hardware anomalies into RPi4 MQTT. |
| [`deploy_all.py`](deploy_all.py) | **LOQ** | **Deployment** | Auto-deploys all files over SSH/SFTP to RPi4 and IdeaPad. |
| [`setup_rpi4_fresh.py`](setup_rpi4_fresh.py) | **LOQ** | **RPi4 Setup** | Full automated fresh RPi4 setup (apt -> pip -> deploy -> systemd). |

## 🏗️ Phase G: Advanced Architecture (Complete)

Phase G moves the system from a flat 28-feature vector to a **hierarchical, hardware-agnostic** architecture:

- **G1 ✅** Auto-Discovery + Plugin-Based Collection + v2 Hierarchical Schema
- **G2 ✅** TimescaleDB on LOQ (replaces InfluxDB) + data ingestion pipeline
- **G3 ✅** New Edge Models: TCN Storage RUL + Fan Bearing 1D-CNN (INT8 TFLite) + RPi4 Deploy
- **G4 ✅** New Cloud Models: MAD-GAN + STGCN + Cloud Decision Fusion (PyTorch CUDA)
- **G5 ✅** Adaptive Thresholds (P85/P97 percentile) + Dashboard v2 (Cloud Fusion tab)
- **G6 ✅** Integration Testing + Comprehensive Documentation (PRODUCTION CERTIFIED)

See [`agent_handover/PHASE_G_RESEARCH_PLAN.md`](agent_handover/PHASE_G_RESEARCH_PLAN.md) for the full research plan.

## 📊 Model Performance

| Model | Dataset | Key Metric |
|---|---|---|
| 1D-CNN Multi-Task | AI4I 2020 | **99.30% val accuracy** |
| LSTM-Autoencoder | AI4I 2020 healthy | **0.3883 MSE**, threshold 0.3657 |
| XGBoost HDD | 17.1M Backblaze records | 500 trees, GPU |
| TFLite INT8 | Quantized from FP32 | **132.57 KB** |
| TCN Storage RUL | Backblaze SMART | **15.94 days MAE**, INT8 141 KB |
| Fan Bearing CNN | Synthetic PWM/RPM | **100% accuracy**, INT8 27.2 KB |
| MAD-GAN | SMD (5 machines) | **F1=0.05** (unsupervised), 32s train |
| STGCN | SKAB (35 CSVs) | **F1=0.57**, Recall=1.00, 11s train |

## 📄 Documentation & Audit Artifacts

| Document | Purpose |
|---|---|
| [`MEMORIES.md`](MEMORIES.md) | Single source of truth for project memory, sprint status & architectural decisions |
| [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) | Executive chronological timeline of changes and milestones |
| [`docs/VIVA_MASTER_GUIDE.md`](docs/VIVA_MASTER_GUIDE.md) | Comprehensive 22-section academic defense and technical Q&A reference |
| [`docs/project_viva_walkthrough.md`](docs/project_viva_walkthrough.md) | 18-question examiner defense cheatsheet |
| [`audit/e2e_smoke_test.md`](audit/e2e_smoke_test.md) | Verified live latency audit across all 5 hops |
| [`audit/demo_runbook.md`](audit/demo_runbook.md) | Step-by-step Demo Day presentation script & emergency runbook |
| [`audit/dependency_versions.md`](audit/dependency_versions.md) | Complete cross-platform pinned runtime & package audit |
| [`audit/manifest_diff.json`](audit/manifest_diff.json) | Cryptographic SHA-256 multi-machine verification (zero code skew) |
| [`docs/master_execution_guide.md`](docs/master_execution_guide.md) | 3-tier startup and verification guide |
| [`docs/rpi4_setup_guide.md`](docs/rpi4_setup_guide.md) | Complete RPi4 OS & LiteRT setup guide |

## 🖥️ Device Inventory (Tailscale Mesh: `tail1ab139.ts.net`)

| Device | MagicDNS Hostname | Tailscale IP | Role | OS | Key Software |
|---|---|---|---|---|---|
| **Lenovo LOQ** | `loq.tail1ab139.ts.net` | `100.111.172.121` | Hospital (Cloud) | Windows 11 | RTX 4060, PyTorch 2.6 CUDA, Ollama Qwen 3.5 4B, TimescaleDB, Grafana |
| **Raspberry Pi 4** | `rpi4-edge.tail1ab139.ts.net` | `100.90.160.78` | Doctor (Edge) | Debian 13 Trixie | Google LiteRT 2.2.0, Mosquitto 2.0.21, ADXL345 (I2C), INMP441 (ALSA) |
| **IdeaPad S145** | `ideapads145.tail1ab139.ts.net` | `100.97.69.16` | Patient (Monitored) | Windows 10 | Telemetry Agent v2 (`collector_v2.py`), Windows Scheduled Task |

## 📜 License

MIT License

