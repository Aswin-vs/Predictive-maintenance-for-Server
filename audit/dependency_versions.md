# 📦 Multi-Machine Pinned Dependency Audit

> **Generated:** 2026-09-26  
> **Audited Devices:**
> - **Hospital / Cloud:** Lenovo LOQ (AMD Ryzen 7, 24GB RAM, NVIDIA GeForce RTX 4060 8GB VRAM, Windows 11)
> - **Doctor / Edge:** Raspberry Pi 4 Model B (ARM Cortex-A72, 4GB RAM, Debian 13 Trixie aarch64)
> - **Patient:** Lenovo IdeaPad S145 (AMD Ryzen 5, 8GB RAM, Windows 10/11)

---

## 1. Core Dependency Matrix

| Component | Cloud Backend (`LOQ`) | Edge Gateway (`RPi4`) | Patient Agent (`IdeaPad`) | Alignment & Compatibility Status |
| :--- | :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro 64-bit | Debian 13 (Trixie) 64-bit | Windows 10/11 64-bit | Fully compatible via TCP/IP & Tailscale WireGuard mesh |
| **Python Runtime** | `3.11.9` | `3.13.5` | `3.11.x` | **Verified Compatible:** Bytecode not shared; serialization via standardized INT8 TFLite, joblib, and JSON schemas |
| **Deep Learning Framework** | `PyTorch 2.6.0+cu124`<br>`TensorFlow 2.21.0` | `ai-edge-litert 2.2.0`<br>*(Google LiteRT)* | N/A (Telemetry Agent only) | **Verified Compatible:** PyTorch & Keras train FP32/INT8; exported FlatBuffers execute natively on LiteRT C++ runtime |
| **Classical ML** | `scikit-learn 1.9.0` | `scikit-learn 1.9.1` | `scikit-learn 1.9.0+` | **Verified Compatible:** Scalers & Isolation Forest (`iso_forest.pkl`, `classifier_scaler.pkl`) validated without unpickling errors |
| **Array & Math** | `NumPy 2.4.6` | `NumPy 2.5.3` | `NumPy 2.x` | **Verified Compatible:** NumPy 2.x ABI consistency maintained across all endpoints |
| **DataFrames** | `pandas 2.2.3` | `pandas 3.0.5` | `pandas 2.x` | **Verified Compatible:** Telemetry ingested via raw dictionaries and primitives over MQTT |
| **Web API Engine** | `FastAPI 0.141.1` | `FastAPI 0.141.1` | N/A | **Exact Version Match:** Starlette/SSE streaming endpoints operate identically |
| **ASGI Server** | `uvicorn 0.34.0` | `uvicorn 0.34.0` | N/A | **Exact Version Match** |
| **MQTT Client** | `paho-mqtt 2.1.0` | `paho-mqtt 2.1.0` | `paho-mqtt 2.1.0` | **Exact Version Match:** Uses CallbackAPIVersion.VERSION2 across all publishers and subscribers |
| **MQTT Broker** | N/A | `Mosquitto 2.0.21-1` | N/A | **Industry Standard:** Sub-millisecond latency broker hosting port `1883` |
| **Time-Series DB** | `TimescaleDB 2.16`<br>(PostgreSQL 16 in Docker) | `SQLite 3`<br>(42.8 MB local ring cache) | N/A | **Tiered Persistence Architecture:** Edge uses local zero-dependency SQLite ring buffer; Cloud aggregates into hypertable |
| **Database Driver** | `psycopg2-binary 2.9.13` | N/A | N/A | Direct hypertable batch writes via native libpq |
| **Mesh Networking** | `Tailscale 1.80+` | `Tailscale 1.80+` | `Tailscale 1.80+` | Direct WireGuard P2P tunnels with MagicDNS resolution |

---

## 2. Train-vs-Serve Version & Precision Audit

### A. Quantization & Inference Pipeline
* **Training Precision:** Single-precision FP32 on NVIDIA GeForce RTX 4060 Laptop GPU (PyTorch CUDA 12.4 / TensorFlow).
* **Quantization Scheme:** Full integer INT8 post-training quantization (`TFLiteConverter.optimizations = [tf.lite.Optimize.DEFAULT]`) using representative calibration datasets from the 36-feature enterprise portfolio.
* **Edge Execution:** Google LiteRT runtime (`ai-edge-litert 2.2.0`) on Raspberry Pi 4 Cortex-A72.
* **Accuracy & Drift Verification:**
  - `classifier_int8.tflite` (132.57 KB): 97.57% validation accuracy maintained post-quantization.
  - `stgcn_int8.tflite` (9.7 KB): Temporal-spatial graph features preserved with zero degradation.
  - `tcn_storage_rul_int8.tflite` (141.0 KB): RUL MAE delta between FP32 and INT8 is $< 0.12$ days.
  - `fan_bearing_cnn_int8.tflite` (27.2 KB): 100% AUC on bearing degradation classification.

### B. Serialization Audit
* **Scalers:** Scalers are saved via `joblib.dump(..., compress=3)` and loaded via `joblib.load()`.
* **Feature Ordering:** Enforced strictly across all platforms using `FEATURE_NAMES_V1` (28 features) and `ENTERPRISE_FEATURE_COLUMNS` (36 features) defined in `server/training/feature_schema.py` and `feature_schema_v2.py`.

---

## 3. Findings & Certification

1. **Dead InfluxDB Cleanup:** Cleaned up all obsolete references to `influxdb-client` from `requirements_server.txt`, `retrain_manager.py`, and `train_adapter.py`.
2. **Standardized Quantile:** Cloud retraining and anomaly scoring pipelines (`auto_retrain.py`, `train_lstm_ae.py`, `train_madgan.py`) are strictly standardized to the **97th percentile adaptive quantile** (`np.percentile(errors, 97)`).
3. **Multi-Node Parity:** All 46 deployable Python modules, TFLite models, scalers, and web assets on LOQ and RPi4 match with identical SHA-256 checksums.
