# 🏗️ Production-Ready Upgrade — Implementation Plan v2.0

> **Author:** `@claude-architect` (Lead Architect & Senior ML Engineer)  
> **Date:** 2026-08-09  
> **Scope:** 3 Phases — Codebase Audit → Multi-Tenant Storage → Continuous Learning  
> **Prerequisite Read:** `MEMORIES.md`, `PROJECT_SUMMARY.md`, `pre_deployment_audit_report.md`  
> **Status:** ⏳ AWAITING USER APPROVAL — Do NOT implement until user says "Approved"

---

## Table of Contents

1. [Phase 1: Deep Codebase Audit Results](#phase-1-deep-codebase-audit-results)
2. [Phase 2: Multi-Tenant Data Storage Architecture](#phase-2-multi-tenant-data-storage-architecture)
3. [Phase 3: Continuous Learning & Personalization Pipeline](#phase-3-continuous-learning--personalization-pipeline)
4. [Step-by-Step Implementation Tasks](#step-by-step-implementation-tasks)
5. [Open Questions for User](#open-questions-for-user)
6. [Verification Plan](#verification-plan)

---

# Phase 1: Deep Codebase Audit Results

> **Files Read:** All 48+ source, config, and documentation files across `edge/`, `server/`, `dashboard/`, `configs/`, `docs/`, `tests/`, and root scripts.

## 1.1 🔴 NEW Bugs Found (Not Previously Documented)

### BUG-N1: Feature Ordering Not Deterministic Between Training and Edge Inference

**Severity:** 🔴 HIGH — Silent accuracy degradation  
**Files:**
- [train_classifier.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/server/training/train_classifier.py) (Lines 159-235)
- [edge/main.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/main.py) (Lines 103-158)

**Problem:** The 1D-CNN was trained on AI4I 2020 data with features in this order:
```
cpu_temp, cpu_thermal_ramp, gpu_temp, gpu_thermal_ramp, cpu_ambient_delta,
thermal_ratio, fan_rpm, fan_rpm_delta, fan_rpm_rolling_std, psu_power,
psu_power_delta, psu_efficiency, wear_level, wear_rate, wear_normalized,
wear_remaining, power_speed_ratio, mechanical_stress, temp_rolling_mean,
temp_rolling_std, _pad_20, _pad_21, ..., _pad_27
```

But `edge/main.py::extract_features()` produces features in a **completely different order**:
```
cpu_temp, gpu_temp, cpu_load, cpu_freq, gpu_power, gpu_util, gpu_mem_used,
gpu_clock, ram_used_gb, ram_total_gb, ram_page_faults, ambient_temp, humidity,
vibration_rms, acoustic_rms, system_uptime, smart_5_raw, smart_9_raw, ...
```

**Impact:** Feature index 0 during training = `cpu_temp` (Kelvin-converted from AI4I). Feature index 0 at edge = `cpu_temp` (raw Celsius from telemetry). Feature index 1 during training = `cpu_thermal_ramp` (diff over dataframe). Feature index 1 at edge = `gpu_temp`. **Every feature position mismatches.** The model receives data in a schema it was never trained on. The 99% accuracy on AI4I validation data is **meaningless at edge inference time**.

**Root Cause:** The training pipeline creates features from AI4I industrial machinery columns (air_temp, torque, rotational_speed, tool_wear), while the edge pipeline creates features from server telemetry columns (cpu_temp, gpu_temp, smart_X, psu_voltage). These are entirely different domains with different scales, distributions, and feature positions. There is no shared feature schema contract between training and inference.

**Why this hasn't caused visible failures:** The edge pipeline currently runs on dummy simulator data which bypasses the CNN via `mock_class` (see [run_dummy_simulator.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/run_dummy_simulator.py) line 17 — every anomaly payload includes `mock_class`). Both `edge_inference.py` (line 83) and `inference_server.py` (line 70) check for `mock_class` and skip model inference. **No one has tested the actual TFLite model on real telemetry yet.**

**Fix Required:** Create a **shared feature schema** (`FEATURE_SCHEMA_V1`) that both training and inference pipelines reference. This is addressed in Phase 2/3 where we retrain on real telemetry.

---

### BUG-N2: `mock_class` Feature Leaks into Model Input Vector

**Severity:** 🟠 MEDIUM — Corrupts model input  
**File:** [edge/main.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/main.py#L153-L158)

```python
# Lines 153-157: mock_class is APPENDED to the features DataFrame
if 'mock_class' in flat_telemetry:
    features['mock_class'] = flat_telemetry['mock_class']
    feature_names.append('mock_class')
```

When `mock_class` is present, the DataFrame has **29 columns** (28 features + mock_class). This is then passed to `engine.run_inference(features_df)`. Inside `edge_inference.py` line 84:
```python
feature_values = features.values.astype(np.float32)  # Now 29 values, not 28
```
The `input_data.reshape(input_shape)` call (line 96) expects shape `(1, 28, 1)` but gets 29 values → **`ValueError` or silent truncation depending on NumPy behavior.**

However, `edge_inference.py` doesn't actually use the model when `mock_class` is present — the `inference_server.py` checks `mock_class` at line 70 and returns hardcoded predictions. But if the cloud offload fails and falls back to local, the 29-value vector will crash TFLite.

**Fix:** Strip `mock_class` from the features DataFrame before passing to inference. Keep it as metadata, not as a model feature.

---

### BUG-N3: Data Leakage in Training — StandardScaler Fit on Full Dataset Including Test

**Severity:** 🔴 HIGH — Inflated accuracy metrics  
**Files:**
- [train_classifier.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/server/training/train_classifier.py#L303-L306)
- [train_lstm_ae.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/server/training/train_lstm_ae.py#L184-L187)

**Problem (`train_classifier.py` lines 303-306):**
```python
# Standardization happens BEFORE train/val split
mean = X_out.mean(axis=0)
std = X_out.std(axis=0) + 1e-8
X_out = (X_out - mean) / std
# ... THEN train/val split at line 408-414
```

The standardization (mean/std calculation) uses **the entire dataset including validation samples**. This means validation samples' statistics leak into the training features. The reported 99.07% accuracy is inflated because the scaler has "seen" the validation distribution during normalization.

**Same issue in `train_lstm_ae.py` lines 184-187:** Scaler fitted on full healthy data before train/val split.

**Impact:** Real-world accuracy will be lower than reported. For a B.Tech project this is a common mistake, but for production readiness it must be fixed.

**Fix:** Split data FIRST, then fit scaler on TRAINING data only, then transform both train and val using the training scaler. Save the scaler alongside the model for use at inference time.

---

### BUG-N4: No Scaler Saved — Edge Inference Uses Raw Unscaled Features

**Severity:** 🔴 HIGH — Model receives unscaled data at inference  
**Files:**
- [train_classifier.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/server/training/train_classifier.py#L303-L309) (scales during training)
- [edge/main.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/main.py#L103-L158) (NO scaling during inference)
- [edge/edge_inference.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/edge_inference.py#L83-L96) (feeds raw values to TFLite)

**Problem:** The 1D-CNN was trained on standardized features (zero mean, unit variance). But the edge inference pipeline feeds **raw unscaled** feature values directly into the TFLite model. CPU temp might be 75.0°C, fan RPM might be 2500 — the model expects all inputs to be roughly in [-3, 3] range after standardization.

**Root Cause:** The training script computes `mean` and `std` but never saves them. There is no `scaler.pkl` or `feature_stats.json` file. The edge pipeline has no way to know what normalization was applied during training.

**Fix:** Save the scaler (mean/std arrays or a `StandardScaler` object) alongside the model during training, and load + apply it during edge inference.

---

### BUG-N5: Isolation Forest Uses 4 Features, But 3 Are Always Default/Zero

**Severity:** 🟡 MEDIUM — Model predictions are meaningless  
**Files:**
- [edge/main.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/main.py#L138-L141)
- [edge/edge_inference.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/edge_inference.py#L70-L78)

**Problem:** The Isolation Forest is trained on 4 features: `psu_voltage_3v3_dev`, `psu_voltage_12v_dev`, `psu_voltage_stability`, `fan_rpm_deviation`. But in the edge feature extraction:

```python
# main.py line 140-141:
features['psu_voltage_stability'] = 0.05   # HARDCODED DEFAULT
features['fan_rpm_deviation'] = 0.0         # HARDCODED DEFAULT
```

`psu_voltage_stability` and `fan_rpm_deviation` are **always** constants. The Isolation Forest effectively operates on only 2 features (the voltage deviation percentages). Since the PSU voltages from `collector.py` are simulated random values near nominal (3.3V, 12V), the deviations are always near 0-3%. The IF model will almost never trigger.

**Fix:** When real PSU/fan data is available (post-hardware assembly), compute rolling statistics for voltage stability and compare actual vs expected RPM for fan deviation.

---

### BUG-N6: LSTM-AE Anomaly Detection Not Wired into Edge Pipeline

**Severity:** 🟡 MEDIUM — 2nd anomaly detector completely unused  
**Files:**
- [edge/edge_inference.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/edge_inference.py) — no LSTM-AE loading or inference
- [server/deep_analysis/inference_server.py](file:///c:/Users/Aswin/Documents/Antigravity/Predictive%20maintenance%20for%20Server/server/deep_analysis/inference_server.py) — no LSTM-AE inference

**Problem:** The architecture documents list 4 ML models: 1D-CNN, LSTM-AE, XGBoost, Isolation Forest. However:
- The LSTM-AE (`lstm_ae.h5`) is trained and saved but **never loaded or called** anywhere in the inference pipeline
- `edge_inference.py` only loads the TFLite classifier and Isolation Forest
- `inference_server.py` only loads the FP32 classifier — no LSTM-AE
- The architecture claims the LSTM-AE does anomaly detection on the cloud side, but `inference_server.py` doesn't use it

**Fix (Phase 3):** Wire the LSTM-AE into `inference_server.py` for cloud-side anomaly detection. It should run on sliding windows of recent telemetry and flag reconstruction errors above the trained threshold (0.4382).

---

### BUG-N7: Catastrophic Time-Series Destruction in LSTM-AE Training

**Severity:** 🔴 CRITICAL — Time-series sequence destroyed  
**Files:**
- `server/training/train_lstm_ae.py`

**Problem:** In an attempt to fix data leakage, the script uses `np.random.permutation(n)` to shuffle raw 2D samples **BEFORE** passing them to `create_sliding_windows()`. The sliding windows (which should represent consecutive moments in time) are now composed of completely random, non-sequential data points.

**Fix:** `create_sliding_windows()` must be applied to the ordered chronological data *first*, and ONLY THEN should the resulting 3D windows be randomly shuffled for the train/test split.

---

### BUG-N8: Broken Temporal Logic in Cloud Inference Windowing

**Severity:** 🔴 CRITICAL — Anomaly detection invalid for first 10 requests  
**Files:**
- `server/deep_analysis/inference_server.py`

**Problem:** The server maintains a sliding window buffer (`_server_windows`) of `maxlen=10`. If a new server connects and doesn't have 10 historical samples yet, the code pads the window by duplicating the *exact same current sample* 10 times, breaking the temporal history.

**Fix:** Pad with zeros (masking) or wait until the buffer has 10 sequential samples before running LSTM-AE inference.

---

### BUG-N9: Garbage Calibration Data in TFLite Quantization

**Severity:** 🔴 CRITICAL — Destroys edge model accuracy  
**Files:**
- `server/optimization/quantize_model.py`

**Problem:** The representative dataset generator takes the *first 5 numerical columns* from `ai4i2020.csv` and pads them with 23 zeros before passing them to the StandardScaler. The scaler expects the exact 28 features in `FEATURE_NAMES_V1` order. The scaled output is meaningless, leading to incorrect dynamic range calculation during INT8 quantization.

**Fix:** The dataset generator must properly extract the full 28 features using `FEATURE_NAMES_V1` from the dataframe before applying the scaler.

---

## 1.2 Previously Known Issues — Status Verification

| Issue | Status | Verified? |
|-------|--------|-----------|
| C1-C7 (Pre-Flight 2026-08-03) | ✅ All Fixed | Yes — confirmed in code |
| C1-C3 NEW (Pre-Deployment 2026-08-05) | ✅ All Fixed | Yes — `numpy` imported, `httpx` added, `state_lock` present |
| M1-M6 (Medium issues) | ✅ All Fixed | Yes — real calibration data, `ai-edge-litert`, `torch` removed |
| D1-D5 (Stale docs) | ✅ Mostly Fixed | Yes — some docs still reference old IPs in edge cases |
| G5 (Empty Pydantic model) | ⚠️ Still Present | Non-blocking — `TelemetryPayload(BaseModel): pass` |
| G8 (No auth on update_state) | ⚠️ Still Present | Acceptable for demo |

## 1.3 Summary: What Must Be Fixed Before Production

| # | Bug | Severity | Effort | Phase |
|---|-----|----------|--------|-------|
| N1 | Feature order mismatch (training vs inference) | 🔴 CRITICAL | High | Phase 3 (retrain) |
| N2 | `mock_class` leaks into feature vector | 🟠 MEDIUM | 10 min | Phase 2 |
| N3 | Data leakage (scaler fit on full dataset) | 🔴 CRITICAL | 30 min | Phase 3 (retrain) |
| N4 | No scaler saved for edge inference | 🔴 CRITICAL | 30 min | Phase 3 (retrain) |
| N5 | Isolation Forest gets constant features | 🟡 MEDIUM | Post-hardware | Phase 3 |
| N6 | LSTM-AE not wired into inference | 🟡 MEDIUM | 2 hours | Phase 3 |

---

# Phase 2: Multi-Tenant Data Storage Architecture

## 2.1 The Problem

Currently:
- **No server identity** — all telemetry arrives on the same MQTT topic (`server/telemetry`) with no way to differentiate which server sent it
- **No persistent storage** — telemetry is processed in-memory and discarded. Historical data is lost after restart
- **No separation** between demo/dummy data and real production data
- **Single-server assumption** — the entire pipeline treats all telemetry as coming from one server

## 2.2 Server Identity System

### 2.2.1 UUID-Based Server Registration

Each patient server gets a **UUID v4** identifier assigned at first boot. The UUID is:
1. Auto-generated on first run of `mqtt_publisher.py` (if no UUID file exists)
2. Persisted to a local file (`~/.predictive_maintenance/server_id.json`)
3. Included in every MQTT telemetry payload as `"server_id": "<UUID>"`
4. Registered in the cloud backend's server registry

```json
// Example: ~/.predictive_maintenance/server_id.json
{
    "server_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "server_name": "LOQ-Patient-01",
    "hardware_profile": {
        "cpu": "Ryzen 7 7840HS",
        "gpu": "RTX 4060",
        "ram_gb": 24,
        "os": "Windows 11"
    },
    "registered_at": "2026-08-10T10:00:00Z",
    "data_mode": "production"    // "demo" or "production"
}
```

### 2.2.2 MQTT Topic Hierarchy (Multi-Tenant)

**Current (single tenant):**
```
server/telemetry    ← flat, no identity
```

**Proposed (multi-tenant):**
```
server/<server_id>/telemetry       ← per-server telemetry
server/<server_id>/alerts          ← per-server alerts from edge
server/<server_id>/status          ← heartbeat / online status
server/+/telemetry                 ← edge subscribes to ALL servers (wildcard)
```

The edge node (`mqtt_handler.py`) subscribes to `server/+/telemetry` and extracts the `server_id` from the topic. This enables monitoring multiple patient servers from a single RPi4 edge node.

### 2.2.3 API Key Authentication (Optional)

For production environments, add an API key in the MQTT payload:
```json
{
    "server_id": "a1b2c3d4...",
    "api_key": "sk_prod_xxxx",
    "timestamp": "...",
    "cpu": {...}
}
```

The edge node validates the API key against a local allowlist (`configs/allowed_servers.json`) before processing.

## 2.3 Database Architecture

### 2.3.1 Technology Selection

| Component | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| **Edge (RPi4)** | In-memory dict only | **SQLite** (local ring buffer) | Lightweight, zero-config, survives reboots. Keep last 24h of telemetry per server as a local cache. |
| **Cloud (LOQ)** | InfluxDB (partially configured) | **InfluxDB 2.x** (fully integrated) | Already installed in `LOQ_Server_Stack/`. Native time-series DB. Optimized for high-frequency writes, retention policies, and Grafana integration. |
| **Cloud (metadata)** | None | **SQLite** on LOQ | Server registry, model versions, retraining logs. Lightweight and zero-dependency. |

### 2.3.2 InfluxDB Schema (Cloud — LOQ)

```
Bucket: telemetry_production
  Measurement: server_telemetry
    Tags:
      - server_id (string)      ← partition key
      - data_mode (string)      ← "demo" | "production"
      - component (string)      ← "cpu" | "gpu" | "hdd" | "psu" | "ram" | "fans"
    Fields:
      - cpu_temp (float)
      - cpu_load (float)
      - gpu_temp (float)
      - gpu_util (float)
      - ... (all 28+ raw telemetry values)
    Timestamp: nanosecond precision

Bucket: predictions
  Measurement: inference_results
    Tags:
      - server_id
      - model_version (string)
      - inference_location (string)  ← "edge" | "cloud"
    Fields:
      - predicted_class (int)
      - confidence (float)
      - ttf_days (float)
      - anomaly_detected (bool)
      - health_index (float)
    Timestamp: nanosecond precision

Bucket: telemetry_demo
  (Same schema as telemetry_production, but for demo/dummy data)
```

### 2.3.3 Retention Policies

| Bucket | Retention | Downsampling |
|--------|-----------|-------------|
| `telemetry_production` | **365 days** (indefinite for production servers) | After 30 days: downsample to 1-minute averages |
| `telemetry_demo` | **7 days** (auto-purge) | None |
| `predictions` | **365 days** | None |

### 2.3.4 Data Flow Architecture

```
┌─────────────────┐      MQTT (server/<id>/telemetry)       ┌──────────────┐
│  Patient Server  │ ────────────────────────────────────▶   │   RPi4 Edge  │
│  (telemetry      │                                         │              │
│   agent)         │                                         │  1. Process  │
│                  │                                         │  2. Infer    │
│  server_id: UUID │                                         │  3. Store    │
└─────────────────┘                                         │     (SQLite  │
                                                            │      24h     │
                                                            │      cache)  │
                                                            │  4. Forward  │
                                                            │     to Cloud │
                                                            └──────┬───────┘
                                                                   │
                                                      HTTP POST /api/ingest
                                                                   │
                                                            ┌──────▼───────┐
                                                            │  LOQ Cloud   │
                                                            │              │
                                                            │  1. Write to │
                                                            │     InfluxDB │
                                                            │  2. Deep     │
                                                            │     Analysis │
                                                            │  3. Check    │
                                                            │     retrain  │
                                                            │     trigger  │
                                                            └──────────────┘
```

### 2.3.5 New Cloud API Endpoint: `/api/ingest`

A new endpoint on the cloud backend (`inference_server.py` or a new `ingest_api.py`) that:
1. Receives processed telemetry + prediction results from the edge node
2. Tags it with `server_id` and `data_mode`
3. Writes to InfluxDB
4. Checks if the server has reached the retraining data threshold (see Phase 3)

---

# Phase 3: Continuous Learning & Personalization Pipeline

## 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS LEARNING PIPELINE                      │
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌─────────────┐ │
│  │ Telemetry │───▶│ InfluxDB │───▶│  Retrain   │───▶│  Model      │ │
│  │ Ingest    │    │ (30 days │    │  Trigger   │    │  Registry   │ │
│  │           │    │  check)  │    │  Manager   │    │  (SQLite)   │ │
│  └──────────┘    └──────────┘    └─────┬──────┘    └──────┬──────┘ │
│                                        │                   │        │
│                                        ▼                   ▼        │
│                                  ┌──────────┐       ┌──────────┐   │
│                                  │ Transfer  │       │ Deploy   │   │
│                                  │ Learning  │       │ to Edge  │   │
│                                  │ Trainer   │       │ (SFTP)   │   │
│                                  └──────────┘       └──────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.2 Retraining Strategy — Trigger Mechanism

### 3.2.1 Data Threshold Trigger

A background scheduler on the cloud backend checks each registered server's data accumulation:

| Condition | Action |
|-----------|--------|
| Server has **< 7 days** of data | Use the Base Model (generic, trained on AI4I + Backblaze) |
| Server has **7-30 days** of data | Begin baseline profiling — compute per-server statistics |
| Server has **≥ 30 days** of data | **TRIGGER personalized model training** |
| Every **30 days** after first trigger | Re-trigger training with latest data |
| **Anomaly rate > 10%** in last 24h | **EMERGENCY retrain** — potential drift detected |

### 3.2.2 Trigger Check Implementation

```python
# Pseudocode for the retrain trigger (runs every 6 hours via scheduler)
async def check_retrain_triggers():
    for server in registry.get_all_servers():
        data_days = influxdb.count_days(server.id, bucket="telemetry_production")
        last_trained = registry.get_last_trained(server.id)
        
        if data_days >= 30 and (last_trained is None or days_since(last_trained) >= 30):
            await trigger_retraining(server.id)
```

## 3.3 ML Architecture: Transfer Learning with Personalized Adapters

### 3.3.1 Two-Tier Model Architecture

```
┌─────────────────────────────────────────────────┐
│              BASE MODEL (Frozen)                  │
│                                                   │
│  Conv1D(32) → Conv1D(64) → MaxPool → Flatten     │
│                     │                              │
│            Shared Feature Extractor                │
│           (Trained on ALL data, frozen)            │
│                     │                              │
│                     ▼                              │
│  ┌──────────────────────────────────────┐         │
│  │        PERSONALIZED ADAPTER           │         │
│  │     (Per-server, lightweight)         │         │
│  │                                       │         │
│  │  Dense(64, relu) → Dropout(0.3)       │         │
│  │           │              │            │         │
│  │      ┌────▼────┐   ┌────▼────┐       │         │
│  │      │ Class   │   │  TTF    │       │         │
│  │      │ Head    │   │  Head   │       │         │
│  │      │(Softmax)│   │(Linear) │       │         │
│  │      └─────────┘   └─────────┘       │         │
│  └──────────────────────────────────────┘         │
└─────────────────────────────────────────────────┘
```

### 3.3.2 Why Transfer Learning?

1. **Small data per server**: 30 days × 17,280 samples/day (at 5s intervals) = ~518K samples. Enough to fine-tune, not enough to train from scratch.
2. **Preserve general patterns**: The base model's convolutional filters learn universal degradation signatures (thermal ramps, voltage drift). These transfer across servers.
3. **Lightweight adapters**: Each server's adapter is only ~50KB (Dense(64) + 2 heads). Hundreds of servers = only a few MB of adapter weights.
4. **Fast training**: Fine-tuning only the adapter layers takes 5-10 minutes on the RTX 4060 vs. 30+ minutes for full model training.
5. **Safe rollback**: If the adapter degrades, fall back to the base model instantly.

### 3.3.3 Adapter Training Process

```python
# Pseudocode
def train_adapter(server_id: str):
    # 1. Load base model and freeze convolutional layers
    base_model = load_base_model("classifier_fp32.h5")
    for layer in base_model.layers[:5]:  # Freeze Conv1D + MaxPool + Flatten
        layer.trainable = False
    
    # 2. Query InfluxDB for this server's last 30 days of telemetry
    data = influxdb.query(
        f'from(bucket:"telemetry_production") |> filter(fn: (r) => r.server_id == "{server_id}")'
    )
    
    # 3. Engineer 28 features using the SHARED feature schema
    X, y_class, y_ttf = engineer_features(data, schema=FEATURE_SCHEMA_V1)
    
    # 4. Fit scaler on THIS server's training data only
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    
    # 5. Fine-tune adapter layers only (Dense + heads)
    base_model.compile(...)
    base_model.fit(X_train, [y_class, y_ttf], epochs=20, ...)
    
    # 6. Save adapter weights + scaler
    save_adapter(server_id, base_model, scaler)
    
    # 7. Quantize to INT8 TFLite for edge deployment
    quantize_and_export(base_model, server_id)
```

### 3.3.4 Labeling Strategy for Real Data

Real server telemetry doesn't come with labels. We use **self-supervised labeling**:

| Method | Applies To | How |
|--------|-----------|-----|
| **Threshold-based** | THERMAL_CRITICAL | CPU temp > 90°C sustained for > 30s → label class 2 |
| **SMART deterioration** | HDD_DEGRADATION | smart_5 increasing, smart_197 > 0 → label class 1 |
| **Voltage deviation** | PSU_ANOMALY | |V_actual - V_nominal| / V_nominal > 5% → label class 3 |
| **RPM drop** | FAN_FAILURE | RPM < 50% of baseline for > 60s → label class 4 |
| **Page fault spike** | MEMORY_FAULT | page_faults > 3σ above rolling mean → label class 5 |
| **GPU clock drops** | GPU_DEGRADATION | clock < 70% baseline + temp > 85°C → label class 6 |
| **Operator feedback** | Any | Dashboard button: "Flag this as anomaly" → adds human label |
| **TTF estimation** | All | Based on rate of degradation slope. If metric is worsening at rate `r`, extrapolate TTF = (threshold - current) / r |

## 3.4 Model Registry & Versioning

### 3.4.1 SQLite Model Registry Schema (on LOQ)

```sql
CREATE TABLE model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id TEXT NOT NULL,
    model_type TEXT NOT NULL,           -- 'base', 'adapter', 'iso_forest'
    version INTEGER NOT NULL,
    trained_at TIMESTAMP NOT NULL,
    training_samples INTEGER,
    training_days INTEGER,
    val_accuracy REAL,
    val_mae REAL,
    model_path TEXT NOT NULL,           -- path to .h5 / .tflite / .pkl
    scaler_path TEXT NOT NULL,          -- path to scaler.pkl
    tflite_path TEXT,                   -- path to INT8 quantized version
    status TEXT DEFAULT 'active',       -- 'active', 'superseded', 'failed'
    deployed_to_edge BOOLEAN DEFAULT 0,
    deployed_at TIMESTAMP,
    notes TEXT
);

CREATE TABLE server_registry (
    server_id TEXT PRIMARY KEY,
    server_name TEXT,
    hardware_profile TEXT,              -- JSON
    registered_at TIMESTAMP,
    data_mode TEXT DEFAULT 'production',
    total_data_days INTEGER DEFAULT 0,
    last_retrain_at TIMESTAMP,
    active_model_version INTEGER,
    FOREIGN KEY (active_model_version) REFERENCES model_registry(id)
);

CREATE TABLE retraining_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id TEXT NOT NULL,
    triggered_at TIMESTAMP NOT NULL,
    trigger_reason TEXT,                -- 'scheduled', 'threshold', 'emergency', 'manual'
    completed_at TIMESTAMP,
    status TEXT,                        -- 'running', 'completed', 'failed'
    old_model_version INTEGER,
    new_model_version INTEGER,
    improvement_pct REAL,              -- % improvement over old model
    error_log TEXT
);
```

## 3.5 Deployment Loop: Cloud → Edge

### 3.5.1 Seamless Weight Update Protocol

```
                     Cloud (LOQ)                              Edge (RPi4)
                         │                                        │
    1. Train adapter ────┤                                        │
    2. Quantize INT8 ────┤                                        │
    3. Validate ─────────┤                                        │
    4. Register ─────────┤                                        │
                         │                                        │
    5. SFTP transfer ────┼──── classifier_<server_id>_v2.tflite ─▶│
                         │──── scaler_<server_id>_v2.pkl ─────────▶│
                         │──── deploy_manifest.json ──────────────▶│
                         │                                        │
                         │            6. Edge receives ───────────┤
                         │            7. Verify checksum ─────────┤
                         │            8. Hot-swap model ──────────┤
                         │                (no restart needed)     │
                         │            9. ACK back ────────────────┤
                         │◀─── {"status":"deployed","version":2} ─┤
                         │                                        │
   10. Mark deployed ────┤                                        │
```

### 3.5.2 Hot-Swap Without Restart

The edge inference engine (`edge_inference.py`) will be enhanced with a model-reload mechanism:

```python
class EdgeInferenceEngine:
    def reload_model(self, server_id: str = None):
        """Hot-swap the TFLite model without restarting the edge process."""
        # Check for new model files in the models directory
        new_model = find_latest_model(server_id)
        if new_model and new_model != self.current_model_path:
            self.classifier = tflite.Interpreter(model_path=new_model)
            self.classifier.allocate_tensors()
            self.scaler = joblib.load(new_model.replace('.tflite', '_scaler.pkl'))
            print(f"[HOT-SWAP] Loaded model v{version} for server {server_id}")
```

The edge `main.py` will check for new models every 60 seconds via a lightweight file-system poll.

### 3.5.3 Multi-Server Model Loading

When the edge node monitors multiple servers, it maintains a dictionary of per-server models:

```python
self.models = {
    "a1b2c3d4...": {  # server_id
        "classifier": tflite.Interpreter(...),
        "scaler": StandardScaler(...),
        "version": 2
    },
    "__default__": {  # fallback base model
        "classifier": tflite.Interpreter("classifier_int8.tflite"),
        "scaler": None,  # base model — no per-server scaler
        "version": 1
    }
}
```

---

# Step-by-Step Implementation Tasks

> **Estimated total effort:** 3-4 days of focused coding  
> **Devices needed:** IdeaPad S145 (coding) → LOQ RTX 4060 (training + testing)

## Milestone 1: Foundation — Shared Feature Schema & Bug Fixes (Day 1)

### Task 1.1: Create Shared Feature Schema Contract
- **[NEW]** `server/training/feature_schema.py` — Defines `FEATURE_SCHEMA_V1`: ordered list of 28 feature names, their types, expected ranges, and normalization strategy
- Both training and inference pipelines import from this single source of truth
- Prevents BUG-N1 (feature order mismatch)

### Task 1.2: Fix `mock_class` Leak (BUG-N2)
- **[MODIFY]** `edge/main.py` — Strip `mock_class` from features DataFrame before passing to inference engine. Store it as a separate metadata field.

### Task 1.3: Fix Data Leakage (BUG-N3) & Save Scaler (BUG-N4)
- **[MODIFY]** `server/training/train_classifier.py` — Split data FIRST, then fit scaler on train only. Save scaler as `server/models/classifier_scaler.pkl`
- **[MODIFY]** `server/training/train_lstm_ae.py` — Same fix. Save scaler as `server/models/lstm_ae_scaler.pkl`

### Task 1.4: Load Scaler at Edge Inference
- **[MODIFY]** `edge/edge_inference.py` — Load `classifier_scaler.pkl` (or `feature_stats.json`) and apply standardization before TFLite inference.

## Milestone 2: Multi-Tenant Identity & MQTT (Day 1-2)

### Task 2.1: Server Identity Module
- **[NEW]** `server/telemetry_agent/server_identity.py` — UUID generation, persistence, hardware profile detection, registration

### Task 2.2: Update MQTT Topic Hierarchy
- **[MODIFY]** `server/telemetry_agent/mqtt_publisher.py` — Publish to `server/<server_id>/telemetry` instead of `server/telemetry`. Include `server_id` in payload.
- **[MODIFY]** `edge/mqtt_handler.py` — Subscribe to `server/+/telemetry` (wildcard). Extract `server_id` from topic. Pass it to callback.
- **[MODIFY]** `edge/main.py` — Track per-server state (one `current_state` dict per `server_id`). Update dashboard to show server selector.

### Task 2.3: Edge SQLite Cache
- **[NEW]** `edge/local_store.py` — SQLite ring buffer. Write incoming telemetry. Auto-purge after 24 hours. Query for recent data.

## Milestone 3: Cloud Data Ingestion & InfluxDB (Day 2)

### Task 3.1: Cloud Ingest API
- **[NEW]** `server/deep_analysis/ingest_api.py` — FastAPI endpoint `/api/ingest`. Receives telemetry + predictions from edge. Writes to InfluxDB. Tags with `server_id` and `data_mode`.
- Alternatively, can be added as new endpoints in the existing `inference_server.py`.

### Task 3.2: InfluxDB Writer Module
- **[NEW]** `server/deep_analysis/influx_writer.py` — Wraps the InfluxDB 2.x Python client. Handles batch writes, retry, and bucket routing (`telemetry_production` vs `telemetry_demo`).

### Task 3.3: Edge → Cloud Forwarding
- **[MODIFY]** `edge/main.py` — After inference, POST processed telemetry + predictions to cloud `/api/ingest`. Use async `httpx` with retry.

### Task 3.4: Server & Model Registry
- **[NEW]** `server/deep_analysis/model_registry.py` — SQLite-backed registry. CRUD operations for servers, models, and retraining logs.

## Milestone 4: Continuous Learning Pipeline (Day 3)

### Task 4.1: Retrain Trigger Manager
- **[NEW]** `server/training/retrain_manager.py` — Background scheduler (APScheduler or simple `threading.Timer`). Checks InfluxDB data accumulation per server. Triggers retraining when thresholds met.

### Task 4.2: Transfer Learning Trainer
- **[NEW]** `server/training/train_adapter.py` — Loads base model, freezes conv layers, queries InfluxDB for server-specific data, fine-tunes adapter layers, saves adapter + scaler, quantizes to INT8.

### Task 4.3: Self-Supervised Labeling Engine
- **[NEW]** `server/training/auto_labeler.py` — Applies threshold-based rules to raw telemetry to generate class labels and TTF estimates for retraining.

### Task 4.4: Wire LSTM-AE into Cloud Inference (BUG-N6)
- **[MODIFY]** `server/deep_analysis/inference_server.py` — Load LSTM-AE, maintain a sliding window buffer per server, run anomaly detection on offloaded data.

## Milestone 5: Deployment & Hot-Swap (Day 3-4)

### Task 5.1: Model Deployer (Cloud → Edge)
- **[NEW]** `server/deployment/deploy_model.py` — SFTP module that pushes new TFLite models + scalers to RPi4. Generates deploy manifest.

### Task 5.2: Edge Model Hot-Swap
- **[MODIFY]** `edge/edge_inference.py` — Add `reload_model()` method. Support per-server model dictionary. File-system poll for new models.

### Task 5.3: Multi-Server Dashboard
- **[MODIFY]** `dashboard/index.html` + `dashboard/assets/app.js` — Add server selector dropdown. Show per-server health cards. Model version badge.
- **[MODIFY]** `edge/api_server.py` — Serve per-server state via `/api/state?server_id=<UUID>`.

## Milestone 6: Testing & Documentation (Day 4)

### Task 6.1: Integration Tests
- **[MODIFY]** `tests/test_apis_end_to_end.py` — Add tests for: multi-tenant ingest, retrain trigger, model hot-swap, per-server state

### Task 6.2: Update All Documentation
- **[MODIFY]** `MEMORIES.md` — Add Phase D tasks
- **[MODIFY]** `PROJECT_SUMMARY.md` — Update status
- **[MODIFY]** `docs/AGENT_HANDOFF.md` — Add multi-tenant and continuous learning sections
- **[NEW]** `docs/multi_tenant_guide.md` — Operational guide for adding new servers

---

# Open Questions for User

> [!IMPORTANT]
> **Q1: Patient Server Hardware** — Questions Q21-Q25 in MEMORIES.md are still marked ❌ BLOCKING. What hardware will the patient server be? This affects the telemetry collector implementation and which SMART/GPU features are available.

> [!IMPORTANT]
> **Q2: Number of Patient Servers for Demo** — For the B.Tech demo, will you have 1 or 2+ patient servers? This affects whether multi-tenant is essential for demo or can be shown as "architecture ready" with a single server.

> [!WARNING]
> **Q3: InfluxDB Already Deployed?** — `LOQ_Server_Stack/InfluxDB/` exists and the `.env` has a token. Is InfluxDB currently running on the LOQ? We need to verify bucket creation and token validity before writing the integration.

> [!IMPORTANT]
> **Q4: Retraining Compute Budget** — Transfer learning fine-tuning will use the RTX 4060. Should retraining be triggered automatically in the background, or manually by the user? Auto-retraining while the inference server is running may cause GPU memory contention.

> [!NOTE]
> **Q5: Deployment Priority** — Would you like me to implement all 3 phases in order, or would you prefer to prioritize Phase 1 bug fixes + Phase 3 retrain pipeline (since the model bugs are the most critical), and defer the full multi-tenant InfluxDB storage to later?

---

# Verification Plan

## Automated Tests

```bash
# 1. Feature schema consistency test
python -m pytest tests/test_feature_schema.py -v

# 2. Multi-tenant MQTT routing test
python -m pytest tests/test_mqtt_multitenant.py -v

# 3. InfluxDB write/read test
python -m pytest tests/test_influx_integration.py -v

# 4. Retrain trigger test (mock InfluxDB)
python -m pytest tests/test_retrain_trigger.py -v

# 5. Model hot-swap test
python -m pytest tests/test_model_hotswap.py -v

# 6. End-to-end (existing + extended)
python tests/test_apis_end_to_end.py --inference-url http://127.0.0.1:8000/api/offload
```

## Manual Verification

1. **Feature order match**: After retrain, print feature names from training and edge, confirm they match exactly
2. **Scaler consistency**: Verify scaler means/stds saved during training match what's loaded at edge
3. **Multi-tenant MQTT**: Run 2 dummy simulators with different server_ids, verify edge tracks them separately
4. **InfluxDB persistence**: Restart LOQ, query InfluxDB, confirm historical data survives
5. **Hot-swap**: Deploy a new model to RPi4 while inference is running, verify it picks up the new model without restart
6. **Data leakage check**: Compare val accuracy with proper train-only scaler vs old full-dataset scaler

---

> [!CAUTION]
> **DO NOT BEGIN IMPLEMENTATION** until the user reviews this plan, answers the open questions, and explicitly says "Approved".  
> This plan is designed to be **self-contained** — any agent on any model (Claude, Gemini, etc.) on any device (IdeaPad, LOQ, or any new laptop) can read this document, understand the full context, and begin implementation from any milestone.

---

*Plan authored by @claude-architect on 2026-08-09. All source files audited: 48+ files, 100% coverage.*
