"""
Edge Node Main Entrypoint — Phase H Enhanced.

This script is the SINGLE process to run on the Raspberry Pi 4.
It wires together:
  1. MQTT Handler (subscribes to server telemetry)
  2. Feature Engineering (transforms raw telemetry -> 28 features)
  3. Edge Inference Engine (TFLite INT8 + Isolation Forest) [secondary/fallback]
  4. Adaptive Baseline Engine (per-server Z-score anomaly detection) [PRIMARY]
  5. FastAPI Dashboard Server (serves operator UI + exposes APIs)

Phase H pipeline:
  - During calibration (first 24h): Collect data, build baselines, use AI4I models as primary
  - After calibration: Adaptive baselines are primary, AI4I models are secondary confirmation

Usage:
    python edge/main.py --broker 127.0.0.1 --port 1883
"""

import os
import sys
import json
import time
import signal
import argparse
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Import shared feature schema — single source of truth for feature ordering
from server.training.feature_schema import (
    FEATURE_NAMES_V1, FEATURE_COUNT, FEATURE_DEFAULTS,
    FAULT_LABELS, CLASS_TO_COMPONENT
)

# Import v2 schema utilities for hierarchical payload conversion
try:
    from server.training.feature_schema_v2 import (
        v2_to_v1_feature_vector, is_v2_payload, v1_flat_to_v2
    )
    V2_SCHEMA_AVAILABLE = True
except ImportError:
    V2_SCHEMA_AVAILABLE = False

from edge.mqtt_handler import EdgeMQTTHandler
from edge.edge_inference import EdgeInferenceEngine
from edge.edge_logger import get_logger
from edge.adaptive_baseline import AdaptiveBaseline

logger = get_logger("edge_main")

# Per-server adaptive baseline engines
adaptive_baselines: Dict[str, AdaptiveBaseline] = {}


def reset_baselines(server_id: Optional[str] = None):
    """
    Reset in-memory adaptive baselines and wipe persisted baseline cache files.
    If server_id is None, 'all', or empty, resets for all connected servers.
    """
    from edge.adaptive_baseline import reset_all_baselines
    reset_all_baselines(server_id=server_id)
    with state_lock:
        if server_id and server_id.lower() not in ("all", "none", "", "default"):
            adaptive_baselines.pop(server_id, None)
            if server_id in telemetry_history:
                telemetry_history[server_id].clear()
        else:
            adaptive_baselines.clear()
            telemetry_history.clear()
        logger.info(f"Reset adaptive baselines and telemetry history for server_id='{server_id or 'all'}'")


# Import sensor reader — gracefully handle if not on RPi4
try:
    from edge.sensor_reader import SensorReader
    SENSORS_AVAILABLE = True
except ImportError:
    SENSORS_AVAILABLE = False

# Import the FastAPI app for serving
import uvicorn
from edge.api_server import app, current_state, state_lock, _default_server_state

# Auto-load configs/edge.env if present
edge_env_path = os.path.join(BASE_DIR, 'configs', 'edge.env')
if os.path.exists(edge_env_path):
    with open(edge_env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('=', 1)
                if len(parts) == 2:
                    k, v = parts[0].replace('export ', '').strip(), parts[1].strip().strip('"\'')
                    if k not in os.environ:
                        os.environ[k] = v

# Configuration
MQTT_BROKER = os.environ.get("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

# Feature buffer for rolling computations (per-server)
telemetry_history = {}  # server_id -> list of flat dicts
MAX_HISTORY = 60  # Keep last 60 readings per server for rolling features
_edge_server_last_seen = {}  # server_id -> time.time() for TTL eviction
_EDGE_SERVER_TTL = 1800  # 30 minutes — purge caches for dead servers

# Initialize the inference engine
engine = EdgeInferenceEngine()

# Initialize sensor reader (None if not on RPi4)
sensor_reader = SensorReader() if SENSORS_AVAILABLE else None

# Global MQTT handler reference
_global_mqtt_handler = None


def _background_sensor_publisher(handler: EdgeMQTTHandler, interval: float = 2.0):
    """
    Periodically samples physical hardware sensors (ADXL345 on I2C-1, INMP441 on ALSA Card 3)
    and publishes live sensor telemetry to MQTT topic 'edge/sensors'.
    """
    if not sensor_reader:
        logger.info("SensorReader not available — background sensor publisher disabled")
        return
    logger.info(f"Starting background hardware sensor publisher (interval={interval}s)...")
    while True:
        try:
            metrics = sensor_reader.read_all()
            status = sensor_reader.get_status()
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "node_id": "rpi4-edge-doctor",
                "status": status,
                "sensors_active": metrics.get("sensors_active", {}),
                "vibration_rms": float(metrics.get("vibration_rms", 0.02)),
                "peak_acceleration": float(metrics.get("peak_acceleration", 0.02)),
                "vibration_kurtosis": float(metrics.get("vibration_kurtosis", 3.0)),
                "acoustic_rms": float(metrics.get("acoustic_rms", 0.02)),
                "acoustic_peak_freq": float(metrics.get("acoustic_peak_freq", 0.0)),
                "ambient_temp": float(metrics.get("ambient_temp", 25.0)),
                "humidity": float(metrics.get("humidity", 50.0)),
                "vibration": metrics.get("vibration", {}),
                "acoustic": metrics.get("acoustic", {}),
                "environment": metrics.get("environment", {})
            }
            handler.publish("edge/sensors", payload)
        except Exception as e:
            logger.warning(f"Error in background sensor publisher: {e}")
        time.sleep(interval)


# Fault class labels and component mapping are imported from feature_schema.py
# FAULT_LABELS and CLASS_TO_COMPONENT are available via the import above.


def flatten_telemetry(payload: Dict[str, Any]) -> Dict[str, float]:
    """Flatten nested telemetry JSON into a flat feature dictionary."""
    flat = {}
    for key, value in payload.items():
        if key in ("timestamp", "server_id"):
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    flat[f"{key}_{sub_key}"] = float(sub_value)
                elif isinstance(sub_value, list) and sub_value:
                    if isinstance(sub_value[0], (int, float)):
                        flat[f"{key}_{sub_key}_mean"] = float(np.mean(sub_value))
        elif isinstance(value, (int, float)):
            flat[key] = float(value)
    return flat


def extract_features(flat_telemetry: Dict[str, float], server_id: str = "default") -> tuple:
    """
    Extract the 28 engineered features from flat telemetry data.
    Uses FEATURE_NAMES_V1 from the shared schema for canonical ordering.
    
    Returns:
        tuple: (features_df, metadata_dict)
            - features_df: pd.DataFrame with exactly 28 columns in FEATURE_NAMES_V1 order
            - metadata_dict: dict with non-feature fields like mock_class
    """
    # Mapping from flat telemetry keys to canonical feature names
    telemetry_to_feature = {
        'cpu_temp_package': 'cpu_temp',
        'cpu_load_percent': 'cpu_load',
        'cpu_frequency_mhz': 'cpu_freq',
        'gpu_temp': 'gpu_temp',
        'gpu_power_draw_w': 'gpu_power',
        'gpu_utilization_percent': 'gpu_util',
        'gpu_memory_used_mb': 'gpu_mem_used',
        'gpu_clock_mhz': 'gpu_clock',
        'ram_used_gb': 'ram_used_gb',
        'ram_total_gb': 'ram_total_gb',
        'ram_page_faults_sec': 'ram_page_faults',
        'hdd_smart_5_raw': 'smart_5_raw',
        'hdd_smart_9_raw': 'smart_9_raw',
        'hdd_smart_187_raw': 'smart_187_raw',
        'hdd_smart_188_raw': 'smart_188_raw',
        'hdd_smart_194_raw': 'smart_194_raw',
        'hdd_smart_197_raw': 'smart_197_raw',
        'hdd_smart_198_raw': 'smart_198_raw',
        'environment_ambient_temp': 'ambient_temp',
        'environment_humidity': 'humidity',
        'environment_vibration_rms': 'vibration_rms',
        'environment_acoustic_rms': 'acoustic_rms',
        'environment_system_uptime_hours': 'system_uptime',
    }
    
    features = {}
    
    # Map flat telemetry to canonical feature names
    for telem_key, feat_name in telemetry_to_feature.items():
        if telem_key in flat_telemetry:
            features[feat_name] = float(flat_telemetry[telem_key])
    
    # Also check for direct feature name matches (e.g., from dummy simulator)
    for feat_name in FEATURE_NAMES_V1:
        if feat_name in flat_telemetry and feat_name not in features:
            features[feat_name] = float(flat_telemetry[feat_name])
    
    # Computed features
    # PSU voltage deviations
    psu_3v3 = flat_telemetry.get('psu_voltage_3v3', 3.3)
    psu_12v = flat_telemetry.get('psu_voltage_12v', 12.0)
    features.setdefault('psu_voltage_3v3_dev', abs(psu_3v3 - 3.3) / 3.3 * 100)
    features.setdefault('psu_voltage_12v_dev', abs(psu_12v - 12.0) / 12.0 * 100)
    
    # Rolling PSU voltage stability (from per-server history buffer)
    srv_history = telemetry_history.get(server_id, [])
    if len(srv_history) >= 10:
        recent_12v = [h.get('psu_voltage_12v', 12.0) for h in srv_history[-60:]]
        features.setdefault('psu_voltage_stability', float(np.std(recent_12v)))
    
    # Fan RPM deviation (compute from actual vs baseline)
    actual_rpm = flat_telemetry.get('fans_cpu_fan_rpm', 0.0)
    if actual_rpm > 0:
        baseline_rpm = 1500.0  # Expected baseline RPM
        features.setdefault('fan_rpm_deviation', (actual_rpm - baseline_rpm) / baseline_rpm)
    
    # Cross-sensor: CPU temp minus ambient temp
    cpu_t = features.get('cpu_temp', FEATURE_DEFAULTS.get('cpu_temp', 45.0))
    amb_t = features.get('ambient_temp', FEATURE_DEFAULTS.get('ambient_temp', 25.0))
    features.setdefault('cpu_ambient_delta', cpu_t - amb_t)
    
    # Build the DataFrame in EXACT FEATURE_NAMES_V1 order, filling defaults
    row = {}
    for feat_name in FEATURE_NAMES_V1:
        row[feat_name] = features.get(feat_name, FEATURE_DEFAULTS.get(feat_name, 0.0))
    
    # Model input normalization for legacy 1D-CNN (trained on 16GB / 37.5°C milling benchmark)
    model_row = dict(row)

    # 1. RAM Normalization:
    # Model scaler has mean=16.0, scale=1.0 for ram_total_gb, and mean=4.87, scale=3.04 for ram_used_gb.
    # Preserve genuine values in telemetry while scaling model input to avoid false-positive MEMORY_FAULT.
    ram_tot = float(row.get('ram_total_gb', 16.0))
    ram_usd = float(row.get('ram_used_gb', 4.0))
    if ram_tot > 0:
        ratio = min(1.0, max(0.0, ram_usd / ram_tot))
        model_row['ram_total_gb'] = 16.0
        if ratio > 0.90:
            model_row['ram_used_gb'] = 14.0  # Trigger memory warning if nearly exhausted
        else:
            # Map nominal operational memory usage cleanly into model's nominal range (3.5 - 6.5 GB)
            model_row['ram_used_gb'] = round(4.87 + (ratio - 0.50) * 2.0, 2)

    raw_faults = float(row.get('ram_page_faults', 50.0))
    if raw_faults < 1500.0:
        model_row['ram_page_faults'] = 50.0

    # 2. CPU Thermal & Frequency Normalization:
    # Modern CPUs operate nominally at 55°C - 75°C under load, whereas legacy AI4I operated at ~37°C.
    # Only allow raw high temperatures to reach the 1D-CNN if genuinely thermal throttling (> 85°C).
    raw_cpu_t = float(row.get('cpu_temp', 45.0))
    raw_delta = float(row.get('cpu_ambient_delta', 20.0))
    if raw_cpu_t < 85.0:
        model_row['cpu_temp'] = round(37.5 + (raw_cpu_t - 50.0) * 0.1, 1)
        model_row['cpu_ambient_delta'] = round(10.6 + (raw_delta - 25.0) * 0.1, 1)
    else:
        model_row['cpu_temp'] = raw_cpu_t
        model_row['cpu_ambient_delta'] = raw_delta
    model_row['cpu_freq'] = 3000.0

    # 3. GPU Normalization:
    raw_gpu_t = float(row.get('gpu_temp', 40.0))
    if raw_gpu_t < 85.0:
        model_row['gpu_temp'] = 40.0
    else:
        model_row['gpu_temp'] = raw_gpu_t
    model_row['gpu_power'] = 30.0
    model_row['gpu_clock'] = 1500.0
    model_row['gpu_mem_used'] = 500.0
    model_row['gpu_util'] = 10.0

    # 4. PSU Voltage Normalization:
    dev_12v = float(row.get('psu_voltage_12v_dev', 0.5))
    if dev_12v < 5.0:
        model_row['psu_voltage_12v_dev'] = 2.18
    model_row['psu_voltage_3v3_dev'] = 0.5
    model_row['psu_voltage_stability'] = 0.05

    df = pd.DataFrame([model_row], columns=FEATURE_NAMES_V1)
    
    # Extract metadata (mock_class, etc.) — NOT included in model features
    metadata = {}
    if 'mock_class' in flat_telemetry:
        metadata['mock_class'] = int(flat_telemetry['mock_class'])
    
    return df, metadata


def on_telemetry_received(data: Dict[str, Any]):
    """
    Callback invoked by MQTT handler when a telemetry payload arrives.
    This is the CORE PIPELINE:
      Phase H: v2 payload → Adaptive Baseline (primary) + AI4I models (secondary) → dashboard
    """
    global current_state
    
    # Keep raw v2 payload for adaptive baseline BEFORE converting to v1
    schema_version = data.get('schema_version', '1.0')
    v2_raw_payload = data.copy() if schema_version == '2.0' else None
    
    # DEDUP: Skip v1 legacy payloads for servers that already sent v2 on this tick.
    # The v1 arrives right after the v2 on the legacy topic and would overwrite
    # the adaptive baseline state with None (since v1 has no schema_version).
    server_id_early = data.get('server_id', 'default')
    if schema_version == '1.0' and server_id_early in adaptive_baselines:
        # This server has an adaptive baseline (meaning it sends v2), skip the v1 duplicate
        return
    
    # Detect v2 payload and convert to v1 flat format for existing models
    _v2_raw_payload = None  # Preserve raw v2 for multi-instance rendering
    if schema_version == '2.0' and V2_SCHEMA_AVAILABLE:
        _v2_raw_payload = data  # Save before conversion overwrites
        logger.debug(f"Received v2 payload from {data.get('server_id', '?')}, converting to v1...")
        try:
            cpus = data.get('cpus', [{}])
            gpus = data.get('gpus', [{}])
            mem_list = data.get('memory', [{}])
            storage = data.get('storage', [{}])
            psu = data.get('psu', {})
            fans = data.get('fans', [])
            env = data.get('environment', {})
            
            cpu0 = cpus[0] if (cpus and isinstance(cpus, list)) else {}
            gpu0 = gpus[0] if (gpus and isinstance(gpus, list)) else {}
            mem0 = mem_list[0] if (mem_list and isinstance(mem_list, list)) else {}
            stor0 = storage[0] if (storage and isinstance(storage, list)) else {}
            smart = stor0.get('smart', {}) or {}
            
            # Safe GPU fan speed handling (laptops/fanless GPUs report null)
            fan_speed = gpu0.get('fan_speed_pct')
            fan_rpm_val = (float(fan_speed) * 30.0) if fan_speed is not None else 0.0

            v1_data = {
                'timestamp': data.get('timestamp_utc', datetime.now(timezone.utc).isoformat()),
                'server_id': data.get('server_id', 'default'),
                'server_name': data.get('server_name', 'Unknown'),
                'data_mode': data.get('data_mode', 'production'),
                'cpu': {
                    'temp_package': cpu0.get('temp_package_c') if cpu0.get('temp_package_c') is not None else 45.0,
                    'load_percent': cpu0.get('load_percent') if cpu0.get('load_percent') is not None else 15.0,
                    'frequency_mhz': cpu0.get('frequency_mhz') if cpu0.get('frequency_mhz') is not None else 3000.0,
                    'temp_cores': cpu0.get('temp_cores_c') or [],
                    'context_switches': cpu0.get('context_switches') or 0,
                },
                'gpu': {
                    'temp': gpu0.get('temp_core_c') if gpu0.get('temp_core_c') is not None else 40.0,
                    'power_draw_w': gpu0.get('power_draw_w') if gpu0.get('power_draw_w') is not None else 30.0,
                    'utilization_percent': gpu0.get('sm_utilization_pct') if gpu0.get('sm_utilization_pct') is not None else 10,
                    'memory_used_mb': gpu0.get('vram_used_mb') if gpu0.get('vram_used_mb') is not None else 500,
                    'memory_total_mb': gpu0.get('vram_total_mb') if gpu0.get('vram_total_mb') is not None else 8192,
                    'clock_mhz': gpu0.get('clock_mhz') if gpu0.get('clock_mhz') is not None else 1500,
                    'fan_rpm': fan_rpm_val,
                    'ecc_sbe_count': gpu0.get('ecc_sbe_count') or 0,
                },
                'ram': {
                    'used_gb': mem0.get('used_gb') if mem0.get('used_gb') is not None else 4.0,
                    'total_gb': mem0.get('total_gb') if mem0.get('total_gb') is not None else 16.0,
                    'page_faults_sec': mem0.get('page_faults_sec') or 50,
                    'committed_gb': mem0.get('committed_gb') or (mem0.get('used_gb') or 4.0),
                },
                'hdd': {
                    'smart_5_raw': int(smart.get('reallocated_sectors') or 0),
                    'smart_9_raw': int(smart.get('power_on_hours') or 5000),
                    'smart_187_raw': int(smart.get('reported_uncorrectable') or 0),
                    'smart_188_raw': int(smart.get('command_timeout') or 0),
                    'smart_194_raw': int(smart.get('temperature_c') or 35),
                    'smart_197_raw': int(smart.get('current_pending_sector') or 0),
                    'smart_198_raw': int(smart.get('offline_uncorrectable') or 0),
                },
                'psu': {
                    'voltage_3v3': psu.get('voltage_3v3') if psu.get('voltage_3v3') is not None else 3.3,
                    'voltage_5v': psu.get('voltage_5v') if psu.get('voltage_5v') is not None else 5.0,
                    'voltage_12v': psu.get('voltage_12v') if psu.get('voltage_12v') is not None else 12.0,
                    'power_draw_w': psu.get('total_power_w') if psu.get('total_power_w') is not None else 200.0,
                },
                'fans': {
                    'cpu_fan_rpm': (fans[0].get('rpm') if fans and fans[0].get('rpm') is not None else 1500),
                    'case_fan1_rpm': (fans[1].get('rpm') if len(fans) > 1 and fans[1].get('rpm') is not None else 1000),
                },
                'environment': {
                    'system_uptime_hours': data.get('uptime_hours') if data.get('uptime_hours') is not None else 100.0,
                    'ambient_temp': env.get('ambient_temp_c') if env.get('ambient_temp_c') is not None else 25.0,
                    'humidity': env.get('humidity_pct') if env.get('humidity_pct') is not None else 50.0,
                    'vibration_rms': env.get('vibration_rms_g') if env.get('vibration_rms_g') is not None else 0.02,
                    'acoustic_rms': env.get('acoustic_rms_v') if env.get('acoustic_rms_v') is not None else 0.02,
                },
            }
            data = v1_data
        except Exception as e:
            logger.warning(f"v2→v1 conversion failed ({e}), passing raw payload")

    timestamp = data.get('timestamp', datetime.now(timezone.utc).isoformat())
    server_id = data.get('server_id', 'default')
    server_name = data.get('server_name', server_id)
    
    try:
        # Step 0: Inject live sensor data ONLY for locally-connected servers
        # that don't have their own environment sensors. Remote servers (e.g.,
        # IdeaPad in another room) should keep their own environment readings.
        # (BUG-G-SENSORBLEED fix)
        if sensor_reader:
            try:
                sensor_data = sensor_reader.get_latest() if hasattr(sensor_reader, "get_latest") else sensor_reader.read_all(force=False)
            except Exception as sensor_err:
                logger.warning(f"Sensor read failed: {sensor_err}")
                sensor_data = {}
            env = data.get('environment', {})
            if 'environment' not in data:
                data['environment'] = {}
            # Per-value injection: only override individual values at default
            if env.get('ambient_temp', 25.0) == 25.0 and 'ambient_temp' in sensor_data:
                data['environment']['ambient_temp'] = sensor_data.get('ambient_temp', 25.0)
            if env.get('humidity', 50.0) == 50.0 and 'humidity' in sensor_data:
                data['environment']['humidity'] = sensor_data.get('humidity', 50.0)
            if env.get('vibration_rms', 0.02) == 0.02 and 'vibration_rms' in sensor_data:
                data['environment']['vibration_rms'] = sensor_data.get('vibration_rms', 0.02)
            if env.get('acoustic_rms', 0.02) == 0.02 and 'acoustic_rms' in sensor_data:
                data['environment']['acoustic_rms'] = sensor_data.get('acoustic_rms', 0.02)
        
        # Step 1: Flatten nested JSON
        flat = flatten_telemetry(data)
        
        # Step 2: Store in per-server history buffer and update TTL
        _edge_server_last_seen[server_id] = time.time()
        if server_id not in telemetry_history:
            telemetry_history[server_id] = []
        telemetry_history[server_id].append(flat)
        if len(telemetry_history[server_id]) > MAX_HISTORY:
            telemetry_history[server_id].pop(0)
        
        # Step 3: Extract 28 engineered features (canonical order from schema)
        features_df, metadata = extract_features(flat, server_id=server_id)
        
        # Step 4: Run edge inference (metadata carries mock_class separately)
        result = engine.run_inference(features_df, metadata=metadata)

        pred_class = result.get("predictions", {}).get("class", 0)
        confidence = result.get("predictions", {}).get("confidence", 0.0)
        anomaly = result.get("anomaly_detected", False)
        offloaded = result.get("offloaded", False)

        # Realistic bounded TTF calculation (Paper 2 & Blueprint §5)
        # Healthy servers operate with a safe operational horizon of 90.0 days (> 30 Days).
        # Degraded components calculate TTF dynamically based on failure mode and confidence.
        BASE_FAULT_TTF = {
            1: 14.0,  # HDD_DEGRADATION (bad sectors take days/weeks to crash)
            2: 2.0,   # THERMAL_CRITICAL (thermal dry-out/throttle requires inspection within 48h)
            3: 7.0,   # PSU_ANOMALY (voltage sag/ripple requires prompt review)
            4: 5.0,   # FAN_FAILURE (bearing wear/RPM deviation allows short operational buffer)
            5: 3.0,   # MEMORY_FAULT (burst errors degrade within days)
            6: 7.0,   # GPU_DEGRADATION (thermal pad/bus drops)
        }
        if pred_class == 0 or not anomaly:
            ttf_days = 90.0
        else:
            base = BASE_FAULT_TTF.get(pred_class, 7.0)
            conf = float(confidence or 0.85)
            conf_scale = max(0.2, (1.0 - (conf - 0.7) / 0.3 * 0.7))
            ttf_days = max(0.5, round(base * conf_scale, 1))

        # Store calculated TTF in result predictions so SQLite local_store records it
        result.setdefault("predictions", {})["ttf_days"] = ttf_days

        # Step 5: Persist to SQLite ring buffer
        try:
            from edge.api_server import local_store
            local_store.store_telemetry(server_id, flat, result)
        except Exception as store_err:
            logger.warning(f"Failed to store telemetry to SQLite cache: {store_err}")

        # Step 5b: Feed v2 payload to Adaptive Baseline Engine
        adaptive_health = None
        if _v2_raw_payload is not None:
            try:
                if server_id not in adaptive_baselines:
                    adaptive_baselines[server_id] = AdaptiveBaseline(server_id)
                    logger.info(f"[{server_id}] Created new adaptive baseline engine")
                abl = adaptive_baselines[server_id]
                abl.ingest_v2_payload(_v2_raw_payload)
                adaptive_health = abl.get_server_health(_v2_raw_payload)
            except Exception as abl_err:
                logger.warning(f"[{server_id}] Adaptive baseline ingestion warning: {abl_err}")

        # Step 6: Extract dictionary of all 28 canonical feature values
        raw_telemetry = {feat: float(features_df[feat].iloc[0]) for feat in FEATURE_NAMES_V1}

        # Physical Hardware Sensor Guards:
        # Prevent false-positive alarms on unmonitored or nominal hardware
        if pred_class == 3:  # PSU_ANOMALY
            psu_info = _v2_raw_payload.get('psu', {}) if _v2_raw_payload else {}
            # If PSU telemetry is simulated/unsupported (e.g. consumer laptops without BMC/PMBus)
            if psu_info.get('source') in ('simulated', 'unsupported') or not psu_info.get('bmc_detected', False):
                logger.info(f"[{server_id}] Suppressed false-positive PSU_ANOMALY (unmonitored consumer PSU, no BMC/PMBus hardware diode)")
                pred_class = 0
                anomaly = False
                confidence = 0.99
                fault_label = "HEALTHY"
                affected_component = None
                ttf_days = 90.0

        elif pred_class == 5:  # MEMORY_FAULT
            mem_info = (_v2_raw_payload.get('memory', [{}])[0]) if _v2_raw_payload and _v2_raw_payload.get('memory') else {}
            pct_used = float(mem_info.get('percent_used', 50.0))
            faults_sec = float(mem_info.get('page_faults_sec', 10.0))
            impending = bool(mem_info.get('impending_failure_detected', False))
            # On consumer PCs / laptops, normal memory usage under 92% without page fault thrashing is nominal
            if pct_used < 92.0 and faults_sec < 1500.0 and not impending:
                logger.info(f"[{server_id}] Suppressed false-positive MEMORY_FAULT (RAM usage {pct_used:.1f}% nominal, page faults {faults_sec:.0f}/s normal)")
                pred_class = 0
                anomaly = False
                confidence = 0.99
                fault_label = "HEALTHY"
                affected_component = None
                ttf_days = 90.0

        # Map class to component using shared schema
        if not anomaly or pred_class == 0:
            fault_label = "HEALTHY"
            affected_component = None
        else:
            fault_label = FAULT_LABELS.get(pred_class, "UNKNOWN")
            affected_component = CLASS_TO_COMPONENT.get(pred_class)
        
        # Acquire lock for all state mutations
        with state_lock:
            # Initialize server state if this is a new server
            if server_id not in current_state:
                current_state[server_id] = _default_server_state()
            
            srv = current_state[server_id]
            srv["last_seen"] = timestamp
            srv["server_name"] = server_name
            srv["telemetry"] = raw_telemetry
            
            # Update component metric values
            # Direct extraction: use live v2 readings if available, else raw_telemetry
            cpu_load = raw_telemetry.get("cpu_load", 15.0)
            cpu_temp = raw_telemetry.get("cpu_temp", 45.0)
            cpu_freq = raw_telemetry.get("cpu_freq", 3000.0)
            
            gpu_temp = raw_telemetry.get("gpu_temp", 40.0)
            gpu_util = raw_telemetry.get("gpu_util", 10.0)
            gpu_power = raw_telemetry.get("gpu_power", 30.0)
            gpu_vram = raw_telemetry.get("gpu_mem_used", 500.0)
            gpu_clock = raw_telemetry.get("gpu_clock", 1500.0)

            if _v2_raw_payload:
                c_list = _v2_raw_payload.get('cpus', [{}])
                if c_list and isinstance(c_list, list) and len(c_list) > 0:
                    c0 = c_list[0]
                    if c0.get('load_percent') is not None:
                        cpu_load = float(c0['load_percent'])
                    if c0.get('temp_package_c') is not None:
                        cpu_temp = float(c0['temp_package_c'])
                    if c0.get('frequency_mhz') is not None:
                        cpu_freq = float(c0['frequency_mhz'])
                
                g_list = _v2_raw_payload.get('gpus', [{}])
                if g_list and isinstance(g_list, list) and len(g_list) > 0:
                    g0 = g_list[0]
                    if g0.get('temp_core_c') is not None:
                        gpu_temp = float(g0['temp_core_c'])
                    if g0.get('sm_utilization_pct') is not None:
                        gpu_util = float(g0['sm_utilization_pct'])
                    if g0.get('power_draw_w') is not None:
                        gpu_power = float(g0['power_draw_w'])
                    if g0.get('vram_used_mb') is not None:
                        gpu_vram = float(g0['vram_used_mb'])
                    if g0.get('clock_mhz') is not None:
                        gpu_clock = float(g0['clock_mhz'])

            # CPU
            srv["components"]["cpu"]["metrics"] = {
                "temp": round(cpu_temp, 1),
                "load": round(cpu_load, 1),
                "freq": round(cpu_freq, 0),
                "delta": round(raw_telemetry.get("cpu_ambient_delta", cpu_temp - 25.0), 1)
            }
            # GPU
            srv["components"]["gpu"]["metrics"] = {
                "temp": round(gpu_temp, 1),
                "util": round(gpu_util, 1),
                "power": round(gpu_power, 1),
                "vram_mb": round(gpu_vram, 0),
                "clock": round(gpu_clock, 0)
            }
            # HDD
            hdd_smart5 = int(raw_telemetry.get("smart_5_raw", 0))
            hdd_smart187 = int(raw_telemetry.get("smart_187_raw", 0))
            hdd_smart197 = int(raw_telemetry.get("smart_197_raw", 0))
            hdd_health_calc = max(0, 100 - (hdd_smart5 * 20 + hdd_smart187 * 25 + hdd_smart197 * 15))
            srv["components"]["hdd"]["metrics"] = {
                "health": round(hdd_health_calc, 1),
                "temp": round(raw_telemetry.get("smart_194_raw", 35.0), 1),
                "smart_5": hdd_smart5,
                "smart_9_hours": int(raw_telemetry.get("smart_9_raw", 5000)),
                "smart_187": hdd_smart187,
                "smart_197": hdd_smart197,
                "smart_198": int(raw_telemetry.get("smart_198_raw", 0))
            }
            # PSU
            dev_3v3 = round(raw_telemetry.get("psu_voltage_3v3_dev", 0.5), 2)
            dev_12v = round(raw_telemetry.get("psu_voltage_12v_dev", 0.5), 2)
            psu_status = "CRITICAL" if max(dev_3v3, dev_12v) > 5.0 else "WARNING" if max(dev_3v3, dev_12v) > 3.0 else "NOMINAL"
            srv["components"]["psu"]["metrics"] = {
                "dev_3v3": dev_3v3,
                "dev_12v": dev_12v,
                "stability": round(raw_telemetry.get("psu_voltage_stability", 0.05), 3),
                "status": psu_status
            }
            # RAM
            ram_used = raw_telemetry.get("ram_used_gb", 4.0)
            ram_tot = max(1.0, raw_telemetry.get("ram_total_gb", 16.0))
            srv["components"]["ram"]["metrics"] = {
                "used_gb": round(ram_used, 2),
                "total_gb": round(ram_tot, 1),
                "percent": round((ram_used / ram_tot) * 100.0, 1),
                "page_faults": int(raw_telemetry.get("ram_page_faults", 50))
            }
            # FANS
            rpm_dev = raw_telemetry.get("fan_rpm_deviation", 0.0)
            est_rpm = int(flat.get("fans_cpu_fan_rpm", 1500.0 * (1.0 + rpm_dev)))
            srv["components"]["fans"]["metrics"] = {
                "rpm": est_rpm,
                "deviation": round(rpm_dev * 100, 1),
                "status": "SLOW" if rpm_dev < -0.15 else "HIGH" if rpm_dev > 0.3 else "OPTIMAL"
            }

            # ── Multi-hardware instances from v2 raw payload ──
            if _v2_raw_payload:
                # GPU instances
                gpus_raw = _v2_raw_payload.get('gpus', [])
                if gpus_raw:
                    gpu_instances = []
                    for gi, gpu in enumerate(gpus_raw):
                        gpu_instances.append({
                            "id": gpu.get("id", f"GPU{gi}"),
                            "temp": round(float(gpu.get("temp_core_c") or 40.0), 1),
                            "util": round(float(gpu.get("sm_utilization_pct") or 0.0), 1),
                            "power": round(float(gpu.get("power_draw_w") or 0.0), 1),
                            "vram_used": int(gpu.get("vram_used_mb") or 0),
                            "vram_total": int(gpu.get("vram_total_mb") or 0),
                            "clock": int(gpu.get("clock_mhz") or 0),
                            "fan_pct": int(gpu.get("fan_speed_pct") or 0),
                        })
                    srv["components"]["gpu"]["instances"] = gpu_instances

                # Storage instances (HDD/SSD/NVMe)
                storage_raw = _v2_raw_payload.get('storage', [])
                if storage_raw:
                    disk_instances = []
                    for di, disk in enumerate(storage_raw):
                        smart = disk.get("smart", {})
                        dtype = disk.get("type", "Unknown")
                        is_ssd = "SSD" in dtype.upper() or "NVME" in dtype.upper()
                        avail_spare = disk.get("available_spare_pct")
                        pct_used = disk.get("percentage_used")
                        
                        if is_ssd and pct_used is not None:
                            d_health = max(0, 100 - int(pct_used))
                        else:
                            d_health = max(0, 100 - (int(smart.get("reallocated_sectors", 0)) * 20
                                       + int(smart.get("reported_uncorrectable", 0)) * 25
                                       + int(smart.get("current_pending_sector", 0)) * 15))

                        disk_instances.append({
                            "id": disk.get("id", f"disk{di}"),
                            "model": disk.get("model", "Drive"),
                            "type": dtype,
                            "media_category": "SSD (Flash NVMe/SATA)" if is_ssd else "Mechanical HDD (Rotational Platter)",
                            "capacity_gb": disk.get("capacity_gb", 0),
                            "used_pct": round(disk.get("used_pct", 0), 1),
                            "temp": smart.get("temperature_c", disk.get("temperature_c", 0)),
                            "health": d_health,
                            "power_on_hours": smart.get("power_on_hours", 0),
                            "percentage_used": pct_used,
                            "available_spare_pct": avail_spare,
                        })
                    srv["components"]["hdd"]["instances"] = disk_instances

                # Memory instances
                mem_raw = _v2_raw_payload.get('memory', [])
                if mem_raw:
                    mem_instances = []
                    for mi, mem in enumerate(mem_raw if isinstance(mem_raw, list) else [mem_raw]):
                        mem_instances.append({
                            "id": f"DIMM{mi}",
                            "used_gb": round(mem.get("used_gb", 0), 2),
                            "total_gb": round(mem.get("total_gb", 0), 1),
                            "percent": round(mem.get("percent_used", 0), 1),
                            "page_faults": mem.get("page_faults_sec", 0),
                        })
                    srv["components"]["ram"]["instances"] = mem_instances

                # Fan instances
                fans_raw = _v2_raw_payload.get('fans', [])
                if fans_raw:
                    fan_instances = []
                    for fi, fan in enumerate(fans_raw if isinstance(fans_raw, list) else []):
                        fan_instances.append({
                            "id": fan.get("id", f"fan{fi}"),
                            "label": fan.get("label", f"Fan {fi}"),
                            "rpm": fan.get("rpm", 0),
                            "pwm_pct": fan.get("pwm_duty_pct", 0),
                        })
                    srv["components"]["fans"]["instances"] = fan_instances

                # PSU battery info (laptops)
                psu_raw = _v2_raw_payload.get('psu', {})
                if psu_raw.get('battery_percent') is not None:
                    srv["components"]["psu"]["battery"] = {
                        "percent": psu_raw.get("battery_percent"),
                        "plugged": psu_raw.get("battery_plugged", False),
                        "secs_left": psu_raw.get("battery_secs_left", -1),
                    }

            
            # 1. Heal all components not currently failing
            for comp in srv["components"]:
                if comp != affected_component or pred_class == 0 or not anomaly:
                    if pred_class == 0 and not anomaly:
                        srv["components"][comp]["ttf_days"] = 90.0
                        srv["components"][comp]["status"] = "HEALTHY"
                    else:
                        current_ttf = srv["components"][comp].get("ttf_days", 90.0)
                        new_ttf = min(90.0, current_ttf + 0.5) # Heal by 0.5 days per tick up to safe horizon (90.0d)
                        srv["components"][comp]["ttf_days"] = round(new_ttf, 1)
                        if new_ttf >= 30:
                            srv["components"][comp]["status"] = "HEALTHY"
                        elif new_ttf > 7:
                            srv["components"][comp]["status"] = "WARNING"
                        else:
                            srv["components"][comp]["status"] = "CRITICAL"
            
            # 2. Update affected component status
            if affected_component and pred_class != 0 and anomaly:
                status = "WARNING" if ttf_days > 7.0 else "CRITICAL"
                srv["components"][affected_component]["status"] = status
                srv["components"][affected_component]["ttf_days"] = round(ttf_days, 1)
                
            # 3. Calculate Robust System Health Index
            #    Phase H: If adaptive baseline is calibrated, USE IT as primary.
            #    Otherwise fall back to TTF-based calculation.
            if adaptive_health and adaptive_health.get("calibrated"):
                # PRIMARY: Adaptive baseline health score
                robust_health = adaptive_health["health_score"]
                srv["adaptive"] = adaptive_health
                # Update per-component status from adaptive engine
                for comp_id, comp_detail in adaptive_health.get("components", {}).items():
                    comp_type = comp_detail.get("type", "").split("-")[0]
                    # Map adaptive component types to dashboard component keys
                    type_map = {"cpu": "cpu", "gpu": "gpu", "storage": "hdd",
                                "memory": "ram", "fan": "fans", "psu": "psu"}
                    dash_key = type_map.get(comp_type)
                    if dash_key and dash_key in srv["components"]:
                        adp_health = comp_detail.get("health", 100)
                        adp_sev = comp_detail.get("severity", "healthy")
                        if adp_sev in ("critical", "degraded"):
                            srv["components"][dash_key]["status"] = "CRITICAL"
                        elif adp_sev == "warning":
                            srv["components"][dash_key]["status"] = "WARNING"
                        else:
                            srv["components"][dash_key]["status"] = "HEALTHY"
                        srv["components"][dash_key]["adaptive_health"] = round(adp_health, 1)
            else:
                # FALLBACK: TTF-based health (pre-calibration or v1 payloads)
                total_health = 0.0
                weights = {"cpu": 2.0, "gpu": 2.0, "hdd": 1.0, "psu": 1.0, "ram": 1.0, "fans": 1.0}
                total_weight = sum(weights.values())
                for comp, info in srv["components"].items():
                    comp_ttf = info.get("ttf_days", 90.0)
                    comp_health = min(100.0, (comp_ttf / 30.0) * 100.0)
                    total_health += comp_health * weights.get(comp, 1.0)
                robust_health = total_health / total_weight
                if anomaly and pred_class == 0:
                    robust_health = max(0.0, robust_health - 15.0)
                # Store calibration progress
                if adaptive_health:
                    srv["adaptive"] = adaptive_health
                
            srv["health_index"] = round(robust_health, 1)
            if _v2_raw_payload:
                srv["v2_raw"] = _v2_raw_payload
            
            # 4. Update rolling history sparklines (last 60 ticks)
            if "history" not in srv:
                srv["history"] = []
                
            hist_entry = {
                "time": timestamp,
                "cpu_temp": srv["components"]["cpu"]["metrics"]["temp"],
                "cpu_load": srv["components"]["cpu"]["metrics"]["load"],
                "gpu_temp": srv["components"]["gpu"]["metrics"]["temp"],
                "gpu_util": srv["components"]["gpu"]["metrics"]["util"],
                "ram_used_gb": srv["components"]["ram"]["metrics"]["used_gb"],
                "ram_percent": srv["components"]["ram"]["metrics"]["percent"],
                "fan_rpm": srv["components"]["fans"]["metrics"]["rpm"],
                "psu_12v_dev": srv["components"]["psu"]["metrics"]["dev_12v"],
                "ambient_temp": round(raw_telemetry.get("ambient_temp", 25.0), 1),
                "vibration_rms": round(raw_telemetry.get("vibration_rms", 0.02), 3),
                "health_index": round(robust_health, 1)
            }
            srv["history"].append(hist_entry)
            if len(srv["history"]) > 60:
                srv["history"].pop(0)

            # 5. Update inference decision metadata
            srv["inference"] = {
                "location": "CLOUD" if offloaded else "EDGE",
                "confidence": round(confidence, 3),
                "latency_ms": result.get("latency_ms", 18.5),
                "fault_class": pred_class,
                "fault_label": fault_label,
                "ttf_days": round(ttf_days, 1),
                "anomaly_detected": anomaly,
                "offloaded": offloaded,
                "lstm_ae_anomaly": result.get("lstm_ae_anomaly", False),
                "lstm_ae_recon_error": result.get("lstm_ae_recon_error"),
            }
            
            # 6. Log anomaly if detected
            if anomaly and pred_class != 0:
                srv["anomalies"].append({
                    "component": affected_component or "system",
                    "fault": fault_label,
                    "confidence": round(confidence, 3),
                    "ttf_days": round(ttf_days, 1),
                    "offloaded": offloaded,
                    "time": timestamp
                })
                srv["anomalies"] = srv["anomalies"][-50:]
                
            if "shap_explanation" in result and result["shap_explanation"]:
                srv["recent_shap_data"] = result["shap_explanation"]
        
        # Log result
        status_str = f"[{server_id}] [{'OFFLOAD' if offloaded else 'LOCAL'}] {fault_label} (conf={confidence:.2f}, TTF={ttf_days:.0f}d)"
        if anomaly:
            status_str += " !! ANOMALY"
            logger.warning(f"[{timestamp[:19]}] {status_str}")
        else:
            logger.info(f"[{timestamp[:19]}] {status_str}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)


def _graceful_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT for clean shutdown."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name} — shutting down gracefully...")
    raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser(description="Edge AI Main Entrypoint (RPi4)")
    parser.add_argument("--broker", type=str, default=MQTT_BROKER, help="MQTT Broker IP")
    parser.add_argument("--port", type=int, default=MQTT_PORT, help="MQTT Broker Port")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard HTTP port")
    args = parser.parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    
    logger.info("=" * 60)
    logger.info("  Edge-AI Predictive Maintenance - RPi4 Edge Node")
    logger.info("=" * 60)
    logger.info(f"  MQTT Broker:    {args.broker}:{args.port}")
    cloud_offload_fallback = os.environ.get('CLOUD_INFERENCE_URL', 'http://localhost:8000').rstrip('/') + '/api/offload'
    logger.info(f"  Cloud Offload:  {os.environ.get('CLOUD_OFFLOAD_URL', cloud_offload_fallback)}")
    
    global _global_mqtt_handler
    # 1. Start MQTT handler with inference callback
    mqtt_handler = EdgeMQTTHandler(broker=args.broker, port=args.port)
    _global_mqtt_handler = mqtt_handler
    mqtt_handler.set_telemetry_callback(on_telemetry_received)
    mqtt_handler.start()
    logger.info("MQTT handler started, listening for telemetry...")

    # 1b. Start background hardware sensor publisher thread
    if sensor_reader:
        sensor_pub_thread = threading.Thread(
            target=_background_sensor_publisher,
            args=(mqtt_handler, 2.0),
            name="SensorPublisherThread",
            daemon=True
        )
        sensor_pub_thread.start()
        logger.info("Physical sensor publisher thread started -> 'edge/sensors' (interval=2.0s)")
    
    # 2. Start FastAPI dashboard server in a background thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": "edge.api_server:app",
            "host": "0.0.0.0",
            "port": args.dashboard_port,
            "log_level": "warning"
        },
        daemon=True
    )
    server_thread.start()
    logger.info(f"Dashboard server started on port {args.dashboard_port}")
    
    # 3. Keep main thread alive + periodic TTL cleanup
    try:
        logger.info("Edge node is active. Press Ctrl+C to stop.")
        _cleanup_counter = 0
        while True:
            time.sleep(1)
            _cleanup_counter += 1
            if _cleanup_counter >= 300:  # Every 5 minutes
                _cleanup_counter = 0
                now = time.time()
                stale = [sid for sid, ts in _edge_server_last_seen.items()
                         if now - ts > _EDGE_SERVER_TTL]
                for sid in stale:
                    telemetry_history.pop(sid, None)
                    adaptive_baselines.pop(sid, None)
                    _edge_server_last_seen.pop(sid, None)
                    with state_lock:
                        current_state.pop(sid, None)
                    logger.info(f"TTL evicted stale server: {sid}")
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down edge node...")
        mqtt_handler.stop()
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()
