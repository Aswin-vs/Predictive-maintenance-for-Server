"""
Shared Feature Schema Contract — FEATURE_SCHEMA_V1.

THIS IS THE SINGLE SOURCE OF TRUTH for the 28 engineered features used across
the entire system. Both training pipelines and inference pipelines MUST import
from this module to guarantee feature ordering consistency.

HISTORY:
    - v1.0 (2026-08-09): Created by @claude-architect to fix BUG-N1 (feature
      ordering mismatch between training and edge inference).

USAGE:
    from server.training.feature_schema import FEATURE_NAMES_V1, FEATURE_COUNT
    # or, from the edge:
    import sys; sys.path.insert(0, BASE_DIR)
    from server.training.feature_schema import FEATURE_NAMES_V1
"""

# Version identifier — increment when changing the feature set
SCHEMA_VERSION = "1.0"

# Total number of features the models expect
FEATURE_COUNT = 28

# Canonical ordered list of feature names.
# ALL models (1D-CNN, LSTM-AE, Isolation Forest, XGBoost) and ALL inference
# pipelines (edge/main.py, server/deep_analysis/inference_server.py) MUST
# use this exact ordering.
#
# Features are grouped by component for readability, but the ORDER is what
# matters for the model input vector.
FEATURE_NAMES_V1 = [
    # --- CPU (indices 0-3) ---
    "cpu_temp",                 # 0: CPU package temperature (°C)
    "cpu_load",                 # 1: CPU load (%)
    "cpu_freq",                 # 2: CPU frequency (MHz)
    "cpu_ambient_delta",        # 3: CPU temp minus ambient temp (°C)

    # --- GPU (indices 4-8) ---
    "gpu_temp",                 # 4: GPU temperature (°C)
    "gpu_power",                # 5: GPU power draw (W)
    "gpu_util",                 # 6: GPU utilization (%)
    "gpu_mem_used",             # 7: GPU memory used (MB)
    "gpu_clock",                # 8: GPU clock speed (MHz)

    # --- RAM (indices 9-11) ---
    "ram_used_gb",              # 9: RAM used (GB)
    "ram_total_gb",             # 10: RAM total (GB)
    "ram_page_faults",          # 11: Page faults per second

    # --- HDD / SMART (indices 12-18) ---
    "smart_5_raw",              # 12: Reallocated Sectors Count
    "smart_9_raw",              # 13: Power-On Hours
    "smart_187_raw",            # 14: Reported Uncorrectable Errors
    "smart_188_raw",            # 15: Command Timeout Count
    "smart_194_raw",            # 16: HDD Temperature (°C)
    "smart_197_raw",            # 17: Current Pending Sector Count
    "smart_198_raw",            # 18: Offline Uncorrectable Sector Count

    # --- PSU (indices 19-21) ---
    "psu_voltage_3v3_dev",      # 19: 3.3V rail deviation (%)
    "psu_voltage_12v_dev",      # 20: 12V rail deviation (%)
    "psu_voltage_stability",    # 21: 12V rolling std deviation

    # --- Fans (index 22) ---
    "fan_rpm_deviation",        # 22: (actual - expected) / expected RPM

    # --- Environment / External Sensors (indices 23-26) ---
    "ambient_temp",             # 23: Ambient temperature (°C) — from DHT22
    "humidity",                 # 24: Relative humidity (%) — from DHT22
    "vibration_rms",            # 25: Vibration RMS (g) — from ADXL345
    "acoustic_rms",             # 26: Acoustic RMS — from MAX4466

    # --- System (index 27) ---
    "system_uptime",            # 27: System uptime (hours)
]

assert len(FEATURE_NAMES_V1) == FEATURE_COUNT, (
    f"FEATURE_NAMES_V1 has {len(FEATURE_NAMES_V1)} entries, expected {FEATURE_COUNT}"
)

# Isolation Forest subset — uses PSU + Fan features only
IF_FEATURE_NAMES = [
    "psu_voltage_3v3_dev",      # index 19
    "psu_voltage_12v_dev",      # index 20
    "psu_voltage_stability",    # index 21
    "fan_rpm_deviation",        # index 22
]

# XGBoost HDD subset — uses SMART features only
XGB_FEATURE_NAMES = [
    "smart_5_raw",              # index 12
    "smart_9_raw",              # index 13
    "smart_187_raw",            # index 14
    "smart_188_raw",            # index 15
    "smart_194_raw",            # index 16
    "smart_197_raw",            # index 17
    "smart_198_raw",            # index 18
]

# Feature indices for quick lookup
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES_V1)}

# Default values for features when telemetry data is missing.
# These represent "normal/healthy" baseline values.
FEATURE_DEFAULTS = {
    "cpu_temp": 45.0,
    "cpu_load": 15.0,
    "cpu_freq": 3000.0,
    "cpu_ambient_delta": 20.0,
    "gpu_temp": 40.0,
    "gpu_power": 30.0,
    "gpu_util": 10.0,
    "gpu_mem_used": 500.0,
    "gpu_clock": 1500.0,
    "ram_used_gb": 4.0,
    "ram_total_gb": 16.0,
    "ram_page_faults": 50.0,
    "smart_5_raw": 0.0,
    "smart_9_raw": 5000.0,
    "smart_187_raw": 0.0,
    "smart_188_raw": 0.0,
    "smart_194_raw": 35.0,
    "smart_197_raw": 0.0,
    "smart_198_raw": 0.0,
    "psu_voltage_3v3_dev": 0.5,
    "psu_voltage_12v_dev": 0.5,
    "psu_voltage_stability": 0.05,
    "fan_rpm_deviation": 0.0,
    "ambient_temp": 25.0,
    "humidity": 50.0,
    "vibration_rms": 0.02,
    "acoustic_rms": 0.02,
    "system_uptime": 100.0,
}

# Fault class labels — shared across training and inference
FAULT_LABELS = {
    0: "HEALTHY",
    1: "HDD_DEGRADATION",
    2: "THERMAL_CRITICAL",
    3: "PSU_ANOMALY",
    4: "FAN_FAILURE",
    5: "MEMORY_FAULT",
    6: "GPU_DEGRADATION",
}
NUM_CLASSES = len(FAULT_LABELS)

# Maps fault class index to the affected hardware component
CLASS_TO_COMPONENT = {
    0: None,        # HEALTHY — no component affected
    1: "hdd",
    2: "cpu",       # THERMAL_CRITICAL
    3: "psu",
    4: "fans",
    5: "ram",
    6: "gpu",
}


def get_ordered_defaults():
    """Return a list of default feature values in FEATURE_NAMES_V1 order."""
    return [FEATURE_DEFAULTS[name] for name in FEATURE_NAMES_V1]


if __name__ == "__main__":
    print(f"Feature Schema v{SCHEMA_VERSION}")
    print(f"Total features: {FEATURE_COUNT}")
    print(f"\nFeature order:")
    for i, name in enumerate(FEATURE_NAMES_V1):
        default = FEATURE_DEFAULTS[name]
        print(f"  [{i:2d}] {name:30s} default={default}")
    print(f"\nIF subset: {IF_FEATURE_NAMES}")
    print(f"XGB subset: {XGB_FEATURE_NAMES}")
    print(f"\nFault classes: {FAULT_LABELS}")
