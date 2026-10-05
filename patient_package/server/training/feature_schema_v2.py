"""
Feature Schema v2.0 — Phase G1.4

Defines the hierarchical telemetry schema for multi-hardware support.
Maintains backward compatibility with FEATURE_NAMES_V1 (28 flat features)
for existing INT8 models on the Edge.

This file is the SINGLE SOURCE OF TRUTH for:
  - v2 hierarchical schema definition
  - v1 ↔ v2 conversion utilities
  - Feature defaults for missing hardware components
  - Fault labels and classifications

Author: @claude-architect (Phase G)
Date: 2026-09-04
"""

from typing import Dict, Any, List, Optional

# ─── Schema Version ───────────────────────────────────────────────────────────
SCHEMA_VERSION = "2.0"

# ─── v1 Backward Compatibility (28 flat features) ────────────────────────────
# These are imported by edge_inference.py, inference_server.py, and all
# existing training scripts. DO NOT REMOVE OR RENAME.
FEATURE_NAMES_V1 = [
    "cpu_temp", "cpu_load", "cpu_freq", "cpu_ambient_delta",
    "gpu_temp", "gpu_power", "gpu_util", "gpu_mem_used", "gpu_clock",
    "ram_used_gb", "ram_total_gb", "ram_page_faults",
    "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw",
    "smart_194_raw", "smart_197_raw", "smart_198_raw",
    "psu_voltage_3v3_dev", "psu_voltage_12v_dev", "psu_voltage_stability",
    "fan_rpm_deviation",
    "ambient_temp", "humidity", "vibration_rms", "acoustic_rms",
    "system_uptime",
]

FEATURE_COUNT = len(FEATURE_NAMES_V1)  # 28
assert FEATURE_COUNT == 28, f"Expected 28 features, got {FEATURE_COUNT}"

# Default values for each v1 feature (used when hardware is missing)
FEATURE_DEFAULTS_V1 = {
    "cpu_temp": 45.0, "cpu_load": 15.0, "cpu_freq": 3000.0,
    "cpu_ambient_delta": 20.0,
    "gpu_temp": 40.0, "gpu_power": 30.0, "gpu_util": 10.0,
    "gpu_mem_used": 500.0, "gpu_clock": 1500.0,
    "ram_used_gb": 4.0, "ram_total_gb": 16.0, "ram_page_faults": 50.0,
    "smart_5_raw": 0.0, "smart_9_raw": 5000.0, "smart_187_raw": 0.0,
    "smart_188_raw": 0.0, "smart_194_raw": 35.0, "smart_197_raw": 0.0,
    "smart_198_raw": 0.0,
    "psu_voltage_3v3_dev": 0.5, "psu_voltage_12v_dev": 0.5,
    "psu_voltage_stability": 0.05,
    "fan_rpm_deviation": 0.0,
    "ambient_temp": 25.0, "humidity": 50.0,
    "vibration_rms": 0.02, "acoustic_rms": 0.02,
    "system_uptime": 100.0,
}

# ─── Fault Labels ─────────────────────────────────────────────────────────────
FAULT_LABELS = {
    0: "Healthy",
    1: "CPU_Thermal_Failure",
    2: "GPU_Power_Failure",
    3: "HDD_SMART_Failure",
    4: "PSU_Voltage_Instability",
    5: "Fan_Bearing_Degradation",
    6: "Memory_Degradation",
}

FAULT_LABEL_COUNT = len(FAULT_LABELS)

# ─── v2 Subsystem Definitions ────────────────────────────────────────────────
# Each subsystem defines what fields are expected in the hierarchical payload

V2_CPU_FIELDS = [
    "load_percent", "frequency_mhz", "temp_package_c", "temp_cores_c",
    "context_switches", "io_wait_percent", "throttle_detected",
    "per_core_load", "min_freq_mhz", "max_freq_mhz", "interrupts",
]

V2_GPU_FIELDS = [
    "temp_core_c", "temp_junction_c", "power_draw_w",
    "vram_used_mb", "vram_total_mb", "clock_mhz",
    "sm_utilization_pct", "mem_utilization_pct", "fan_speed_pct",
    "ecc_sbe_count", "ecc_dbe_count", "pcie_link_width", "pcie_gen",
]

V2_STORAGE_FIELDS_HDD = [
    "reallocated_sectors", "power_on_hours", "reported_uncorrectable",
    "command_timeout", "temperature_c", "current_pending_sector",
    "offline_uncorrectable",
]

V2_STORAGE_FIELDS_NVME = [
    "wear_leveling_pct", "available_spare_pct", "media_errors",
    "power_loss_count", "temperature_c",
]

V2_MEMORY_FIELDS = [
    "used_gb", "total_gb", "available_gb", "percent_used",
    "committed_gb", "swap_used_gb", "swap_percent", "page_faults_sec",
]

V2_PSU_FIELDS = [
    "voltage_3v3", "voltage_5v", "voltage_12v", "total_power_w",
]

V2_FAN_FIELDS = [
    "rpm", "pwm_duty_pct",
]

V2_ENVIRONMENT_FIELDS = [
    "ambient_temp_c", "humidity_pct", "vibration_rms_g", "acoustic_rms_v",
]


# ─── Conversion Utilities ────────────────────────────────────────────────────

def v2_to_v1_feature_vector(v2_payload: Dict[str, Any]) -> List[float]:
    """
    Convert a v2 hierarchical payload to a flat 28-element feature vector
    matching FEATURE_NAMES_V1 order. Used by existing INT8 edge models.

    Missing components use FEATURE_DEFAULTS_V1 values.
    """
    flat = {}

    # CPU (first CPU)
    cpu = (v2_payload.get("cpus") or [{}])[0]
    flat["cpu_temp"] = cpu.get("temp_package_c", FEATURE_DEFAULTS_V1["cpu_temp"])
    flat["cpu_load"] = cpu.get("load_percent", FEATURE_DEFAULTS_V1["cpu_load"])
    flat["cpu_freq"] = cpu.get("frequency_mhz", FEATURE_DEFAULTS_V1["cpu_freq"])

    # Ambient
    env = v2_payload.get("environment", {})
    ambient = env.get("ambient_temp_c", FEATURE_DEFAULTS_V1["ambient_temp"])
    flat["cpu_ambient_delta"] = flat["cpu_temp"] - ambient

    # GPU (first GPU)
    gpu = (v2_payload.get("gpus") or [{}])[0]
    flat["gpu_temp"] = gpu.get("temp_core_c", FEATURE_DEFAULTS_V1["gpu_temp"])
    flat["gpu_power"] = gpu.get("power_draw_w", FEATURE_DEFAULTS_V1["gpu_power"])
    flat["gpu_util"] = float(gpu.get("sm_utilization_pct", FEATURE_DEFAULTS_V1["gpu_util"]))
    flat["gpu_mem_used"] = float(gpu.get("vram_used_mb", FEATURE_DEFAULTS_V1["gpu_mem_used"]))
    flat["gpu_clock"] = float(gpu.get("clock_mhz", FEATURE_DEFAULTS_V1["gpu_clock"]))

    # RAM (first entry)
    mem = (v2_payload.get("memory") or [{}])[0]
    flat["ram_used_gb"] = mem.get("used_gb", FEATURE_DEFAULTS_V1["ram_used_gb"])
    flat["ram_total_gb"] = mem.get("total_gb", FEATURE_DEFAULTS_V1["ram_total_gb"])
    flat["ram_page_faults"] = float(mem.get("page_faults_sec", FEATURE_DEFAULTS_V1["ram_page_faults"]))

    # Storage SMART (first drive)
    storage = (v2_payload.get("storage") or [{}])[0]
    smart = storage.get("smart", {})
    flat["smart_5_raw"] = float(smart.get("reallocated_sectors", FEATURE_DEFAULTS_V1["smart_5_raw"]))
    flat["smart_9_raw"] = float(smart.get("power_on_hours", FEATURE_DEFAULTS_V1["smart_9_raw"]))
    flat["smart_187_raw"] = float(smart.get("reported_uncorrectable", FEATURE_DEFAULTS_V1["smart_187_raw"]))
    flat["smart_188_raw"] = float(smart.get("command_timeout", FEATURE_DEFAULTS_V1["smart_188_raw"]))
    flat["smart_194_raw"] = float(smart.get("temperature_c", FEATURE_DEFAULTS_V1["smart_194_raw"]))
    flat["smart_197_raw"] = float(smart.get("current_pending_sector", FEATURE_DEFAULTS_V1["smart_197_raw"]))
    flat["smart_198_raw"] = float(smart.get("offline_uncorrectable", FEATURE_DEFAULTS_V1["smart_198_raw"]))

    # PSU
    psu = v2_payload.get("psu", {})
    v_3v3 = psu.get("voltage_3v3", 3.3)
    v_12v = psu.get("voltage_12v", 12.0)
    flat["psu_voltage_3v3_dev"] = abs(v_3v3 - 3.3) / 3.3 * 100
    flat["psu_voltage_12v_dev"] = abs(v_12v - 12.0) / 12.0 * 100
    flat["psu_voltage_stability"] = FEATURE_DEFAULTS_V1["psu_voltage_stability"]

    # Fans
    flat["fan_rpm_deviation"] = FEATURE_DEFAULTS_V1["fan_rpm_deviation"]

    # Environment
    flat["ambient_temp"] = ambient
    flat["humidity"] = env.get("humidity_pct", FEATURE_DEFAULTS_V1["humidity"])
    flat["vibration_rms"] = env.get("vibration_rms_g", FEATURE_DEFAULTS_V1["vibration_rms"])
    flat["acoustic_rms"] = env.get("acoustic_rms_v", FEATURE_DEFAULTS_V1["acoustic_rms"])

    # System
    flat["system_uptime"] = v2_payload.get("uptime_hours", FEATURE_DEFAULTS_V1["system_uptime"])

    # Build ordered feature vector
    return [flat.get(name, FEATURE_DEFAULTS_V1[name]) for name in FEATURE_NAMES_V1]


def is_v2_payload(payload: Dict[str, Any]) -> bool:
    """Check if a telemetry payload is v2 (hierarchical) or v1 (flat)."""
    return payload.get("schema_version") == SCHEMA_VERSION


def v1_flat_to_v2(v1_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a v1 flat payload (from legacy publisher) to v2 hierarchical format.
    Used for backward compatibility when old-format publishers connect.
    """
    v2 = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": v1_payload.get("timestamp", ""),
        "uptime_hours": v1_payload.get("environment", {}).get("system_uptime_hours", 0),
    }

    # CPU
    cpu = v1_payload.get("cpu", {})
    v2["cpus"] = [{
        "id": "CPU0",
        "temp_package_c": cpu.get("temp_package", 45.0),
        "temp_cores_c": cpu.get("temp_cores", []),
        "load_percent": cpu.get("load_percent", 15.0),
        "frequency_mhz": cpu.get("frequency_mhz", 3000.0),
        "context_switches": cpu.get("context_switches", 0),
    }]

    # GPU
    gpu = v1_payload.get("gpu", {})
    v2["gpus"] = [{
        "id": "GPU0",
        "temp_core_c": gpu.get("temp", 40.0),
        "power_draw_w": gpu.get("power_draw_w", 30.0),
        "vram_used_mb": gpu.get("memory_used_mb", 500),
        "vram_total_mb": gpu.get("memory_total_mb", 8192),
        "clock_mhz": gpu.get("clock_mhz", 1500),
        "sm_utilization_pct": gpu.get("utilization_percent", 10),
        "fan_speed_pct": 0,
        "ecc_sbe_count": gpu.get("ecc_sbe_count", 0),
        "ecc_dbe_count": 0,
    }]

    # RAM
    ram = v1_payload.get("ram", {})
    v2["memory"] = [{
        "used_gb": ram.get("used_gb", 4.0),
        "total_gb": ram.get("total_gb", 16.0),
        "page_faults_sec": ram.get("page_faults_sec", 50),
        "committed_gb": ram.get("committed_gb", 0.0),
    }]

    # Storage
    hdd = v1_payload.get("hdd", {})
    v2["storage"] = [{
        "id": "sda",
        "type": "HDD",
        "smart": {
            "reallocated_sectors": hdd.get("smart_5_raw", 0),
            "power_on_hours": hdd.get("smart_9_raw", 5000),
            "reported_uncorrectable": hdd.get("smart_187_raw", 0),
            "command_timeout": hdd.get("smart_188_raw", 0),
            "temperature_c": hdd.get("smart_194_raw", 35),
            "current_pending_sector": hdd.get("smart_197_raw", 0),
            "offline_uncorrectable": hdd.get("smart_198_raw", 0),
        },
    }]

    # PSU
    psu = v1_payload.get("psu", {})
    v2["psu"] = {
        "voltage_3v3": psu.get("voltage_3v3", 3.3),
        "voltage_5v": psu.get("voltage_5v", 5.0),
        "voltage_12v": psu.get("voltage_12v", 12.0),
        "total_power_w": psu.get("power_draw_w", 200.0),
    }

    # Fans
    fans = v1_payload.get("fans", {})
    v2["fans"] = [
        {"id": k, "rpm": v} for k, v in fans.items()
    ] if isinstance(fans, dict) else []

    # Environment
    env = v1_payload.get("environment", {})
    v2["environment"] = {
        "ambient_temp_c": env.get("ambient_temp", 25.0),
        "humidity_pct": env.get("humidity", 50.0),
        "vibration_rms_g": env.get("vibration_rms", 0.02),
        "acoustic_rms_v": env.get("acoustic_rms", 0.02),
    }

    return v2


# ─── Per-Subsystem Feature Extractors for Specialist Models ───────────────────

def extract_storage_features(v2_payload: Dict, drive_index: int = 0) -> Optional[List[float]]:
    """Extract the 7 SMART features for the TCN storage RUL model."""
    drives = v2_payload.get("storage", [])
    if drive_index >= len(drives):
        return None
    smart = drives[drive_index].get("smart", {})
    return [
        float(smart.get("reallocated_sectors", 0)),
        float(smart.get("power_on_hours", 0)),
        float(smart.get("reported_uncorrectable", 0)),
        float(smart.get("command_timeout", 0)),
        float(smart.get("temperature_c", 35)),
        float(smart.get("current_pending_sector", 0)),
        float(smart.get("offline_uncorrectable", 0)),
    ]


def extract_fan_features(v2_payload: Dict, fan_index: int = 0) -> Optional[List[float]]:
    """Extract fan features for the 1D-CNN fan bearing detector."""
    fans = v2_payload.get("fans", [])
    if fan_index >= len(fans):
        return None
    fan = fans[fan_index]
    return [
        float(fan.get("pwm_duty_pct", 50.0)),
        float(fan.get("rpm", 1000)),
        # PWM-RPM ratio: a degraded bearing shows lower RPM per unit PWM
        float(fan.get("rpm", 1000)) / max(float(fan.get("pwm_duty_pct", 50.0)), 1.0),
    ]


def extract_psu_features(v2_payload: Dict) -> List[float]:
    """Extract PSU features for the Isolation Forest anomaly detector."""
    psu = v2_payload.get("psu", {})
    fans = v2_payload.get("fans", [])

    v_3v3 = psu.get("voltage_3v3", 3.3)
    v_12v = psu.get("voltage_12v", 12.0)
    fan_rpm = fans[0].get("rpm", 1000) if fans else 1000

    return [
        abs(v_3v3 - 3.3) / 3.3 * 100,  # 3.3V deviation %
        abs(v_12v - 12.0) / 12.0 * 100,  # 12V deviation %
        0.05,  # Voltage stability (needs rolling history)
        0.0,   # Fan RPM deviation (needs baseline)
    ]
