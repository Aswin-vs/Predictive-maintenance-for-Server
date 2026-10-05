"""
Unified Enterprise Schema Harmonization Engine.
Phase 5: Enterprise ML Training & Survival Analysis.

Harmonizes disparate hardware telemetry and research datasets into a unified
36-feature enterprise tensor representation:
1. Backblaze Hard Drive Stats (Quarterly SMART telemetry)
2. Alibaba Cloud PAKDD 2021 DRAM Failure Prediction Dataset
3. Alibaba Cluster-Trace-GPU-2020 (NSDI '22)
4. AI4I 2020 Predictive Maintenance Dataset
5. Live Hierarchical v2 Telemetry Agent Payloads

POLICY FOR MISSING ATTRIBUTES:
- Never silently zero-fill unmeasured channels.
- Missing values are imputed using physics-grounded operational baselines
  (e.g., ambient 25°C, 0.02g resting vibration, nominal +12V rail).
- Simultaneously generates a binary Missingness Mask Tensor (N, 36) where
  1.0 = genuine physical sensor reading, 0.0 = imputed channel.
  This allows Spatio-Temporal Graph Convolutions to distinguish zero degradation
  from unmonitored hardware.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


# ─── 36-Feature Canonical Schema ───────────────────────────────────────────────
ENTERPRISE_36_FEATURES = [
    # Subsystem 1: CPU Core Dynamics & Degradation (6)
    "cpu_temp",               # °C: Core/package temperature
    "cpu_load",               # %: Total CPU utilization
    "cpu_freq",               # MHz: Operating frequency
    "cpu_ambient_delta",      # °C: Temperature delta (cpu_temp - ambient)
    "cpu_thermal_slope",      # °C/(W*s): 60s sliding window thermal resistance slope
    "prochot_events_count",   # count: Hardware PROCHOT thermal throttle events

    # Subsystem 2: GPU Core, VRAM & Bus Stability (7)
    "gpu_temp",               # °C: GPU die core temperature
    "gpu_power",              # W: Board power draw
    "gpu_util",               # %: SM compute utilization
    "gpu_mem_used",           # MB: VRAM consumption
    "gpu_clock",              # MHz: SM engine graphics clock
    "gpu_hotspot_delta",      # °C: Hotspot gradient (T_junction - T_edge)
    "gpu_nvml_ecc_errors",    # count: Aggregate single-bit and double-bit errors

    # Subsystem 3: System Memory & ECC Telemetry (4)
    "ram_used_gb",            # GB: Physical memory consumed
    "ram_total_gb",           # GB: Installed physical capacity
    "ram_page_faults",        # /s: Memory page faults per second
    "mem_ecc_ce_burst_rate",  # count/min: Single-bit correctable error rate

    # Subsystem 4: Storage Endurance, SMART & NVMe Log 0x02 (9)
    "storage_nvme_wear_pct",  # %: Percentage used (wear endurance)
    "storage_nvme_spare_pct", # %: Available spare capacity remaining
    "smart_5_raw",            # count: Reallocated sector count
    "smart_9_raw",            # hours: Power-on hours count
    "smart_187_raw",          # count: Reported uncorrectable errors
    "smart_188_raw",          # count: Command timeouts
    "smart_194_raw",          # °C: Drive enclosure/controller temperature
    "smart_197_raw",          # count: Current pending sectors count
    "smart_198_raw",          # count: Offline uncorrectable sectors count

    # Subsystem 5: Power Supply Rails & Voltage Sag (4)
    "psu_voltage_3v3_dev",    # %: Absolute deviation from +3.3V nominal
    "psu_voltage_12v_dev",    # %: Absolute deviation from +12.0V nominal
    "psu_voltage_stability",  # std: Rolling voltage stability standard deviation
    "psu_rail_12v_sag",       # V: Voltage sag depth below 11.4V (-5% ATX limit)

    # Subsystem 6: Cooling Fans, Acoustic & Vibration Physics (5)
    "fan_rpm_deviation",      # %: Deviation from PWM target speed
    "ambient_temp",           # °C: Chassis intake ambient temperature
    "humidity",               # %: Relative humidity percentage
    "vibration_rms",          # g: 3-axis dynamic vibration root-mean-square
    "acoustic_rms",           # V: Normalized acoustic pressure root-mean-square

    # Subsystem 7: Temporal Operating Horizon (1)
    "system_uptime",          # hours: Continuous running operational uptime
]

ENTERPRISE_FEATURE_COUNT = len(ENTERPRISE_36_FEATURES)
assert ENTERPRISE_FEATURE_COUNT == 36, f"Expected 36 features, got {ENTERPRISE_FEATURE_COUNT}"


# ─── Explicit Physics-Grounded Imputation Baselines ────────────────────────────
ENTERPRISE_DEFAULTS: Dict[str, float] = {
    # CPU
    "cpu_temp": 45.0,
    "cpu_load": 15.0,
    "cpu_freq": 3000.0,
    "cpu_ambient_delta": 20.0,
    "cpu_thermal_slope": 0.0,
    "prochot_events_count": 0.0,

    # GPU
    "gpu_temp": 40.0,
    "gpu_power": 30.0,
    "gpu_util": 10.0,
    "gpu_mem_used": 500.0,
    "gpu_clock": 1500.0,
    "gpu_hotspot_delta": 8.0,
    "gpu_nvml_ecc_errors": 0.0,

    # Memory
    "ram_used_gb": 4.0,
    "ram_total_gb": 16.0,
    "ram_page_faults": 50.0,
    "mem_ecc_ce_burst_rate": 0.0,

    # Storage
    "storage_nvme_wear_pct": 10.0,
    "storage_nvme_spare_pct": 100.0,
    "smart_5_raw": 0.0,
    "smart_9_raw": 5000.0,
    "smart_187_raw": 0.0,
    "smart_188_raw": 0.0,
    "smart_194_raw": 35.0,
    "smart_197_raw": 0.0,
    "smart_198_raw": 0.0,

    # PSU
    "psu_voltage_3v3_dev": 0.5,
    "psu_voltage_12v_dev": 0.5,
    "psu_voltage_stability": 0.05,
    "psu_rail_12v_sag": 0.0,

    # Fans & Environment
    "fan_rpm_deviation": 0.0,
    "ambient_temp": 25.0,
    "humidity": 50.0,
    "vibration_rms": 0.02,
    "acoustic_rms": 0.02,

    # Temporal
    "system_uptime": 100.0,
}


# ─── Missingness Policy Specification ──────────────────────────────────────────
FIELD_POLICIES: Dict[str, Dict[str, Any]] = {
    "cpu_temp": {"policy": "impute_physics_baseline", "baseline": 45.0, "unit": "deg_C"},
    "cpu_load": {"policy": "impute_nominal_idle", "baseline": 15.0, "unit": "pct"},
    "cpu_freq": {"policy": "impute_nominal_clock", "baseline": 3000.0, "unit": "MHz"},
    "cpu_ambient_delta": {"policy": "derive_or_impute", "baseline": 20.0, "unit": "deg_C"},
    "cpu_thermal_slope": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "deg_C_per_W_s"},
    "prochot_events_count": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "count"},

    "gpu_temp": {"policy": "impute_physics_baseline", "baseline": 40.0, "unit": "deg_C"},
    "gpu_power": {"policy": "impute_idle_tdp", "baseline": 30.0, "unit": "Watts"},
    "gpu_util": {"policy": "impute_nominal_idle", "baseline": 10.0, "unit": "pct"},
    "gpu_mem_used": {"policy": "impute_nominal_idle", "baseline": 500.0, "unit": "MB"},
    "gpu_clock": {"policy": "impute_nominal_clock", "baseline": 1500.0, "unit": "MHz"},
    "gpu_hotspot_delta": {"policy": "impute_healthy_delta", "baseline": 8.0, "unit": "deg_C"},
    "gpu_nvml_ecc_errors": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "count"},

    "ram_used_gb": {"policy": "impute_nominal_idle", "baseline": 4.0, "unit": "GB"},
    "ram_total_gb": {"policy": "impute_nominal_capacity", "baseline": 16.0, "unit": "GB"},
    "ram_page_faults": {"policy": "impute_nominal_idle", "baseline": 50.0, "unit": "faults_per_s"},
    "mem_ecc_ce_burst_rate": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "errors_per_min"},

    "storage_nvme_wear_pct": {"policy": "impute_healthy_endurance", "baseline": 10.0, "unit": "pct"},
    "storage_nvme_spare_pct": {"policy": "impute_full_spare", "baseline": 100.0, "unit": "pct"},
    "smart_5_raw": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "count"},
    "smart_9_raw": {"policy": "impute_nominal_age", "baseline": 5000.0, "unit": "hours"},
    "smart_187_raw": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "count"},
    "smart_188_raw": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "count"},
    "smart_194_raw": {"policy": "impute_physics_baseline", "baseline": 35.0, "unit": "deg_C"},
    "smart_197_raw": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "count"},
    "smart_198_raw": {"policy": "impute_zero_wear", "baseline": 0.0, "unit": "count"},

    "psu_voltage_3v3_dev": {"policy": "impute_healthy_regulation", "baseline": 0.5, "unit": "pct"},
    "psu_voltage_12v_dev": {"policy": "impute_healthy_regulation", "baseline": 0.5, "unit": "pct"},
    "psu_voltage_stability": {"policy": "impute_healthy_regulation", "baseline": 0.05, "unit": "std"},
    "psu_rail_12v_sag": {"policy": "impute_zero_sag", "baseline": 0.0, "unit": "Volts"},

    "fan_rpm_deviation": {"policy": "impute_healthy_regulation", "baseline": 0.0, "unit": "pct"},
    "ambient_temp": {"policy": "impute_physics_baseline", "baseline": 25.0, "unit": "deg_C"},
    "humidity": {"policy": "impute_physics_baseline", "baseline": 50.0, "unit": "pct"},
    "vibration_rms": {"policy": "impute_resting_gravity", "baseline": 0.02, "unit": "g_norm"},
    "acoustic_rms": {"policy": "impute_resting_acoustic", "baseline": 0.02, "unit": "pressure_norm"},

    "system_uptime": {"policy": "impute_nominal_age", "baseline": 100.0, "unit": "hours"},
}


class EnterpriseSchemaHarmonizer:
    """
    Transforms any incoming dictionary, pandas DataFrame, or heterogeneous dataset
    into the canonical 36-feature enterprise tensor format.
    """
    def __init__(self, feature_order: Optional[List[str]] = None):
        self.feature_names = feature_order or ENTERPRISE_36_FEATURES
        self.feature_index = {name: i for i, name in enumerate(self.feature_names)}

    def harmonize_record(
        self,
        raw_data: Dict[str, Any],
        source_domain: str = "auto"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonizes a single telemetry or dataset record into:
        1. features: np.ndarray shape (36,)
        2. mask: np.ndarray shape (36,) where 1.0 = genuine, 0.0 = imputed
        """
        flat = self._flatten_dict(raw_data)
        features = np.zeros(len(self.feature_names), dtype=np.float32)
        mask = np.zeros(len(self.feature_names), dtype=np.float32)

        # Cross-sensor derivation
        cpu_t = flat.get("cpu_temp", flat.get("cpu_temp_package_c", flat.get("temp_package_c")))
        amb_t = flat.get("ambient_temp", flat.get("ambient_temp_c", flat.get("environment_ambient_temp_c", 25.0)))
        if "cpu_ambient_delta" not in flat and cpu_t is not None:
            flat["cpu_ambient_delta"] = float(cpu_t) - float(amb_t)

        for i, feat in enumerate(self.feature_names):
            val = self._extract_value(feat, flat)
            if val is not None and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                features[i] = float(val)
                mask[i] = 1.0
            else:
                features[i] = ENTERPRISE_DEFAULTS.get(feat, 0.0)
                mask[i] = 0.0

        return features, mask

    def harmonize_batch(
        self,
        records: List[Dict[str, Any]],
        source_domain: str = "auto"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonize a list of raw record dictionaries directly into:
        1. X: (N, 36) float32 feature array
        2. M: (N, 36) float32 binary missingness mask
        """
        n_rows = len(records)
        X = np.zeros((n_rows, len(self.feature_names)), dtype=np.float32)
        M = np.zeros((n_rows, len(self.feature_names)), dtype=np.float32)
        for i, rec in enumerate(records):
            X[i], M[i] = self.harmonize_record(rec, source_domain=source_domain)
        return X, M

    def harmonize_dataframe(
        self,
        df: pd.DataFrame,
        source_domain: str = "auto"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Harmonize a full pandas DataFrame into:
        1. harmonized_features_df: (N, 36) in exact enterprise order
        2. missingness_mask_df: (N, 36) binary indicators
        """
        n_rows = len(df)
        feature_mat = np.zeros((n_rows, len(self.feature_names)), dtype=np.float32)
        mask_mat = np.zeros((n_rows, len(self.feature_names)), dtype=np.float32)

        for r_idx in range(n_rows):
            row_dict = df.iloc[r_idx].to_dict()
            feat_vec, mask_vec = self.harmonize_record(row_dict, source_domain=source_domain)
            feature_mat[r_idx] = feat_vec
            mask_mat[r_idx] = mask_vec

        feat_df = pd.DataFrame(feature_mat, columns=self.feature_names, index=df.index)
        mask_df = pd.DataFrame(mask_mat, columns=[f"{col}_mask" for col in self.feature_names], index=df.index)
        return feat_df, mask_df

    def _extract_value(self, feat_name: str, flat_dict: Dict[str, Any]) -> Optional[float]:
        """Maps canonical feature names across known aliases in diverse datasets."""
        # Direct match
        if feat_name in flat_dict and flat_dict[feat_name] is not None:
            return float(flat_dict[feat_name])

        # Aliases mapping
        aliases: Dict[str, List[str]] = {
            "cpu_temp": ["temp_package_c", "cpu_temp_package", "cpu_temperature", "Process temperature [K]"],
            "cpu_load": ["load_percent", "cpu_load_percent", "cpu_utilization", "cpu_util"],
            "cpu_freq": ["frequency_mhz", "cpu_frequency_mhz", "clock_mhz", "Rotational speed [rpm]"],
            "gpu_temp": ["temp_core_c", "gpu_temperature", "gpu_temp_c"],
            "gpu_power": ["power_draw_w", "gpu_power_w", "power_w", "Torque [Nm]"],
            "gpu_util": ["sm_utilization_pct", "gpu_utilization", "utilization_gpu"],
            "gpu_mem_used": ["vram_used_mb", "gpu_memory_used_mb", "memory_used"],
            "gpu_clock": ["sm_clock_mhz", "graphics_clock_mhz"],
            "gpu_hotspot_delta": ["hotspot_delta_c"],
            "gpu_nvml_ecc_errors": ["ecc_sbe_count", "ecc_dbe_count"],
            "ram_used_gb": ["used_gb", "memory_used_gb"],
            "ram_total_gb": ["total_gb", "memory_total_gb"],
            "ram_page_faults": ["page_faults_sec", "page_faults"],
            "mem_ecc_ce_burst_rate": ["ecc_ce_burst_rate_per_min", "ecc_burst_rate", "ce_count", "mcelog_count"],
            "storage_nvme_wear_pct": ["percentage_used", "wear_leveling_pct"],
            "storage_nvme_spare_pct": ["available_spare_pct", "available_spare"],
            "smart_5_raw": ["smart_5", "smart_5_raw", "reallocated_sector_count", "reallocated_sectors"],
            "smart_9_raw": ["smart_9", "smart_9_raw", "power_on_hours"],
            "smart_187_raw": ["smart_187", "smart_187_raw", "reported_uncorrectable_errors"],
            "smart_188_raw": ["smart_188", "smart_188_raw", "command_timeout"],
            "smart_194_raw": ["smart_194", "smart_194_raw", "temperature_c", "disk_temp"],
            "smart_197_raw": ["smart_197", "smart_197_raw", "current_pending_sector_count"],
            "smart_198_raw": ["smart_198", "smart_198_raw", "offline_uncorrectable_sector_count"],
            "psu_voltage_3v3_dev": ["dev_3v3", "psu_3v3_dev"],
            "psu_voltage_12v_dev": ["dev_12v", "psu_12v_dev"],
            "psu_rail_12v_sag": ["rail_12v_sag_v", "voltage_12v_sag"],
            "fan_rpm_deviation": ["rpm_deviation_pct", "fan_deviation"],
            "ambient_temp": ["ambient_temp_c", "environment_ambient_temp", "Air temperature [K]"],
            "humidity": ["humidity_pct", "environment_humidity"],
            "vibration_rms": ["vibration_rms_g", "vibration", "environment_vibration_rms"],
            "acoustic_rms": ["acoustic_rms_v", "acoustic", "environment_acoustic_rms"],
            "system_uptime": ["uptime_hours", "uptime", "Tool wear [min]"],
        }

        for alias in aliases.get(feat_name, []):
            if alias in flat_dict and flat_dict[alias] is not None:
                val = flat_dict[alias]
                # Special conversion for Kelvin
                if "temperature [K]" in alias:
                    return float(val) - 273.15
                return float(val)

        return None

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flattens nested dictionaries into single-level dot/underscore keys."""
        items: Dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            elif isinstance(v, list) and v:
                if isinstance(v[0], dict):
                    # Take first instance for flat representation
                    items.update(self._flatten_dict(v[0], new_key))
                elif isinstance(v[0], (int, float)):
                    items[new_key] = float(np.mean(v))
            else:
                items[new_key] = v
                items[k] = v  # also index raw key for fast alias access
        return items
