"""
H2.1: Adaptive Baseline Engine — Phase H
==========================================
Per-server, per-component statistical baseline learning.

Workflow:
  1. CALIBRATION (first 24h): Collects raw metrics, builds distribution profiles
  2. DETECTION (24h+): Compares live data against learned baselines using Z-scores
  3. REFINEMENT (72h+): Continuously updates baselines with exponential moving average

Baselines are stored as JSON files in edge/baselines/{server_id}.json
and auto-persist every 30 minutes.
"""

import os
import json
import time
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

from edge.edge_logger import get_logger

logger = get_logger("adaptive_baseline")

BASELINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")
os.makedirs(BASELINES_DIR, exist_ok=True)

# Default calibration window (seconds)
DEFAULT_CALIBRATION_WINDOW = 24 * 3600  # 24 hours
# Minimum samples needed before a metric is considered calibrated
MIN_CALIBRATION_SAMPLES = 100
# EMA decay for continuous baseline updates after calibration
EMA_ALPHA = 0.001  # Very slow decay — baselines shift gradually


@dataclass
class MetricBaseline:
    """Statistical baseline for a single metric."""
    mean: float = 0.0
    std: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    p5: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    count: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'MetricBaseline':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


ACTIVE_BASELINES: Dict[str, 'AdaptiveBaseline'] = {}


class AdaptiveBaseline:
    """
    Per-server adaptive baseline engine.

    Collects metrics during calibration, computes statistical baselines,
    then uses Z-score anomaly detection for live monitoring.
    """

    def __init__(self, server_id: str, calibration_window: int = DEFAULT_CALIBRATION_WINDOW):
        self.server_id = server_id
        self.calibration_window = calibration_window
        self.start_time = time.time()

        # Raw values collected during calibration: {metric_key: [values]}
        self._calibration_buffer: Dict[str, List[float]] = {}
        # Computed baselines: {metric_key: MetricBaseline}
        self.baselines: Dict[str, MetricBaseline] = {}
        # Component registry: {component_id: component_type}
        self.components: Dict[str, str] = {}

        self._is_calibrated = False
        self._last_persist = time.time()
        self._persist_interval = 1800  # 30 min

        # Try to load existing baselines
        self._load()

        # Register in active singleton registry
        ACTIVE_BASELINES[server_id] = self

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def calibration_progress(self) -> float:
        """Returns calibration progress 0.0 to 1.0."""
        if self._is_calibrated:
            return 1.0
        elapsed = time.time() - self.start_time
        time_progress = min(1.0, elapsed / self.calibration_window)
        # Also check sample count
        if self._calibration_buffer:
            min_samples = min(len(v) for v in self._calibration_buffer.values())
            sample_progress = min(1.0, min_samples / MIN_CALIBRATION_SAMPLES)
            return min(time_progress, sample_progress)
        return 0.0

    @property
    def total_samples(self) -> int:
        if self._calibration_buffer:
            return sum(len(v) for v in self._calibration_buffer.values())
        return sum(b.count for b in self.baselines.values())

    @property
    def metric_count(self) -> int:
        return len(self.baselines) if self._is_calibrated else len(self._calibration_buffer)

    def _metric_key(self, component_id: str, metric_name: str) -> str:
        """Create a unique key for a component+metric pair."""
        return f"{component_id}/{metric_name}"

    # ─── Ingestion ───────────────────────────────────────────────────────

    def ingest_v2_payload(self, v2_payload: Dict[str, Any]):
        """
        Ingest a full v2 telemetry payload.
        Extracts all metrics from all components automatically.
        """
        ts = v2_payload.get("timestamp", datetime.now(timezone.utc).isoformat())

        # CPU metrics (v2 has cpus: [{...}])
        cpus = v2_payload.get("cpus", [])
        if cpus:
            cpu = cpus[0] if isinstance(cpus, list) else cpus
            self.components["cpu-0"] = "cpu"
            for key in ["load_percent", "frequency_mhz", "temp_package_c", "package_power_w"]:
                if key in cpu and cpu[key] is not None:
                    self.ingest("cpu-0", key, float(cpu[key]))
            # Per-core temps
            for i, temp in enumerate(cpu.get("temp_cores_c", [])):
                if temp is not None:
                    self.ingest(f"cpu-core-{i}", "temp_c", float(temp))
        # Fallback: single cpu dict (legacy)
        elif "cpu" in v2_payload and isinstance(v2_payload["cpu"], dict):
            cpu = v2_payload["cpu"]
            self.components["cpu-0"] = "cpu"
            for key in ["load_percent", "frequency_mhz", "temp_package_c", "package_power_w"]:
                if key in cpu and cpu[key] is not None:
                    self.ingest("cpu-0", key, float(cpu[key]))

        # GPU metrics (v2 has gpus: [{...}, ...] — multi-GPU)
        gpus = v2_payload.get("gpus", [])
        for gi, gpu in enumerate(gpus):
            gpu_id = gpu.get("id", f"GPU{gi}").lower().replace(" ", "")
            self.components[gpu_id] = "gpu"
            for key in ["temp_core_c", "temp_junction_c", "power_draw_w", "vram_used_mb",
                         "vram_total_mb", "clock_mhz", "sm_utilization_pct", "fan_speed_pct",
                         "ecc_sbe_count", "ecc_dbe_count"]:
                if key in gpu and gpu[key] is not None:
                    self.ingest(gpu_id, key, float(gpu[key]))

        # Motherboard metrics (v2 has motherboard: {...})
        mb = v2_payload.get("motherboard", {})
        if mb and isinstance(mb, dict):
            self.components["motherboard-0"] = "motherboard"
            for key in ["pch_temp_c", "vrm_temp_c", "system_temp_c", "cmos_battery_voltage"]:
                if key in mb and mb[key] is not None:
                    self.ingest("motherboard-0", key, float(mb[key]))

        # Storage (v2 has storage: [{...}, ...] — multi-disk HDD/SSD/NVMe)
        drives = v2_payload.get("storage", [])
        for i, drive in enumerate(drives):
            drive_id = drive.get("id", f"disk-{i}")
            drive_type = drive.get("type", "Unknown").lower()
            self.components[drive_id] = f"storage-{drive_type}"
            for key in ["used_pct", "capacity_gb", "used_gb", "available_spare_pct", "percentage_used"]:
                if key in drive and drive[key] is not None:
                    self.ingest(drive_id, key, float(drive[key]))
            # SMART attributes
            smart = drive.get("smart", {})
            for skey, sval in smart.items():
                if isinstance(sval, (int, float)):
                    self.ingest(drive_id, f"smart_{skey}", float(sval))

        # Memory (v2 has memory: [{...}, ...] — multi-DIMM)
        mem_list = v2_payload.get("memory", [])
        if isinstance(mem_list, dict):
            mem_list = [mem_list]
        for mi, mem in enumerate(mem_list):
            mem_id = f"ram-{mi}"
            self.components[mem_id] = "memory"
            for key in ["used_gb", "total_gb", "percent_used", "available_gb",
                         "page_faults_sec", "committed_gb", "ecc_ce_count", "ecc_ue_count"]:
                if key in mem and mem[key] is not None:
                    self.ingest(mem_id, key, float(mem[key]))

        # PSU (v2 has psu: {...} — includes battery for laptops)
        psu = v2_payload.get("psu", {})
        if psu:
            self.components["psu-0"] = "psu"
            for key in ["voltage_3v3", "voltage_5v", "voltage_12v", "total_power_w",
                         "battery_percent"]:
                if key in psu and psu[key] is not None:
                    self.ingest("psu-0", key, float(psu[key]))

        # Fans (v2 has fans: [{...}, ...] — multi-fan)
        fans = v2_payload.get("fans", [])
        if isinstance(fans, dict):
            fans = fans.get("fans", [])
        for i, fan in enumerate(fans if isinstance(fans, list) else []):
            fan_id = fan.get("id", f"fan-{i}")
            self.components[fan_id] = "fan"
            for key in ["rpm", "pwm_duty_pct", "rpm_deviation_pct"]:
                if key in fan and fan[key] is not None:
                    self.ingest(fan_id, key, float(fan[key]))

        # Environment
        env = v2_payload.get("environment", {})
        if env:
            self.components["env-0"] = "environment"
            for key in ["ambient_temp_c", "humidity_pct", "vibration_rms_g", "acoustic_rms_v"]:
                if key in env and env[key] is not None:
                    self.ingest("env-0", key, float(env[key]))

        # System uptime
        uptime = v2_payload.get("uptime_hours")
        if uptime is not None:
            self.components["sys-0"] = "system"
            self.ingest("sys-0", "uptime_hours", float(uptime))

        # Check if calibration complete
        if not self._is_calibrated:
            self._check_calibration()

        # Periodic persist
        if time.time() - self._last_persist > self._persist_interval:
            self._persist()

    def ingest(self, component_id: str, metric_name: str, value: float):
        """Ingest a single metric value."""
        if math.isnan(value) or math.isinf(value):
            return

        key = self._metric_key(component_id, metric_name)

        if not self._is_calibrated:
            # Calibration mode: buffer raw values
            self._calibration_buffer.setdefault(key, []).append(value)
        else:
            # Live mode: update baseline with EMA
            if key in self.baselines:
                b = self.baselines[key]
                b.mean = (1 - EMA_ALPHA) * b.mean + EMA_ALPHA * value
                # Update running std approximation
                diff = value - b.mean
                b.std = math.sqrt((1 - EMA_ALPHA) * b.std**2 + EMA_ALPHA * diff**2)
                b.count += 1
                b.min_val = min(b.min_val, value)
                b.max_val = max(b.max_val, value)
                b.last_updated = datetime.now(timezone.utc).isoformat()
            else:
                # New metric discovered after calibration — start fresh mini-buffer
                self._calibration_buffer.setdefault(key, []).append(value)
                if len(self._calibration_buffer[key]) >= MIN_CALIBRATION_SAMPLES:
                    self._compute_baseline_for_key(key)

    # ─── Calibration ─────────────────────────────────────────────────────

    def _check_calibration(self):
        """Check if calibration window has elapsed and enough samples collected."""
        elapsed = time.time() - self.start_time
        if elapsed < self.calibration_window:
            return

        # Check minimum samples per metric
        if not self._calibration_buffer:
            return

        min_samples = min(len(v) for v in self._calibration_buffer.values())
        if min_samples < MIN_CALIBRATION_SAMPLES:
            return

        # Calibration complete — compute baselines
        logger.info(f"[{self.server_id}] Calibration complete! "
                     f"{len(self._calibration_buffer)} metrics, "
                     f"{sum(len(v) for v in self._calibration_buffer.values())} total samples")
        self._compute_all_baselines()
        self._is_calibrated = True
        self._persist()

    def force_calibrate(self):
        """Force calibration with whatever data is available (for testing/manual trigger)."""
        if not self._calibration_buffer:
            logger.warning(f"[{self.server_id}] No data to calibrate from")
            return False

        min_samples = min(len(v) for v in self._calibration_buffer.values())
        if min_samples < 10:
            logger.warning(f"[{self.server_id}] Not enough data (min {min_samples} samples)")
            return False

        self._compute_all_baselines()
        self._is_calibrated = True
        self._persist()
        logger.info(f"[{self.server_id}] Forced calibration with {len(self.baselines)} metrics")
        return True

    def _compute_all_baselines(self):
        """Compute baselines for all buffered metrics."""
        for key in list(self._calibration_buffer.keys()):
            self._compute_baseline_for_key(key)
        # Clear buffer to free memory
        self._calibration_buffer.clear()

    def _compute_baseline_for_key(self, key: str):
        """Compute baseline statistics for a single metric key."""
        values = self._calibration_buffer.get(key, [])
        if not values:
            return

        arr = np.array(values, dtype=np.float64)
        self.baselines[key] = MetricBaseline(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            p5=float(np.percentile(arr, 5)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            count=len(values),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    # ─── Anomaly Scoring ─────────────────────────────────────────────────

    def score_metric(self, component_id: str, metric_name: str, value: float) -> float:
        """
        Returns anomaly score for a single metric.
        0.0 = perfectly normal, 1.0 = extreme anomaly.
        """
        key = self._metric_key(component_id, metric_name)
        if key not in self.baselines:
            return 0.0  # Unknown metric — can't score

        b = self.baselines[key]
        if b.std < 1e-8:
            # Near-zero variance: constant or simulated metric (e.g. constant voltage defaults)
            if abs(value - b.mean) < 1e-6:
                return 0.0
            if "voltage" in metric_name.lower() or "battery" in metric_name.lower() or "power" in metric_name.lower():
                return 0.0
            return 0.05  # Very mild drift, never trigger an anomaly

        # Domain-aware variance floor: prevent hypersensitive false positives
        # on tight idle baselines when operating within normal hardware bounds
        min_std = 0.5
        metric_lower = metric_name.lower()
        if "percent" in metric_lower or "pct" in metric_lower:
            min_std = 15.0  # Normal 15% load/usage swing
        elif "battery" in metric_lower:
            min_std = 25.0  # Normal battery charging/discharging swing
        elif "used_gb" in metric_lower or "available_gb" in metric_lower or "committed_gb" in metric_lower:
            min_std = 4.0   # Normal 4 GB OS/application allocation swing
        elif "load" in metric_lower or "util" in metric_lower:
            min_std = 15.0  # Normal 15% computational swing
        elif "mhz" in metric_lower or "clock" in metric_lower:
            min_std = 250.0 # Turbo boost / P-state frequency changes
        elif "rpm" in metric_lower:
            min_std = 150.0 # Fan speed thermal regulation
        elif "fault" in metric_lower:
            min_std = 500.0 # Normal Windows page faults swing
        elif "voltage" in metric_lower:
            min_std = 0.5   # Voltage variance floor

        effective_std = max(b.std, min_std)
        z = abs(value - b.mean) / effective_std
        # Smooth sigmoid mapping: z=1→0.0, z=2→0.15, z=3→0.5, z=4→0.85, z=5→0.97
        return 1.0 / (1.0 + math.exp(-1.5 * (z - 3.0)))

    def update_metric(self, component_id: str, metric_name: str, value: float):
        """
        Incrementally update baseline statistics for a single metric using Welford's algorithm.
        Enables tracking canonical v1 features even during cold-start or early ticks.
        """
        if math.isnan(value) or math.isinf(value):
            return
        key = self._metric_key(component_id, metric_name)
        if key not in self.baselines:
            self.baselines[key] = MetricBaseline(
                mean=float(value),
                std=0.0,
                min_val=float(value),
                max_val=float(value),
                count=1,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            return

        b = self.baselines[key]
        b.count += 1
        delta = value - b.mean
        b.mean += delta / b.count
        delta2 = value - b.mean
        if b.count > 1:
            var = ((b.count - 2) * (b.std ** 2) + delta * delta2) / (b.count - 1)
            b.std = math.sqrt(max(0.0, var))
        b.min_val = min(b.min_val, value)
        b.max_val = max(b.max_val, value)
        b.last_updated = datetime.now(timezone.utc).isoformat()

    def get_z_score(self, component_id: str, metric_name: str, value: float) -> float:
        """
        Return the raw signed Z-score for a metric value against its learned baseline.
        Uses the same domain-aware variance floor as score_metric().
        Returns 0.0 if metric is unknown, count < 2, or uncalibrated.
        """
        key = self._metric_key(component_id, metric_name)
        if key not in self.baselines:
            return 0.0

        b = self.baselines[key]
        if b.count < 2 or b.std < 1e-8:
            return 0.0

        min_std = 0.5
        metric_lower = metric_name.lower()
        if "percent" in metric_lower or "pct" in metric_lower:
            min_std = 15.0
        elif "battery" in metric_lower:
            min_std = 25.0
        elif "used_gb" in metric_lower or "available_gb" in metric_lower or "committed_gb" in metric_lower:
            min_std = 4.0
        elif "load" in metric_lower or "util" in metric_lower:
            min_std = 15.0
        elif "mhz" in metric_lower or "clock" in metric_lower:
            min_std = 250.0
        elif "rpm" in metric_lower:
            min_std = 150.0
        elif "fault" in metric_lower:
            min_std = 500.0
        elif "voltage" in metric_lower:
            min_std = 0.5

        effective_std = max(b.std, min_std)
        return (value - b.mean) / effective_std

    def score_component(self, component_id: str, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Score all metrics for a component. Returns (health_score_0to100, per_metric_scores).
        """
        if not metrics:
            return 100.0, {}

        anomaly_scores = {}
        for metric_name, value in metrics.items():
            anomaly_scores[metric_name] = self.score_metric(component_id, metric_name, value)

        if not anomaly_scores:
            return 100.0, anomaly_scores

        # Component health = 100 - max_anomaly * 100 (worst metric dominates)
        max_anomaly = max(anomaly_scores.values())
        avg_anomaly = sum(anomaly_scores.values()) / len(anomaly_scores)
        # Blend: 70% worst metric, 30% average
        blended = 0.7 * max_anomaly + 0.3 * avg_anomaly
        health = max(0.0, min(100.0, 100.0 * (1.0 - blended)))

        return round(health, 1), anomaly_scores

    def get_server_health(self, v2_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute holistic server health from a v2 payload.
        Returns detailed per-component breakdown + overall score.
        """
        if not self._is_calibrated:
            prog = round(self.calibration_progress * 100, 1)
            return {
                "calibrated": False,
                "progress": prog,
                "progress_pct": prog,
                "samples": self.total_samples,
                "total_samples": self.total_samples,
                "metrics": self.metric_count,
                "metric_count": self.metric_count,
                "health_score": None,
                "severity": "calibrating",
            }

        component_scores = {}
        component_details = {}

        # Score each registered component
        for comp_id, comp_type in self.components.items():
            prefix = self._metric_key(comp_id, "")
            # Collect current values for this component from baselines keys
            metrics = {}
            for key, baseline in self.baselines.items():
                if key.startswith(prefix):
                    metric_name = key[len(prefix):]
                    # Get current value from payload (best effort)
                    current_val = self._extract_metric_from_payload(v2_payload, comp_id, metric_name)
                    if current_val is not None:
                        metrics[metric_name] = current_val

            health, scores = self.score_component(comp_id, metrics)
            component_scores[comp_id] = health
            component_details[comp_id] = {
                "type": comp_type,
                "health": health,
                "metric_scores": {k: round(v, 3) for k, v in scores.items()},
                "severity": self._health_to_severity(health),
            }

        # Holistic server health (weighted by component type)
        weights = {
            'cpu': 0.20, 'gpu': 0.15, 'storage': 0.25,
            'memory': 0.15, 'fan': 0.10, 'psu': 0.15,
            'environment': 0.0, 'system': 0.0,
        }

        total_weight = 0.0
        weighted_health = 0.0
        critical_any = False

        for comp_id, health in component_scores.items():
            comp_type = self.components.get(comp_id, "unknown")
            # Match weight by prefix (storage-ssd, storage-hdd, etc.)
            base_type = comp_type.split("-")[0] if "-" in comp_type else comp_type
            w = weights.get(base_type, 0.05)
            total_weight += w
            weighted_health += w * health
            if health < 20:
                critical_any = True

        overall = round(weighted_health / max(total_weight, 0.01), 1)
        severity = "critical" if critical_any else self._health_to_severity(overall)

        return {
            "calibrated": True,
            "progress": 100.0,
            "progress_pct": 100.0,
            "samples": self.total_samples,
            "total_samples": self.total_samples,
            "health_score": overall,
            "severity": severity,
            "components": component_details,
            "component_count": len(self.components),
            "metric_count": len(self.baselines),
            "metrics": len(self.baselines),
        }

    def _extract_metric_from_payload(self, payload: Dict, comp_id: str, metric_name: str) -> Optional[float]:
        """Extract a metric value from v2 payload by component ID and metric name."""
        try:
            if comp_id.startswith("cpu"):
                cpus = payload.get("cpus", payload.get("cpu", {}))
                if isinstance(cpus, list):
                    return cpus[0].get(metric_name) if cpus else None
                elif isinstance(cpus, dict):
                    return cpus.get(metric_name)
            elif comp_id.startswith("gpu"):
                gpus_data = payload.get("gpus", payload.get("gpu", {}))
                if isinstance(gpus_data, list):
                    idx = int(comp_id.replace("gpu", "")) if comp_id.replace("gpu", "").isdigit() else 0
                    if idx < len(gpus_data):
                        return gpus_data[idx].get(metric_name)
                elif isinstance(gpus_data, dict):
                    gpus = gpus_data.get("gpus", [])
                    idx = int(comp_id.replace("gpu", "")) if comp_id.replace("gpu", "").isdigit() else 0
                    if idx < len(gpus):
                        return gpus[idx].get(metric_name)
            elif comp_id.startswith("disk"):
                storage = payload.get("storage", {})
                if isinstance(storage, list):
                    drives = storage
                elif isinstance(storage, dict):
                    drives = storage.get("drives", [])
                else:
                    drives = []
                idx = int(comp_id.split("-")[1]) if "-" in comp_id else 0
                if idx < len(drives):
                    if metric_name.startswith("smart_"):
                        return drives[idx].get("smart", {}).get(metric_name.replace("smart_", ""))
                    return drives[idx].get(metric_name)
            elif comp_id.startswith("ram"):
                mem = payload.get("memory", {})
                if isinstance(mem, list):
                    return mem[0].get(metric_name) if mem else None
                elif isinstance(mem, dict):
                    return mem.get(metric_name)
            elif comp_id.startswith("psu"):
                psu = payload.get("psu", {})
                if isinstance(psu, dict):
                    return psu.get(metric_name)
            elif comp_id.startswith("fan"):
                fans_data = payload.get("fans", {})
                if isinstance(fans_data, list):
                    fans = fans_data
                elif isinstance(fans_data, dict):
                    fans = fans_data.get("fans", [])
                else:
                    fans = []
                idx = int(comp_id.split("-")[1]) if "-" in comp_id else 0
                if idx < len(fans):
                    return fans[idx].get(metric_name)
            elif comp_id.startswith("env"):
                env = payload.get("environment", {})
                if isinstance(env, dict):
                    return env.get(metric_name)
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        return None

    @staticmethod
    def _health_to_severity(health: float) -> str:
        if health >= 80: return "healthy"
        if health >= 60: return "monitoring"
        if health >= 40: return "warning"
        if health >= 20: return "degraded"
        return "critical"

    def reset(self):
        """Reset all calibration buffers, learned baselines, and delete persisted JSON file."""
        self._calibration_buffer.clear()
        self.baselines.clear()
        self.components.clear()
        self._is_calibrated = False
        self.start_time = time.time()
        path = self._persist_path()
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"[{self.server_id}] Could not delete baseline file: {e}")
        logger.info(f"[{self.server_id}] Adaptive baseline engine fully reset.")


    # ─── Persistence ─────────────────────────────────────────────────────

    def _persist_path(self) -> str:
        safe_id = self.server_id.replace("/", "_").replace("\\", "_")
        return os.path.join(BASELINES_DIR, f"{safe_id}.json")

    def _persist(self):
        """Save baselines to disk."""
        data = {
            "server_id": self.server_id,
            "is_calibrated": self._is_calibrated,
            "start_time": self.start_time,
            "components": self.components,
            "baselines": {k: v.to_dict() for k, v in self.baselines.items()},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._persist_path()
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self._last_persist = time.time()
            logger.debug(f"[{self.server_id}] Baselines persisted ({len(self.baselines)} metrics)")
        except Exception as e:
            logger.error(f"[{self.server_id}] Failed to persist baselines: {e}")

    def _load(self):
        """Load baselines from disk if they exist."""
        path = self._persist_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._is_calibrated = data.get("is_calibrated", False)
            self.start_time = data.get("start_time", self.start_time)
            self.components = data.get("components", {})
            for key, bdict in data.get("baselines", {}).items():
                self.baselines[key] = MetricBaseline.from_dict(bdict)
            logger.info(f"[{self.server_id}] Loaded baselines: "
                        f"{len(self.baselines)} metrics, calibrated={self._is_calibrated}")
        except Exception as e:
            logger.warning(f"[{self.server_id}] Failed to load baselines: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current baseline engine status for API/dashboard."""
        return {
            "server_id": self.server_id,
            "calibrated": self._is_calibrated,
            "progress_pct": round(self.calibration_progress * 100, 1),
            "total_samples": self.total_samples,
            "metric_count": self.metric_count,
            "component_count": len(self.components),
            "components": dict(self.components),
            "calibration_window_h": self.calibration_window / 3600,
            "elapsed_h": round((time.time() - self.start_time) / 3600, 1),
        }

    def get_baselines_summary(self) -> Dict[str, Any]:
        """Get a summary of all baselines for dashboard display."""
        summary = {}
        for key, b in self.baselines.items():
            summary[key] = {
                "mean": round(b.mean, 3),
                "std": round(b.std, 3),
                "range": [round(b.min_val, 3), round(b.max_val, 3)],
                "p95": round(b.p95, 3),
                "count": b.count,
            }
        return summary


def reset_all_baselines(server_id: Optional[str] = None):
    """Reset all active baselines in memory and delete on disk."""
    if server_id and server_id.lower() not in ("all", "none", "", "default"):
        if server_id in ACTIVE_BASELINES:
            ACTIVE_BASELINES[server_id].reset()
            ACTIVE_BASELINES.pop(server_id, None)
        safe_id = server_id.replace("/", "_").replace("\\", "_")
        bpath = os.path.join(BASELINES_DIR, f"{safe_id}.json")
        if os.path.exists(bpath):
            try:
                os.remove(bpath)
            except OSError:
                pass
        logger.info(f"Reset adaptive baseline for server '{server_id}'")
    else:
        for sid, abl in list(ACTIVE_BASELINES.items()):
            try:
                abl.reset()
            except Exception:
                pass
        ACTIVE_BASELINES.clear()
        if os.path.exists(BASELINES_DIR):
            for f in os.listdir(BASELINES_DIR):
                if f.endswith(".json"):
                    try:
                        os.remove(os.path.join(BASELINES_DIR, f))
                    except OSError:
                        pass
        logger.info("Reset all adaptive baselines across all servers")

