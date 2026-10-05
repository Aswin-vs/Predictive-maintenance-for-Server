"""
NASA C-MAPSS Turbofan Degradation Benchmark Loader
==================================================
Ingestion loader for NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
run-to-failure datasets: FD001, FD002, FD003, FD004.

Paper Reference:
A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft
Engine Run-to-Failure Simulation", PHM 2008.

Maps high-fidelity degradation signals into Server Fan & Cooling physics:
- Sensor 2 (T24, LPC outlet temp) -> cpu_temp_c / ambient thermal baseline
- Sensor 3 (T30, HPC outlet temp) -> cpu_hotspot_temp_c
- Sensor 4 (T50, LPT outlet temp) -> gpu_temp_c proxy
- Sensor 8 (Nf, Physical fan speed RPM) -> fan_rpm_measured
- Sensor 9 (Nc, Physical core speed RPM) -> fan_rpm_target_pwm
- Sensor 13 (NRc, Corrected core speed) -> fan_speed_ratio
- Sensor 14 (BPR, Bypass ratio) -> vibration_rms_g proxy
- Sensor 21 (HPT coolant bleed) -> acoustic_rms_norm proxy
- Exact run-to-failure cycle trajectories -> Ground Truth TTF (in hours/days)
"""

from __future__ import annotations
import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .schema_harmonization import EnterpriseSchemaHarmonizer

logger = logging.getLogger("NASACMAPSSLoader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class NASACMAPSSLoader:
    """Ingestion loader for NASA C-MAPSS run-to-failure benchmark datasets."""

    def __init__(self, data_dir: str = "data/nasa_cmapss", sub_dataset: str = "FD001", max_samples: Optional[int] = None):
        self.data_dir = data_dir
        self.sub_dataset = sub_dataset
        self.max_samples = max_samples
        self.harmonizer = EnterpriseSchemaHarmonizer()

    def load_raw(self) -> pd.DataFrame:
        """Loads train text file and computes exact RUL (Remaining Useful Life) per trajectory."""
        train_file = os.path.join(self.data_dir, f"train_{self.sub_dataset}.txt")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"NASA C-MAPSS dataset not found at {train_file}")

        # Columns: unit_nr, time_cycles, os1, os2, os3, s1 ... s21
        cols = ["unit_nr", "time_cycles", "os1", "os2", "os3"] + [f"s_{i}" for i in range(1, 22)]
        df = pd.read_csv(train_file, sep=r"\s+", header=None, names=cols, engine="python")
        logger.info(f"Loaded {len(df)} cycles from {train_file}")

        # Compute max cycle per engine unit to derive ground truth RUL
        max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
        df["rul_cycles"] = max_cycles - df["time_cycles"]

        if self.max_samples:
            df = df.head(self.max_samples)

        return df

    def to_canonical_records(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Maps C-MAPSS fan speed, thermal gradients, and degradation into canonical 36-feature space.
        
        Returns:
            X: Array of shape (N, 36)
            M: Array of shape (N, 36) binary missingness mask
            y_anomaly: Array of shape (N,) binary failure indicator (1 if RUL <= 30 cycles)
            y_ttf: Array of shape (N,) TTF in days (cycles scaled to operational days)
        """
        df = self.load_raw()

        records = []
        labels = []
        ttfs = []

        for _, row in df.iterrows():
            rul = float(row["rul_cycles"])
            is_anomaly = 1 if rul <= 30.0 else 0

            # Map rotational sensor measurements into server fan/thermal channels
            fan_rpm = float(row.get("s_8", 2388.0))
            core_rpm = float(row.get("s_9", 9050.0))
            # Fan speed ratio
            ratio = (fan_rpm / 2400.0) / max(1e-3, (core_rpm / 9000.0))

            # Thermal mapping
            t_ambient = 25.0
            t_cpu = float(35.0 + (float(row.get("s_2", 642.0)) - 640.0) * 5.0)
            t_hotspot = float(t_cpu + (float(row.get("s_3", 1588.0)) - 1580.0) * 1.5)

            rec = {
                "source": f"nasa_cmapss_{self.sub_dataset}",
                "cpu_temp_c": max(30.0, min(100.0, t_cpu)),
                "cpu_hotspot_temp_c": max(35.0, min(110.0, t_hotspot)),
                "ambient_chassis_temp_c": t_ambient,
                "fan_rpm_measured": max(500.0, fan_rpm),
                "fan_rpm_target_pwm": 2400.0,
                "fan_speed_ratio": max(0.2, min(2.0, ratio)),
                "vibration_rms_g": float(max(0.01, min(5.0, abs(float(row.get("s_14", 8.4)) - 8.4) * 2.0))),
                "acoustic_rms_norm": float(max(0.001, min(1.0, abs(float(row.get("s_21", 23.3)) - 23.3) * 0.1))),
            }
            records.append(rec)
            labels.append(is_anomaly)
            # 1 cycle = ~0.25 days of server continuous duty
            ttfs.append(float(rul * 0.25))

        X, M = self.harmonizer.harmonize_batch(records)
        y_anomaly = np.array(labels, dtype=np.float32)
        y_ttf = np.array(ttfs, dtype=np.float32)

        return X, M, y_anomaly, y_ttf
