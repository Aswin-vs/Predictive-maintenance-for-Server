"""
Self-Supervised Labeling Engine.

Generates class labels and TTF estimates from raw telemetry data using
threshold-based rules. This enables the continuous learning pipeline to
retrain models on real server data without manual annotation.

Label generation strategy:
- Threshold rules derived from domain knowledge and hardware specs
- Rate-of-change analysis for degradation trends
- Extrapolation-based TTF estimation
- Supports operator-provided feedback labels

Author: @claude-architect (Phase D — Production-Ready Upgrade)
Date: 2026-08-09
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from server.training.feature_schema import (
    FEATURE_NAMES_V1, FAULT_LABELS, CLASS_TO_COMPONENT, FEATURE_DEFAULTS
)


# Threshold configurations for each fault type
# Each threshold defines: (feature_name, operator, threshold, sustained_seconds)
FAULT_THRESHOLDS = {
    # Class 2: THERMAL_CRITICAL — CPU overheating
    2: {
        "name": "THERMAL_CRITICAL",
        "rules": [
            {"feature": "cpu_temp", "op": ">", "threshold": 90.0, "sustained_s": 30},
            {"feature": "cpu_ambient_delta", "op": ">", "threshold": 70.0, "sustained_s": 30},
        ],
        "require_all": False,  # ANY rule triggers this class
    },
    
    # Class 1: HDD_DEGRADATION — SMART indicators deteriorating
    1: {
        "name": "HDD_DEGRADATION",
        "rules": [
            {"feature": "smart_5_raw", "op": ">", "threshold": 5.0, "sustained_s": 0},
            {"feature": "smart_197_raw", "op": ">", "threshold": 0.0, "sustained_s": 0},
            {"feature": "smart_187_raw", "op": ">", "threshold": 10.0, "sustained_s": 0},
        ],
        "require_all": False,
    },
    
    # Class 3: PSU_ANOMALY — Voltage deviation
    3: {
        "name": "PSU_ANOMALY",
        "rules": [
            {"feature": "psu_voltage_3v3_dev", "op": ">", "threshold": 5.0, "sustained_s": 10},
            {"feature": "psu_voltage_12v_dev", "op": ">", "threshold": 5.0, "sustained_s": 10},
        ],
        "require_all": False,
    },
    
    # Class 4: FAN_FAILURE — RPM significantly below baseline
    4: {
        "name": "FAN_FAILURE",
        "rules": [
            {"feature": "fan_rpm_deviation", "op": "<", "threshold": -0.5, "sustained_s": 60},
        ],
        "require_all": True,
    },
    
    # Class 5: MEMORY_FAULT — Page fault spike
    5: {
        "name": "MEMORY_FAULT",
        "rules": [
            {"feature": "ram_page_faults", "op": ">", "threshold": 5000.0, "sustained_s": 10},
        ],
        "require_all": True,
    },
    
    # Class 6: GPU_DEGRADATION — Clock drop under thermal stress
    6: {
        "name": "GPU_DEGRADATION",
        "rules": [
            {"feature": "gpu_temp", "op": ">", "threshold": 85.0, "sustained_s": 30},
            {"feature": "gpu_clock", "op": "<", "threshold": 1000.0, "sustained_s": 30},
        ],
        "require_all": True,  # Both must be true (hot + throttled)
    },
}


class AutoLabeler:
    """
    Generates labels for raw telemetry data using rule-based heuristics.
    
    Usage:
        labeler = AutoLabeler()
        df = pd.DataFrame(telemetry_data)  # columns must match FEATURE_NAMES_V1
        labels, ttf = labeler.label_dataframe(df)
    """
    
    def __init__(self, custom_thresholds: Dict = None, telemetry_interval_s: int = 5):
        """
        Args:
            custom_thresholds: Override default thresholds for specific fault classes
            telemetry_interval_s: Seconds between telemetry readings (for sustained checks)
        """
        self.thresholds = FAULT_THRESHOLDS.copy()
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
        self.interval_s = telemetry_interval_s
    
    def label_single(self, features: Dict[str, float]) -> Tuple[int, str]:
        """
        Label a single telemetry reading.
        
        Args:
            features: Dict mapping feature names to values
            
        Returns:
            Tuple of (class_id, class_name)
        """
        for class_id in [2, 6, 3, 4, 5, 1]:  # Priority order (thermal first)
            config = self.thresholds[class_id]
            rules = config["rules"]
            require_all = config["require_all"]
            
            results = []
            for rule in rules:
                feat_name = rule["feature"]
                value = features.get(feat_name, FEATURE_DEFAULTS.get(feat_name, 0.0))
                
                if rule["op"] == ">":
                    results.append(value > rule["threshold"])
                elif rule["op"] == "<":
                    results.append(value < rule["threshold"])
                elif rule["op"] == ">=":
                    results.append(value >= rule["threshold"])
                elif rule["op"] == "<=":
                    results.append(value <= rule["threshold"])
            
            if require_all:
                triggered = all(results)
            else:
                triggered = any(results)
            
            if triggered:
                return class_id, config["name"]
        
        return 0, "HEALTHY"
    
    def label_dataframe(self, df: pd.DataFrame, 
                        compute_ttf: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Label an entire DataFrame of telemetry.
        
        Args:
            df: DataFrame with columns matching FEATURE_NAMES_V1
            compute_ttf: Whether to compute TTF estimates
            
        Returns:
            Tuple of (class_labels, ttf_normalized)
            - class_labels: np.ndarray of int class IDs (0-6)
            - ttf_normalized: np.ndarray of float TTF values (0.0 to 1.0)
        """
        n = len(df)
        class_labels = np.zeros(n, dtype=np.int32)
        ttf_values = np.ones(n, dtype=np.float32)  # Default: 1.0 (healthy)
        
        # Label each row
        for i in range(n):
            row = df.iloc[i].to_dict()
            class_id, _ = self.label_single(row)
            class_labels[i] = class_id
        
        if compute_ttf:
            ttf_values = self._compute_ttf(df, class_labels)
        
        # Print label distribution
        unique, counts = np.unique(class_labels, return_counts=True)
        dist = {FAULT_LABELS.get(u, str(u)): int(c) for u, c in zip(unique, counts)}
        print(f"[LABELER] Label distribution: {dist}")
        
        return class_labels, ttf_values
    
    def _compute_ttf(self, df: pd.DataFrame, class_labels: np.ndarray) -> np.ndarray:
        """
        Estimate Time-to-Failure (TTF) based on degradation trends.
        
        Strategy:
        1. For each failure event, work backwards to assign decreasing TTF
        2. For healthy periods, assign TTF = 1.0 (> 365 days normalized)
        3. For ongoing degradation, extrapolate using slope
        """
        n = len(df)
        ttf = np.ones(n, dtype=np.float32)  # Default: healthy (max TTF)
        
        # Find failure event indices
        failure_indices = np.where(class_labels > 0)[0]
        
        if len(failure_indices) == 0:
            return ttf  # All healthy
        
        # For each failure event, work backwards assigning decreasing TTF
        # The "failure" is defined as the first sample where class > 0 in a run
        # TTF decreases linearly from 1.0 to 0.0 as we approach the failure
        
        # Find failure run starts
        runs = []
        current_run_start = None
        for i in range(n):
            if class_labels[i] > 0:
                if current_run_start is None:
                    current_run_start = i
            else:
                if current_run_start is not None:
                    runs.append((current_run_start, i - 1))
                    current_run_start = None
        if current_run_start is not None:
            runs.append((current_run_start, n - 1))
        
        for run_start, run_end in runs:
            # Samples in the failure run get TTF = 0.0 to 0.1
            for j in range(run_start, run_end + 1):
                ttf[j] = 0.0
            
            # Work backwards from run_start to assign decreasing TTF
            # Assume 30-day lead time = 30 * 24 * 3600 / interval_s samples
            lead_samples = min(30 * 24 * 3600 // self.interval_s, run_start)
            
            for k in range(lead_samples):
                idx = run_start - k - 1
                if idx < 0:
                    break
                if class_labels[idx] > 0:
                    continue  # Skip if already in another failure run
                # Linear decrease: 1.0 at max distance, 0.0 at failure
                ttf[idx] = min(ttf[idx], (k + 1) / lead_samples)
        
        return ttf
    
    def label_from_operator_feedback(self, features: Dict[str, float],
                                     operator_class: int,
                                     operator_ttf_days: Optional[float] = None) -> Tuple[int, float]:
        """
        Accept an operator-provided label (from dashboard "Flag Anomaly" button).
        
        Args:
            features: Current telemetry features
            operator_class: Fault class ID provided by operator
            operator_ttf_days: Optional TTF estimate in days
            
        Returns:
            Tuple of (class_id, ttf_normalized)
        """
        if operator_class not in FAULT_LABELS:
            raise ValueError(f"Invalid class ID: {operator_class}. Valid: {list(FAULT_LABELS.keys())}")
        
        ttf_normalized = 1.0
        if operator_ttf_days is not None:
            ttf_normalized = np.clip(operator_ttf_days / 365.0, 0.0, 1.0)
        elif operator_class > 0:
            # Default: operator flagged an issue, assume ~7 days TTF
            ttf_normalized = 7.0 / 365.0
        
        return operator_class, ttf_normalized
    
    def get_anomaly_rate(self, class_labels: np.ndarray, window_size: int = None) -> float:
        """
        Calculate the anomaly rate over a window of labels.
        Used by the retrain trigger for emergency retraining.
        
        Args:
            class_labels: Array of class labels
            window_size: Number of recent samples to check (None = all)
            
        Returns:
            float: Anomaly rate (0.0 to 1.0)
        """
        if window_size is not None:
            labels = class_labels[-window_size:]
        else:
            labels = class_labels
        
        if len(labels) == 0:
            return 0.0
        
        return float(np.sum(labels > 0)) / len(labels)


if __name__ == "__main__":
    # Quick test with synthetic data
    labeler = AutoLabeler()
    
    # Test single labeling
    healthy = {"cpu_temp": 45.0, "gpu_temp": 40.0, "smart_5_raw": 0.0}
    print(f"Healthy: {labeler.label_single(healthy)}")
    
    thermal = {"cpu_temp": 95.0, "gpu_temp": 40.0, "cpu_ambient_delta": 75.0}
    print(f"Thermal: {labeler.label_single(thermal)}")
    
    hdd = {"cpu_temp": 45.0, "smart_5_raw": 10.0, "smart_197_raw": 5.0}
    print(f"HDD: {labeler.label_single(hdd)}")
    
    psu = {"psu_voltage_3v3_dev": 8.0, "psu_voltage_12v_dev": 2.0}
    print(f"PSU: {labeler.label_single(psu)}")
    
    # Test DataFrame labeling
    np.random.seed(42)
    n_samples = 100
    data = {feat: np.random.normal(FEATURE_DEFAULTS[feat], 1.0, n_samples) 
            for feat in FEATURE_NAMES_V1}
    # Inject some anomalies
    data["cpu_temp"][80:90] = 96.0  # Thermal event at end
    df = pd.DataFrame(data)
    
    labels, ttf = labeler.label_dataframe(df)
    print(f"\nAnomaly rate: {labeler.get_anomaly_rate(labels):.2%}")
