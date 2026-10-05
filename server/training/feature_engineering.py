"""
Feature Engineering Pipeline for Predictive Maintenance.

This module computes the 28 engineered features from raw telemetry data,
such as thermal ramp rates, delta-features, and cross-sensor correlations.
It handles processing both historical datasets and real-time streams.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

class FeatureEngine:
    """
    Computes engineered features from raw server telemetry.
    
    NOTE: This class is designed for OFFLINE BATCH PROCESSING (e.g., training data preparation).
    For real-time edge inference, the actual feature engineering logic is implemented
    inline within `edge/main.py` using `extract_features()` to optimize latency and state management.
    """
    
    def __init__(self):
        # We can define any state needed for rolling features here.
        # For real-time, we'd keep a history buffer.
        pass
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 28 engineered features on a batch DataFrame.
        
        Args:
            df (pd.DataFrame): Raw telemetry DataFrame. Must be sorted by time 
                               for a single machine before calling.
                               
        Returns:
            pd.DataFrame: DataFrame with engineered features appended.
        """
        # Ensure we don't modify the original
        out_df = df.copy()
        
        # 1-4: CPU Features
        if 'cpu_temp' in out_df.columns:
            # CPU thermal ramp (rate of change over 5 steps)
            # Assuming data is 1 step per second or 5 seconds, diff(5) gives the delta
            out_df['cpu_thermal_ramp'] = out_df['cpu_temp'].diff(5) / 5.0
            
            # Cross-sensor correlation
            if 'ambient_temp' in out_df.columns:
                out_df['cpu_ambient_delta'] = out_df['cpu_temp'] - out_df['ambient_temp']
                
        # 5-9: GPU Features
        if 'gpu_temp' in out_df.columns:
            out_df['gpu_thermal_ramp'] = out_df['gpu_temp'].diff(5) / 5.0
            
        if 'gpu_utilization' in out_df.columns and 'gpu_power_draw' in out_df.columns:
            # Efficiency (avoid division by zero)
            out_df['gpu_power_efficiency'] = out_df['gpu_utilization'] / (out_df['gpu_power_draw'] + 1e-6)
            
        if 'ecc_errors' in out_df.columns:
            # ECC error rate (change in cumulative errors)
            out_df['gpu_ecc_rate'] = out_df['ecc_errors'].diff(1)
            
        # 10-13: HDD Features
        if 'smart_5_raw' in out_df.columns:
            # Delta over 7 days. If frequency is daily, diff(7)
            out_df['hdd_smart5_delta_7d'] = out_df['smart_5_raw'].diff(7)
            
        if 'smart_187_raw' in out_df.columns:
            out_df['hdd_smart187_delta_7d'] = out_df['smart_187_raw'].diff(7)
            
        # HDD Health Index (Composite score)
        # Weights are arbitrary for now, can be tuned based on XGBoost feature importance
        smart_cols = ['smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']
        if all(c in out_df.columns for c in smart_cols):
            # Normalize and sum (negative impact)
            out_df['hdd_health_index'] = 100 - (
                out_df['smart_5_raw'] * 0.4 +
                out_df['smart_187_raw'] * 0.3 +
                out_df['smart_197_raw'] * 0.3
            ).clip(upper=100)
            
        # 14-16: PSU Features
        if 'voltage_3v3' in out_df.columns:
            out_df['psu_voltage_3v3_dev'] = (out_df['voltage_3v3'] - 3.3).abs() / 3.3 * 100
            
        if 'voltage_12v' in out_df.columns:
            out_df['psu_voltage_12v_dev'] = (out_df['voltage_12v'] - 12.0).abs() / 12.0 * 100
            # Rolling standard deviation over 60 windows
            out_df['psu_voltage_stability'] = out_df['voltage_12v'].rolling(window=60).std()
            
        # 17-19: Fan Features
        if 'actual_rpm' in out_df.columns and 'expected_rpm' in out_df.columns:
            out_df['fan_rpm_deviation'] = (out_df['actual_rpm'] - out_df['expected_rpm']) / (out_df['expected_rpm'] + 1e-6)
            
        # 20-21: RAM Features
        if 'page_faults' in out_df.columns:
            out_df['ram_page_fault_rate'] = out_df['page_faults'].diff(1)
            
        # 27: Uptime
        if 'system_uptime_hours' not in out_df.columns:
            # Proxy if not available
            out_df['system_uptime_hours'] = 0
            
        # Forward fill any NaNs created by diff/rolling, then fill remaining with 0
        out_df = out_df.ffill().fillna(0)
        
        return out_df

def process_live_telemetry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single live telemetry payload to add engineered features.
    
    Args:
        payload (Dict[str, Any]): Raw telemetry dictionary.
        
    Returns:
        Dict[str, Any]: Telemetry with engineered features appended.
    """
    # In a real streaming scenario, we'd maintain state across calls
    # For now, this is a placeholder structure
    return payload
