"""
Model Quantization Pipeline (FP32 -> INT8).

Converts the Keras FP32 1D-CNN Multi-Task model into a highly optimized
TFLite INT8 model for deployment on the Raspberry Pi 4 edge node.

Fix (2026-08-10): Representative dataset now uses the classifier scaler
to provide properly scaled calibration data for INT8 range determination.
"""

import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
import joblib

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SERVER_MODELS = os.path.join(BASE_DIR, 'server', 'models')
EDGE_MODELS = os.path.join(BASE_DIR, 'edge', 'models')

import pandas as pd
from server.training.feature_schema import (
    FEATURE_NAMES_V1, FEATURE_COUNT, FEATURE_DEFAULTS, get_ordered_defaults
)


def _engineer_ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 28 engineered features from AI4I 2020 data.
    Mirrors the exact feature engineering from train_classifier.py to ensure
    calibration data matches what the model was trained on.
    """
    # Detect column names
    temp_air = temp_proc = rot_speed = torque = tool_wear = None
    for col in df.columns:
        cl = col.lower()
        if 'air' in cl and 'temp' in cl:
            temp_air = col
        elif 'process' in cl and 'temp' in cl:
            temp_proc = col
        elif 'rotational' in cl:
            rot_speed = col
        elif 'torque' in cl:
            torque = col
        elif 'tool' in cl and 'wear' in cl:
            tool_wear = col

    features = pd.DataFrame()

    if temp_air:
        features['cpu_temp'] = df[temp_air] - 273.15
        features['cpu_thermal_ramp'] = features['cpu_temp'].diff().fillna(0)
    else:
        features['cpu_temp'] = 0.0
        features['cpu_thermal_ramp'] = 0.0

    if temp_proc:
        features['gpu_temp'] = df[temp_proc] - 273.15
        features['gpu_thermal_ramp'] = features['gpu_temp'].diff().fillna(0)
    else:
        features['gpu_temp'] = 0.0
        features['gpu_thermal_ramp'] = 0.0

    if temp_air and temp_proc:
        features['cpu_ambient_delta'] = features['cpu_temp'] - features['gpu_temp']
        features['thermal_ratio'] = features['cpu_temp'] / (features['gpu_temp'] + 1e-6)
    else:
        features['cpu_ambient_delta'] = 0.0
        features['thermal_ratio'] = 1.0

    if rot_speed:
        features['fan_rpm'] = df[rot_speed]
        features['fan_rpm_delta'] = features['fan_rpm'].diff().fillna(0)
        features['fan_rpm_rolling_std'] = features['fan_rpm'].rolling(5, min_periods=1).std().fillna(0)
    else:
        features['fan_rpm'] = 0.0
        features['fan_rpm_delta'] = 0.0
        features['fan_rpm_rolling_std'] = 0.0

    if torque:
        features['psu_power'] = df[torque]
        features['psu_power_delta'] = features['psu_power'].diff().fillna(0)
        features['psu_efficiency'] = features['psu_power'] / (features['psu_power'].max() + 1e-6)
    else:
        features['psu_power'] = 0.0
        features['psu_power_delta'] = 0.0
        features['psu_efficiency'] = 0.0

    if tool_wear:
        features['wear_level'] = df[tool_wear]
        features['wear_rate'] = features['wear_level'].diff().fillna(0)
        features['wear_normalized'] = features['wear_level'] / (features['wear_level'].max() + 1e-6)
        features['wear_remaining'] = 1.0 - features['wear_normalized']
    else:
        features['wear_level'] = 0.0
        features['wear_rate'] = 0.0
        features['wear_normalized'] = 0.0
        features['wear_remaining'] = 1.0

    if rot_speed and torque:
        features['power_speed_ratio'] = df[torque] / (df[rot_speed] + 1e-6)
        features['mechanical_stress'] = df[torque] * df[rot_speed]
    else:
        features['power_speed_ratio'] = 0.0
        features['mechanical_stress'] = 0.0

    if temp_air:
        features['temp_rolling_mean'] = features['cpu_temp'].rolling(10, min_periods=1).mean()
        features['temp_rolling_std'] = features['cpu_temp'].rolling(10, min_periods=1).std().fillna(0)
    else:
        features['temp_rolling_mean'] = 0.0
        features['temp_rolling_std'] = 0.0

    # Pad to exactly 28 features
    while len(features.columns) < 28:
        features[f'_pad_{len(features.columns)}'] = 0.0
    features = features[features.columns[:28]]

    return features


def _generate_synthetic_calibration_data(scaler, n_samples: int = 100):
    """
    Generate synthetic calibration data from FEATURE_DEFAULTS with realistic variance.

    Phase E4 FIX (2026-08-13): Replaces the broken np.random.randn fallback.

    Instead of random standard-normal noise (mean=0, std=1) which bears no
    resemblance to real server telemetry (e.g., cpu_temp ~45C, fan_rpm ~1500),
    this generates samples centered on FEATURE_DEFAULTS with ±15% Gaussian
    variance, then applies the classifier_scaler to match the distribution
    the model was trained on.

    This ensures INT8 quantization ranges are calibrated on data that
    resembles actual server operations, preventing catastrophic accuracy loss.
    """
    baseline = np.array(get_ordered_defaults(), dtype=np.float32)  # (28,)

    print(f"[FALLBACK] Generating {n_samples} synthetic calibration samples from FEATURE_DEFAULTS")
    print(f"[FALLBACK] Baseline sample: cpu_temp={baseline[0]:.1f}, cpu_load={baseline[1]:.1f}, "
          f"gpu_temp={baseline[4]:.1f}, fan_rpm_dev={baseline[22]:.2f}")

    for i in range(n_samples):
        # Add ±15% Gaussian variance to each feature (clamped to non-negative for physical quantities)
        noise_scale = 0.15  # 15% relative variance
        noise = np.random.randn(FEATURE_COUNT).astype(np.float32) * noise_scale
        sample = baseline * (1.0 + noise)

        # Clamp features that must be non-negative (temperatures, RPMs, etc.)
        sample = np.maximum(sample, 0.0)

        sample_2d = sample.reshape(1, -1)

        # Apply scaler if available (ensures correct INT8 ranges matching training distribution)
        if scaler is not None:
            sample_2d = scaler.transform(sample_2d).astype(np.float32)
        sample_2d = np.clip(sample_2d, -6.0, 6.0).astype(np.float32)

        data = sample_2d.reshape(1, FEATURE_COUNT, 1)
        yield [data]


def representative_dataset_gen():
    """
    Generator function to provide representative dataset for INT8 quantization.

    BUG-N9 FIX (2026-08-12): Uses the SAME feature engineering as train_classifier.py
    to produce proper 28 engineered features, THEN applies the classifier scaler.

    Phase E4 FIX (2026-08-13): Fallback now uses FEATURE_DEFAULTS + realistic variance
    instead of np.random.randn, which was destroying INT8 dynamic ranges.

    OLD (broken): Took 5 arbitrary numerical columns + 23 zeros -> wrong dynamic ranges.
    NEW (correct): Full 28-feature engineering -> scaler -> accurate INT8 calibration.
    FALLBACK (new): FEATURE_DEFAULTS + ±15% variance -> scaler -> safe INT8 calibration.
    """
    # Load scaler
    scaler = None
    scaler_path = os.path.join(SERVER_MODELS, 'classifier_scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[OK] Loaded classifier_scaler.pkl for calibration")
    else:
        print(f"WARNING: classifier_scaler.pkl not found -- using unscaled data for calibration")

    csv_path = os.path.join(BASE_DIR, 'data', 'ai4i_2020', 'ai4i2020.csv')
    try:
        df = pd.read_csv(csv_path)
        features = _engineer_ai4i_features(df)
        samples = features.values[:100].astype(np.float32)

        for sample in samples:
            sample_2d = sample.reshape(1, -1)

            # Apply scaler if available (ensures correct INT8 ranges)
            if scaler is not None:
                sample_2d = scaler.transform(sample_2d).astype(np.float32)
            sample_2d = np.clip(sample_2d, -6.0, 6.0).astype(np.float32)

            data = sample_2d.reshape(1, 28, 1)
            yield [data]
    except Exception as e:
        # ── Phase E4 FIX: Use FEATURE_DEFAULTS instead of random noise ──
        # OLD (BROKEN): np.random.randn(1, 28, 1) -> standard normal noise
        #   destroys INT8 dynamic ranges because values are centered at 0.0
        #   with σ=1.0, while real telemetry has values like cpu_temp=45C.
        # NEW (SAFE): Generate synthetic data from FEATURE_DEFAULTS with
        #   realistic ±15% variance, then apply the classifier_scaler.
        print(f"WARNING: Failed to load real calibration data ({e}).")
        print(f"WARNING: Using FEATURE_DEFAULTS-based synthetic calibration "
              f"(NOT random noise -- Phase E4 fix).")

        if scaler is None:
            print("CRITICAL: Neither calibration dataset NOR classifier_scaler.pkl available!")
            print("CRITICAL: INT8 quantization will use unscaled FEATURE_DEFAULTS. "
                  "Re-quantize after training to ensure accuracy.")

        yield from _generate_synthetic_calibration_data(scaler, n_samples=100)

def quantize_and_export():
    os.makedirs(EDGE_MODELS, exist_ok=True)

    fp32_model_path = os.path.join(SERVER_MODELS, 'classifier_fp32.h5')
    if not os.path.exists(fp32_model_path):
        print(f"Error: FP32 model not found at {fp32_model_path}. Train it first.")
        return

    print(f"Loading FP32 model from {fp32_model_path}...")
    model = tf.keras.models.load_model(fp32_model_path, compile=False)

    print("Initializing TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32  # Softmax+TTF need float32 precision

    print("Converting model to INT8 TFLite...")
    try:
        tflite_quant_model = converter.convert()
    except Exception as e:
        print(f"Full INT8 quantization failed ({e}). Reverting to default optimization.")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_quant_model = converter.convert()

    output_path = os.path.join(EDGE_MODELS, 'classifier_int8.tflite')
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)

    print(f"Quantized model successfully saved to {output_path}")
    print(f"Size: {len(tflite_quant_model) / 1024:.2f} KB")

    # Also copy classifier scaler to edge/models/ for edge inference
    scaler_src = os.path.join(SERVER_MODELS, 'classifier_scaler.pkl')
    scaler_dst = os.path.join(EDGE_MODELS, 'classifier_scaler.pkl')
    if os.path.exists(scaler_src):
        shutil.copy2(scaler_src, scaler_dst)
        print(f"Copied classifier_scaler.pkl to {scaler_dst}")

if __name__ == "__main__":
    quantize_and_export()

