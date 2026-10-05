"""
G3.3: TCN (Temporal Convolutional Network) for Storage RUL Prediction
=====================================================================
Trains a TCN on Backblaze Drive Stats to predict Remaining Useful Life (RUL)
in days for storage devices. Uses sliding windows of SMART data.

Architecture: 4 dilated causal convolution layers (1->2->4->8), 64 filters
Input:  (batch, window_days, n_features)  = (batch, 30, 8)
Output: scalar RUL in days (0–90 range, clamped)

Training Data: Backblaze Q1 2023 -- 90 daily CSVs, ~235K drives/day
Strategy: Track drives that FAIL, build time-series windows leading to failure,
          label each window with RUL = days_until_failure.
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib

# ─── Configuration ──────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "backblaze", "data_Q1_2023")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EDGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "edge", "models")

WINDOW_SIZE = 30       # Days of history to look at
MAX_RUL = 90           # Cap RUL at 90 days
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

# Key SMART features for failure prediction (from Backblaze research papers)
SMART_FEATURES = [
    'smart_5_raw',    # Reallocated Sectors Count (critical failure indicator)
    'smart_9_raw',    # Power-On Hours
    'smart_187_raw',  # Reported Uncorrectable Errors
    'smart_188_raw',  # Command Timeout
    'smart_194_raw',  # Temperature Celsius
    'smart_197_raw',  # Current Pending Sector Count
    'smart_198_raw',  # Offline Uncorrectable
    'smart_241_raw',  # Total LBAs Written
]

N_FEATURES = len(SMART_FEATURES)


# ─── Data Pipeline (Optimized) ──────────────────────────────

def load_targeted_data(data_dir, features, max_healthy_sample=2000):
    """
    Optimized 2-pass loading:
      Pass 1: Scan all CSVs for failed serial numbers (only load 2 columns)
      Pass 2: Reload CSVs filtering to only failed + sampled healthy drives
    
    This reduces 21M rows -> ~100K rows, making window building fast.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "2023-*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"[Pass 1] Scanning {len(csv_files)} files for failures...")
    failed_serials = set()
    all_serials = set()
    
    for i, f in enumerate(csv_files):
        try:
            df_scan = pd.read_csv(f, usecols=['serial_number', 'failure'])
            failed = df_scan[df_scan['failure'] == 1]['serial_number'].unique()
            failed_serials.update(failed)
            # Sample some serials for healthy pool (don't load all 240K)
            if len(all_serials) < max_healthy_sample * 2:
                sample_size = min(500, len(df_scan))
                sampled = df_scan['serial_number'].sample(n=sample_size, random_state=i).unique()
                all_serials.update(sampled)
        except Exception as e:
            print(f"  Warning: Skipped {f}: {e}")
    
    print(f"  Found {len(failed_serials)} failed drives")
    
    # Build target serial set: all failed + sample of healthy
    healthy_candidates = all_serials - failed_serials
    n_healthy = min(len(healthy_candidates), max_healthy_sample)
    healthy_sample = set(np.random.choice(list(healthy_candidates), size=n_healthy, replace=False))
    
    target_serials = failed_serials | healthy_sample
    print(f"  Target serials: {len(failed_serials)} failed + {len(healthy_sample)} healthy = {len(target_serials)} total")
    
    print(f"[Pass 2] Loading data for {len(target_serials)} drives...")
    cols_to_load = ['date', 'serial_number', 'failure'] + features
    
    frames = []
    for i, f in enumerate(csv_files):
        try:
            df = pd.read_csv(f, usecols=lambda c: c in cols_to_load)
            # Filter to only target drives
            df = df[df['serial_number'].isin(target_serials)]
            frames.append(df)
            if (i + 1) % 30 == 0:
                print(f"  Loaded {i+1}/{len(csv_files)} files...")
        except Exception as e:
            print(f"  Warning: Skipped {f}: {e}")
    
    all_data = pd.concat(frames, ignore_index=True)
    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data = all_data.sort_values(['serial_number', 'date'])
    
    print(f"  Total rows: {len(all_data):,} (from {len(target_serials)} drives)")
    
    return all_data, np.array(list(failed_serials)), np.array(list(healthy_sample))


def build_rul_windows(df, failed_serials, healthy_serials, features, 
                      window_size=30, max_rul=90):
    """
    Build sliding windows using groupby for efficient iteration.
    """
    X_windows = []
    y_rul = []
    failed_set = set(failed_serials)
    
    print(f"Building windows (groupby)...")
    
    # Group by serial for O(n) iteration instead of O(n*k) filtering
    grouped = df.groupby('serial_number')
    
    processed = 0
    for serial, group in grouped:
        group = group.sort_values('date')
        
        if len(group) < window_size:
            continue
        
        feature_values = group[features].ffill().fillna(0).values
        dates = group['date'].values
        
        if serial in failed_set:
            # Failed drive: create RUL-labeled windows
            failure_rows = group[group['failure'] == 1]
            if len(failure_rows) == 0:
                continue
            failure_date = failure_rows['date'].iloc[0]
            
            for i in range(len(group) - window_size + 1):
                window = feature_values[i:i + window_size]
                window_end_date = pd.Timestamp(dates[i + window_size - 1])
                rul = (failure_date - window_end_date).days
                
                if rul < 0:
                    continue
                
                rul = min(rul, max_rul)
                X_windows.append(window)
                y_rul.append(rul)
        else:
            # Healthy drive: take one random window with RUL = max_rul
            start_idx = np.random.randint(0, max(1, len(group) - window_size))
            end_idx = min(start_idx + window_size, len(feature_values))
            if end_idx - start_idx < window_size:
                continue
            
            window = feature_values[start_idx:end_idx]
            X_windows.append(window)
            y_rul.append(max_rul)
        
        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed} drives, {len(X_windows)} windows so far...")
    
    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_rul, dtype=np.float32)
    
    n_failed_windows = sum(1 for r in y_rul if r < max_rul)
    n_healthy_windows = len(y_rul) - n_failed_windows
    print(f"  Total windows: {len(X)} ({n_failed_windows} failed-drive, {n_healthy_windows} healthy)")
    print(f"  RUL distribution: min={y.min():.0f}, max={y.max():.0f}, mean={y.mean():.1f}")
    
    return X, y


# ─── TCN Model ──────────────────────────────────────────────

class CausalConv1D(layers.Layer):
    """Causal convolution with dilation for TCN."""
    
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding_size = (kernel_size - 1) * dilation_rate
        
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation=None
        )
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        self.dropout = layers.Dropout(0.2)
    
    def call(self, x, training=False):
        out = self.conv(x)
        out = self.bn(out, training=training)
        out = self.activation(out)
        out = self.dropout(out, training=training)
        return out


def build_tcn_model(input_shape, dilations=[1, 2, 4, 8], n_filters=64, kernel_size=3):
    """
    Build a Temporal Convolutional Network for RUL prediction.
    
    Architecture:
    - 4 dilated causal convolution blocks (dilation 1->2->4->8)
    - Each block: CausalConv1D -> BatchNorm -> ReLU -> Dropout(0.2)
    - Residual connections between blocks
    - Global Average Pooling -> Dense(32) -> Dense(1)
    """
    inputs = keras.Input(shape=input_shape, name='smart_features')
    
    x = inputs
    
    # Project input to n_filters channels
    x = layers.Conv1D(n_filters, 1, padding='same', name='input_projection')(x)
    
    for i, d in enumerate(dilations):
        # Causal convolution block
        conv_out = layers.Conv1D(
            n_filters, kernel_size, 
            dilation_rate=d, padding='causal',
            name=f'tcn_conv_{i}'
        )(x)
        conv_out = layers.BatchNormalization(name=f'tcn_bn_{i}')(conv_out)
        conv_out = layers.Activation('relu', name=f'tcn_relu_{i}')(conv_out)
        conv_out = layers.Dropout(0.2, name=f'tcn_drop_{i}')(conv_out)
        
        # Second conv in block
        conv_out = layers.Conv1D(
            n_filters, kernel_size,
            dilation_rate=d, padding='causal',
            name=f'tcn_conv2_{i}'
        )(conv_out)
        conv_out = layers.BatchNormalization(name=f'tcn_bn2_{i}')(conv_out)
        conv_out = layers.Activation('relu', name=f'tcn_relu2_{i}')(conv_out)
        conv_out = layers.Dropout(0.2, name=f'tcn_drop2_{i}')(conv_out)
        
        # Residual connection
        x = layers.Add(name=f'tcn_residual_{i}')([x, conv_out])
    
    # Global pooling -> regression head
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(32, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.3, name='dense_drop')(x)
    outputs = layers.Dense(1, activation='relu', name='rul_output')(x)  # ReLU ensures non-negative RUL
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='TCN_Storage_RUL')
    return model


def asymmetric_rul_loss(y_true, y_pred):
    """
    Asymmetric loss: penalize late predictions (under-estimating RUL) 
    more than early predictions (over-estimating RUL).
    
    Late prediction means the system thinks there's more life left than there actually is.
    This is dangerous -- could lead to unexpected failures.
    """
    error = y_true - y_pred  # positive = over-estimated RUL (early), negative = under-estimated (late)
    
    # Asymmetric weights
    late_weight = 2.0   # Under-estimating RUL is 2x more costly
    early_weight = 1.0  # Over-estimating is less dangerous
    
    weights = tf.where(error < 0, late_weight, early_weight)
    return tf.reduce_mean(weights * tf.square(error))


# ─── Training ───────────────────────────────────────────────

def train_tcn():
    """Full TCN training pipeline."""
    
    print("=" * 60)
    print("  G3.3: TCN Storage RUL -- Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load data (optimized 2-pass -- only loads relevant drives)
    data_dir = os.path.abspath(DATA_DIR)
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    df, failed_serials, healthy_serials = load_targeted_data(data_dir, SMART_FEATURES)
    
    if len(failed_serials) == 0:
        print("WARNING: No failed drives found! Using degradation-based labeling instead.")
        failed_serials = find_degrading_drives(df)
    
    # Step 2: Drive-Level Stratified Train/Val Split BEFORE windowing (eliminates cross-window drive leakage)
    print("\nSplitting drive serial numbers into Train/Val sets (GroupKFold drive isolation)...")
    if len(failed_serials) >= 2 and len(healthy_serials) >= 2:
        failed_train, failed_val = train_test_split(failed_serials, test_size=VALIDATION_SPLIT, random_state=42)
        healthy_train, healthy_val = train_test_split(healthy_serials, test_size=VALIDATION_SPLIT, random_state=42)
        
        df_train = df[df['serial_number'].isin(set(failed_train) | set(healthy_train))]
        df_val = df[df['serial_number'].isin(set(failed_val) | set(healthy_val))]
        
        X_train_raw, y_train = build_rul_windows(df_train, failed_train, healthy_train, SMART_FEATURES, WINDOW_SIZE, MAX_RUL)
        X_val_raw, y_val = build_rul_windows(df_val, failed_val, healthy_val, SMART_FEATURES, WINDOW_SIZE, MAX_RUL)
    else:
        # Fallback for very small sample sets
        X, y = build_rul_windows(df, failed_serials, healthy_serials, SMART_FEATURES, WINDOW_SIZE, MAX_RUL)
        if len(X) < 100:
            print(f"WARNING: Only {len(X)} windows. Adding degradation-based samples...")
            X_deg, y_deg = build_degradation_windows(df, SMART_FEATURES, WINDOW_SIZE, MAX_RUL)
            X = np.concatenate([X, X_deg])
            y = np.concatenate([y, y_deg])
        n_val = max(1, int(len(X) * VALIDATION_SPLIT))
        X_train_raw, y_train = X[n_val:], y[n_val:]
        X_val_raw, y_val = X[:n_val], y[:n_val]
    
    # Step 5: Fit scaler on TRAINING data ONLY (BUG-N3 fix: no data leakage)
    print("Normalizing features (fit on training set only)...")
    n_train, n_steps, n_feat = X_train_raw.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, n_feat)).reshape(n_train, n_steps, n_feat)
    X_val = scaler.transform(X_val_raw.reshape(-1, n_feat)).reshape(len(X_val_raw), n_steps, n_feat)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    
    # Step 6: Build model
    print("\nBuilding TCN model...")
    model = build_tcn_model(
        input_shape=(WINDOW_SIZE, N_FEATURES),
        dilations=[1, 2, 4, 8],
        n_filters=64,
        kernel_size=3
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=asymmetric_rul_loss,
        metrics=['mae']
    )
    
    model.summary()
    
    # Step 7: Train
    print("\nTraining TCN...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 8: Evaluate
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation MAE:  {val_mae:.2f} days")
    
    # Step 9: Save FP32 model
    os.makedirs(MODEL_DIR, exist_ok=True)
    fp32_path = os.path.join(MODEL_DIR, "tcn_storage_rul_fp32.keras")
    model.save(fp32_path)
    print(f"Saved FP32 model: {fp32_path}")
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "tcn_storage_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler: {scaler_path}")
    
    # Step 10: Quantize to INT8 TFLite
    print("\nQuantizing to INT8 TFLite...")
    
    def representative_dataset():
        for i in range(min(200, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32  # Keep output as float for RUL
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(MODEL_DIR, "tcn_storage_rul_int8.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_kb = os.path.getsize(tflite_path) / 1024
    print(f"Saved INT8 TFLite: {tflite_path} ({tflite_size_kb:.1f} KB)")
    
    # Copy to edge models directory
    os.makedirs(EDGE_MODEL_DIR, exist_ok=True)
    edge_tflite_path = os.path.join(EDGE_MODEL_DIR, "tcn_storage_rul_int8.tflite")
    import shutil
    shutil.copy2(tflite_path, edge_tflite_path)
    
    edge_scaler_path = os.path.join(EDGE_MODEL_DIR, "tcn_storage_scaler.joblib")
    shutil.copy2(scaler_path, edge_scaler_path)
    print(f"Copied to edge: {edge_tflite_path}")
    
    # Save training metadata
    metadata = {
        "model_name": "TCN Storage RUL",
        "phase": "G3.3 + G3.6",
        "architecture": "TCN (4 dilated causal conv layers, 64 filters, kernel=3)",
        "input_shape": [WINDOW_SIZE, N_FEATURES],
        "features": SMART_FEATURES,
        "output": "RUL in days (0-90)",
        "training_data": "Backblaze Q1 2023",
        "val_mae_days": float(val_mae),
        "val_loss": float(val_loss),
        "fp32_path": fp32_path,
        "tflite_path": tflite_path,
        "tflite_size_kb": tflite_size_kb,
        "scaler_path": scaler_path,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epochs_run": len(history.history['loss']),
        "best_epoch": int(np.argmin(history.history['val_loss'])) + 1,
    }
    
    meta_path = os.path.join(MODEL_DIR, "tcn_storage_rul_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")
    
    print("\n" + "=" * 60)
    print(f"  TCN Storage RUL -- TRAINING COMPLETE")
    print(f"  Val MAE: {val_mae:.2f} days | INT8 Size: {tflite_size_kb:.1f} KB")
    print("=" * 60)
    
    return model, history


def find_degrading_drives(df):
    """Find drives showing degradation (rising reallocated sectors) as proxy for failure."""
    print("Searching for degrading drives (rising smart_5_raw)...")
    
    degrading = []
    for serial, group in df.groupby('serial_number'):
        if len(group) < 30:  # Need at least 30 days
            continue
        
        s5 = group['smart_5_raw'].dropna()
        if len(s5) < 2:
            continue
        
        # Check if reallocated sectors increased significantly
        if s5.iloc[-1] > s5.iloc[0] + 5:
            degrading.append(serial)
        
        if len(degrading) >= 500:  # Cap at 500
            break
    
    print(f"Found {len(degrading)} degrading drives")
    return np.array(degrading)


def build_degradation_windows(df, features, window_size, max_rul, n_samples=2000):
    """Build windows using degradation score instead of actual failure."""
    print(f"Building {n_samples} degradation-based windows...")
    
    X_windows = []
    y_rul = []
    
    # Sample random drives
    serials = df['serial_number'].unique()
    sampled = np.random.choice(serials, size=min(n_samples, len(serials)), replace=False)
    
    for serial in sampled:
        drive_data = df[df['serial_number'] == serial].sort_values('date')
        
        if len(drive_data) < window_size:
            continue
        
        feature_values = drive_data[features].values
        feature_df = pd.DataFrame(feature_values, columns=features)
        feature_df = feature_df.ffill().fillna(0)
        feature_values = feature_df.values
        
        # Use smart_5_raw (reallocated sectors) as health proxy
        s5_idx = features.index('smart_5_raw')
        s5_values = feature_values[:, s5_idx]
        
        # Calculate health score: more reallocated sectors = lower RUL
        max_s5 = max(s5_values.max(), 1)
        
        # Take one window
        start_idx = np.random.randint(0, max(1, len(drive_data) - window_size))
        end_idx = start_idx + window_size
        if end_idx > len(feature_values):
            continue
        
        window = feature_values[start_idx:end_idx]
        
        # RUL based on degradation: high sectors = low RUL
        current_s5 = s5_values[end_idx - 1]
        if current_s5 > 100:
            rul = max(0, max_rul * (1 - current_s5 / 1000))
        else:
            rul = max_rul  # Healthy
        
        X_windows.append(window)
        y_rul.append(rul)
    
    return np.array(X_windows, dtype=np.float32), np.array(y_rul, dtype=np.float32)


if __name__ == "__main__":
    train_tcn()
