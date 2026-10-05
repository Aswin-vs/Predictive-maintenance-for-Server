"""
Train 1D-CNN Multi-Task Model (Fault Classification + TTF Regression).

This model takes in the 28 engineered features and outputs:
1. Fault class (Softmax over 7 classes)
2. Normalized Time-To-Failure (Linear regression with Asymmetric Loss)

Supports:
- Real training on AI4I 2020 dataset (--data data/ai4i_2020/ai4i2020.csv)
- Real training on Backblaze data (--data data/processed/backblaze_ttf.csv)
- Dummy training for code testing (--dummy)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure project root and training dir are on sys.path for imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
_training_dir = os.path.join(BASE_DIR, 'server', 'training')
if _training_dir not in sys.path:
    sys.path.insert(0, _training_dir)

from server.training.feature_schema import FEATURE_NAMES_V1, FEATURE_COUNT, FAULT_LABELS, FEATURE_DEFAULTS
# F-024 fix: Use loss.py (alpha=5.0) instead of asymmetric_loss.py (alpha=2.0)
try:
    from server.training.loss import get_asymmetric_ttf_loss_tf as get_asymmetric_ttf_loss
except ImportError:
    from loss import get_asymmetric_ttf_loss_tf as get_asymmetric_ttf_loss
MODEL_DIR = os.path.join(BASE_DIR, 'server', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Number of features
INPUT_SHAPE = (28, 1)  # 28 features treated as 1D sequence
NUM_CLASSES = 7

# Fault class mapping — imported from feature_schema.FAULT_LABELS
# 0=HEALTHY, 1=HDD_DEGRADATION, 2=THERMAL_CRITICAL, 3=PSU_ANOMALY,
# 4=FAN_FAILURE, 5=MEMORY_FAULT, 6=GPU_DEGRADATION


def build_multitask_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Shared Feature Extractor
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Head 1: Fault Classification
    class_out = layers.Dense(NUM_CLASSES, activation='softmax', name='class_output')(x)
    
    # Head 2: TTF Regression
    ttf_out = layers.Dense(1, activation='linear', name='ttf_output')(x)
    
    model = Model(inputs=inputs, outputs=[class_out, ttf_out])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'ttf_output': get_asymmetric_ttf_loss()  # alpha=5.0 from loss.py
        },
        loss_weights={
            'class_output': 1.0,
            'ttf_output': 0.5
        },
        metrics={
            'class_output': 'accuracy',
            'ttf_output': 'mae'
        }
    )
    
    return model


def generate_dummy_data(samples=1000):
    """Generate dummy data for code testing on the dev machine."""
    X = np.random.randn(samples, 28, 1)
    y_class = np.random.randint(0, NUM_CLASSES, samples)
    y_ttf = np.random.uniform(0.0, 1.0, samples)
    return X, y_class, y_ttf


def load_ai4i_2020(csv_path: str):
    """
    Load and transform the AI4I 2020 Predictive Maintenance Dataset.
    
    AI4I 2020 has these columns:
    - UDI, Product ID, Type (L/M/H quality)
    - Air temperature [K], Process temperature [K]
    - Rotational speed [rpm], Torque [Nm], Tool wear [min]
    - Machine failure (binary), TWF, HDF, PWF, OSF, RNF (failure mode flags)
    
    We map AI4I failure modes to our 7-class system:
    - No failure -> 0 (HEALTHY)
    - HDF (Heat Dissipation) -> 2 (THERMAL_CRITICAL)
    - PWF (Power Failure) -> 3 (PSU_ANOMALY)  
    - OSF (Overstrain) -> 4 (FAN_FAILURE — mechanical stress analog)
    - TWF (Tool Wear) -> 1 (HDD_DEGRADATION — wear-based degradation analog)
    - RNF (Random) -> 6 (GPU_DEGRADATION — random electronic failure analog)
    """
    print(f"Loading AI4I 2020 dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"  Raw samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Extract raw features (5 continuous features from AI4I)
    feature_cols = []
    
    # Map AI4I column names (handle both naming conventions)
    temp_air = None
    for col in df.columns:
        if 'air' in col.lower() and 'temp' in col.lower():
            temp_air = col
        elif col == 'Air temperature [K]':
            temp_air = col
    
    temp_proc = None
    for col in df.columns:
        if 'process' in col.lower() and 'temp' in col.lower():
            temp_proc = col
        elif col == 'Process temperature [K]':
            temp_proc = col
    
    rot_speed = None
    for col in df.columns:
        if 'rotational' in col.lower() or col == 'Rotational speed [rpm]':
            rot_speed = col
    
    torque = None
    for col in df.columns:
        if 'torque' in col.lower() or col == 'Torque [Nm]':
            torque = col
    
    tool_wear = None
    for col in df.columns:
        if 'tool' in col.lower() and 'wear' in col.lower():
            tool_wear = col
        elif col == 'Tool wear [min]':
            tool_wear = col
    
    raw_features = [temp_air, temp_proc, rot_speed, torque, tool_wear]
    raw_features = [f for f in raw_features if f is not None]
    
    if len(raw_features) < 5:
        print(f"  WARNING: Only found {len(raw_features)} features: {raw_features}")
        print(f"  Available columns: {list(df.columns)}")
    
    # Create 28 engineered features from the 5 raw AI4I features
    features = pd.DataFrame(index=df.index)
    for feat in FEATURE_NAMES_V1:
        features[feat] = FEATURE_DEFAULTS.get(feat, 0.0)
        
    if temp_air:
        features['ambient_temp'] = df[temp_air] - 273.15  # Kelvin to Celsius
    if temp_proc:
        features['cpu_temp'] = df[temp_proc] - 273.15
    if temp_air and temp_proc:
        features['cpu_ambient_delta'] = features['cpu_temp'] - features['ambient_temp']
    if torque:
        features['cpu_load'] = (df[torque] / (df[torque].max() + 1e-6)) * 100.0
    if rot_speed:
        features['fan_rpm_deviation'] = (df[rot_speed] - 1500.0) / 1500.0
    if tool_wear:
        features['psu_voltage_12v_dev'] = (df[tool_wear] / 250.0) * 5.0
    
    # Map failure modes to our 7-class system
    # Detect failure column names (handle various CSV formats)
    failure_col = None
    for col in df.columns:
        if col.lower() in ['machine failure', 'machine_failure', 'target']:
            failure_col = col
    
    twf = hdf = pwf = osf = rnf = None
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ['twf', 'tool wear failure']:
            twf = col
        elif cl in ['hdf', 'heat dissipation failure']:
            hdf = col
        elif cl in ['pwf', 'power failure']:
            pwf = col
        elif cl in ['osf', 'overstrain failure']:
            osf = col
        elif cl in ['rnf', 'random failures']:
            rnf = col
    
    y_class = np.zeros(len(df), dtype=np.int32)
    
    if twf and hdf and pwf and osf and rnf:
        # AI4I has explicit failure mode columns
        y_class[df[twf] == 1] = 1   # TWF -> HDD_DEGRADATION (wear)
        y_class[df[hdf] == 1] = 2   # HDF -> THERMAL_CRITICAL
        y_class[df[pwf] == 1] = 3   # PWF -> PSU_ANOMALY
        y_class[df[osf] == 1] = 4   # OSF -> FAN_FAILURE (mechanical)
        y_class[df[rnf] == 1] = 6   # RNF -> GPU_DEGRADATION (random)
        # Class 5 (MEMORY_FAULT) has no direct AI4I analog — will be synthetic
    elif failure_col:
        # Binary failure only — assign random class for failures
        failed_mask = df[failure_col] == 1
        y_class[failed_mask] = np.random.choice([1, 2, 3, 4, 6], size=failed_mask.sum())
    
    # Generate synthetic TTF (normalized 0-1)
    # For healthy samples: TTF is high (0.7-1.0)
    # For failing samples: TTF decreases with tool wear
    y_ttf = np.ones(len(df), dtype=np.float32) * 0.85
    if tool_wear:
        wear_normalized = df[tool_wear] / (df[tool_wear].max() + 1e-6)
        y_ttf = 1.0 - wear_normalized.values.astype(np.float32)
    failing = y_class > 0
    y_ttf[failing] = np.clip(y_ttf[failing] * 0.3, 0.01, 0.5)
    
    # Add synthetic Class 5 (MEMORY_FAULT) rows
    num_synthetic = 100
    synth_features = pd.DataFrame(index=range(num_synthetic))
    for feat in FEATURE_NAMES_V1:
        synth_features[feat] = FEATURE_DEFAULTS.get(feat, 0.0)
    synth_features['ram_used_gb'] = np.random.uniform(15.0, 16.0, num_synthetic)
    synth_features['ram_page_faults'] = np.random.uniform(5000, 10000, num_synthetic)
    
    synth_y_class = np.full(num_synthetic, 5, dtype=np.int32)
    synth_y_ttf = np.random.uniform(0.05, 0.2, num_synthetic).astype(np.float32)
    
    X_out = np.vstack([features.values.astype(np.float32), synth_features.values.astype(np.float32)])
    y_class = np.concatenate([y_class, synth_y_class])
    y_ttf = np.concatenate([y_ttf, synth_y_ttf])
    
    # NOTE: Do NOT standardize here — standardization happens AFTER train/val split
    # in train_model() to prevent data leakage (BUG-N3 fix).
    X_out = X_out.astype(np.float32)
    
    print(f"  Final dataset: {X_out.shape[0]} samples, {28} features")
    print(f"  Class distribution: {dict(zip(*np.unique(y_class, return_counts=True)))}")
    
    return X_out, y_class, y_ttf


def load_backblaze_data(csv_path: str):
    """
    Load Backblaze drive stats for HDD-focused classification.
    Maps SMART attributes to the 28-feature format and uses TTF labels.
    """
    print(f"Loading Backblaze data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=50000)  # Limit for memory
    
    smart_cols = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                  'smart_194_raw', 'smart_197_raw', 'smart_198_raw']
    
    available = [c for c in smart_cols if c in df.columns]
    if len(available) < 3:
        print(f"  ERROR: Not enough SMART columns found. Available: {list(df.columns)[:20]}")
        return None, None, None
    
    features = pd.DataFrame()
    for col in smart_cols:
        if col in df.columns:
            features[col] = df[col].fillna(0)
        else:
            features[col] = 0
    
    # Derive additional features from SMART data
    features['smart_5_delta'] = features['smart_5_raw'].diff().fillna(0)
    features['smart_9_delta'] = features['smart_9_raw'].diff().fillna(0)
    features['smart_197_delta'] = features['smart_197_raw'].diff().fillna(0)
    features['temp_rolling'] = features['smart_194_raw'].rolling(10, min_periods=1).mean()
    features['temp_std'] = features['smart_194_raw'].rolling(10, min_periods=1).std().fillna(0)
    features['error_ratio'] = features['smart_187_raw'] / (features['smart_9_raw'] + 1e-6)
    features['wear_index'] = features['smart_5_raw'] + features['smart_197_raw'] + features['smart_198_raw']
    
    # Pad to 28 features
    while len(features.columns) < 28:
        features[f'_pad_{len(features.columns)}'] = 0.0
    features = features[features.columns[:28]]
    
    # Labels: binary (failed or not) + TTF
    if 'failure' in df.columns:
        y_class = np.where(df['failure'] == 1, 1, 0).astype(np.int32)  # 1=HDD_DEGRADATION
    else:
        y_class = np.zeros(len(df), dtype=np.int32)
    
    if 'ttf_days' in df.columns:
        y_ttf = df['ttf_days'].fillna(9999).clip(0, 365).values / 365.0  # Normalize to 0-1
    else:
        y_ttf = np.ones(len(df), dtype=np.float32) * 0.9
    
    # NOTE: Do NOT standardize here — standardization happens AFTER train/val split
    # in train_model() to prevent data leakage (BUG-N3 fix).
    X = features.values.astype(np.float32)
    
    print(f"  Final: {X.shape[0]} samples, class dist: {dict(zip(*np.unique(y_class, return_counts=True)))}")
    
    return X, y_class, y_ttf.astype(np.float32)


def train_model(epochs=10, batch_size=32, dummy=False, data_path=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Building 1D-CNN Multi-Task Model...")
    model = build_multitask_cnn(INPUT_SHAPE)
    model.summary()
    
    if dummy:
        print("Training on dummy data...")
        X_train, y_class_train, y_ttf_train = generate_dummy_data(5000)
        X_val, y_class_val, y_ttf_val = generate_dummy_data(1000)
    elif data_path:
        # Auto-detect dataset type from file content
        if 'ai4i' in data_path.lower() or '2020' in data_path.lower():
            X, y_class, y_ttf = load_ai4i_2020(data_path)
        elif 'backblaze' in data_path.lower() or 'ttf' in data_path.lower():
            X, y_class, y_ttf = load_backblaze_data(data_path)
        else:
            # Try to auto-detect by reading columns
            sample = pd.read_csv(data_path, nrows=5)
            if any('smart' in c.lower() for c in sample.columns):
                X, y_class, y_ttf = load_backblaze_data(data_path)
            else:
                X, y_class, y_ttf = load_ai4i_2020(data_path)
        
        if X is None:
            print("ERROR: Data loading failed. Use --dummy to train on synthetic data.")
            return
        
        # Train/val split (80/20) — SPLIT BEFORE SCALING to prevent data leakage (BUG-N3)
        n = len(X)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]
        
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_class_train, y_class_val = y_class[train_idx], y_class[val_idx]
        y_ttf_train, y_ttf_val = y_ttf[train_idx], y_ttf[val_idx]
        
        # Oversample minority classes in training set ONLY (Bug 1 fix)
        healthy_mask = y_class_train == 0
        failure_mask = ~healthy_mask
        n_healthy = healthy_mask.sum()
        n_failure = failure_mask.sum()
        
        if n_failure > 0 and n_failure < n_healthy:
            oversample_factor = min(n_healthy // n_failure, 10)  # Cap at 10x
            print(f"  Training Class balance: {n_healthy} healthy, {n_failure} failure (oversampling {oversample_factor}x)")
            
            fail_indices = np.where(failure_mask)[0]
            if len(fail_indices) > 0:
                oversample_indices = np.tile(fail_indices, oversample_factor)
                all_indices = np.concatenate([np.arange(len(X_train_raw)), oversample_indices])
                np.random.shuffle(all_indices)
                
                X_train_raw = X_train_raw[all_indices]
                y_class_train = y_class_train[all_indices]
                y_ttf_train = y_ttf_train[all_indices]
        
        # Fit scaler on TRAINING data ONLY (BUG-N3 fix: no data leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)  # Transform val using train stats
        
        # Save scaler for edge inference (BUG-N4 fix)
        scaler_path = os.path.join(MODEL_DIR, 'classifier_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved to {scaler_path} (mean shape: {scaler.mean_.shape})")
        
        # Also save to edge models for direct deployment
        edge_scaler_path = os.path.join(BASE_DIR, 'edge', 'models', 'classifier_scaler.pkl')
        os.makedirs(os.path.dirname(edge_scaler_path), exist_ok=True)
        joblib.dump(scaler, edge_scaler_path)
        print(f"  Scaler copied to {edge_scaler_path}")
        
        # Reshape for 1D-CNN: (samples, 28, 1)
        X_train = X_train_scaled.reshape(-1, 28, 1).astype(np.float32)
        X_val = X_val_scaled.reshape(-1, 28, 1).astype(np.float32)
        
        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    else:
        print("ERROR: Provide --data <path> for real training or --dummy for synthetic data.")
        return
        
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'classifier_fp32.h5'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    print("Starting training...")
    history = model.fit(
        X_train,
        {'class_output': y_class_train, 'ttf_output': y_ttf_train},
        validation_data=(X_val, {'class_output': y_class_val, 'ttf_output': y_ttf_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Print final metrics
    val_acc = max(history.history.get('val_class_output_accuracy', [0]))
    val_mae = min(history.history.get('val_ttf_output_mae', [999]))
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE!")
    print(f"  Best Val Classification Accuracy: {val_acc:.4f}")
    print(f"  Best Val TTF MAE: {val_mae:.4f}")
    print(f"  Model saved to: server/models/classifier_fp32.h5")
    print(f"{'='*50}")
    print(f"\nNext steps:")
    print(f"  1. Quantize: python server/optimization/quantize_model.py")
    print(f"  2. Deploy:   copy edge/models/classifier_int8.tflite to RPi4")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D-CNN Multi-Task Classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dummy", action="store_true", help="Train on dummy data for testing")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data CSV (AI4I 2020 or Backblaze)")
    args = parser.parse_args()
    
    if not args.dummy and not args.data:
        # Try to find data automatically
        ai4i_path = os.path.join(DATA_DIR, 'ai4i_2020', 'ai4i2020.csv')
        bb_path = os.path.join(DATA_DIR, 'processed', 'backblaze_ttf.csv')
        
        if os.path.exists(ai4i_path):
            print(f"Auto-detected AI4I 2020 dataset at {ai4i_path}")
            args.data = ai4i_path
        elif os.path.exists(bb_path):
            print(f"Auto-detected Backblaze dataset at {bb_path}")
            args.data = bb_path
        else:
            print("No dataset found! Falling back to dummy training.")
            print(f"  Looked for: {ai4i_path}")
            print(f"  Looked for: {bb_path}")
            args.dummy = True
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, 
                dummy=args.dummy, data_path=args.data)
