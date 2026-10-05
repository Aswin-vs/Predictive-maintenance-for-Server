"""
Train LSTM-Autoencoder (Anomaly Detection).

This model learns the temporal representation of "healthy" server data.
During inference, if the reconstruction error exceeds a threshold, an
anomaly is flagged. Ideal for GPU/CPU degradation detection.

Supports:
- Real training on AI4I 2020 healthy data (--data data/ai4i_2020/ai4i2020.csv)
- Real training on Backblaze healthy data (--data data/processed/backblaze_ttf.csv)
- Dummy training for code testing (--dummy)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure project root is on sys.path for imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, 'server', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Import feature schema for real telemetry detection
from server.training.feature_schema import FEATURE_NAMES_V1

# The sequence length and feature count for temporal analysis
SEQ_LEN = 10   # 10 time steps per window
NUM_FEATURES = 28  # Full feature set


def build_lstm_autoencoder(seq_len, num_features):
    inputs = layers.Input(shape=(seq_len, num_features))
    
    # Encoder
    encoded = layers.LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = layers.LSTM(32, activation='relu', return_sequences=False)(encoded)
    
    # Bottleneck representation (16 dimensions)
    bottleneck = layers.Dense(16, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.RepeatVector(seq_len)(bottleneck)
    decoded = layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
    
    # Output layer reconstructing the input
    outputs = layers.TimeDistributed(layers.Dense(num_features))(decoded)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model


def generate_dummy_sequences(samples=1000):
    """Generate dummy healthy sequence data."""
    X = np.random.normal(0, 1, (samples, SEQ_LEN, NUM_FEATURES))
    return X.astype(np.float32)


def create_sliding_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """Create sliding windows of `window_size` from a 2D array (timesteps, features)."""
    if len(data) < window_size:
        # Pad with the mean if not enough data
        pad = np.tile(data.mean(axis=0), (window_size - len(data), 1))
        data = np.vstack([pad, data])
    
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows, dtype=np.float32)


def load_healthy_data(csv_path: str):
    """
    Load only HEALTHY samples from the dataset and create temporal windows.
    The LSTM-AE is trained ONLY on normal data — it learns to reconstruct
    healthy patterns. During inference, anomalies have high reconstruction error.
    """
    print(f"Loading healthy data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=80000)
    
    # Detect dataset type and extract features
    # Check if it's our pre-extracted 28-feature real telemetry CSV
    if set(FEATURE_NAMES_V1).issubset(set(df.columns)):
        print("  Detected: Real telemetry (28-feature pre-extracted from TSDB)")
        features = df[FEATURE_NAMES_V1].fillna(0.0)
        print(f"  Loaded {len(features):,} rows with all 28 features")
        X = features.values.astype(np.float32)
        return X
    elif any('smart' in c.lower() for c in df.columns):
        # Backblaze data
        print("  Detected: Backblaze SMART data")
        smart_cols = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                      'smart_194_raw', 'smart_197_raw', 'smart_198_raw']
        
        features = pd.DataFrame()
        for col in smart_cols:
            features[col] = df[col].fillna(0) if col in df.columns else 0
        
        # Derive additional features
        features['smart_5_delta'] = features['smart_5_raw'].diff().fillna(0)
        features['smart_9_delta'] = features['smart_9_raw'].diff().fillna(0)
        features['temp_rolling'] = features['smart_194_raw'].rolling(10, min_periods=1).mean()
        features['temp_std'] = features['smart_194_raw'].rolling(10, min_periods=1).std().fillna(0)
        features['error_ratio'] = features['smart_187_raw'] / (features['smart_9_raw'] + 1e-6)
        
        # Pad to 28
        while len(features.columns) < 28:
            features[f'_pad_{len(features.columns)}'] = 0.0
        features = features[features.columns[:28]]
        
        # Filter to healthy drives only (failure == 0 or ttf > 30 days)
        if 'failure' in df.columns:
            healthy = df['failure'] == 0
        elif 'ttf_days' in df.columns:
            healthy = df['ttf_days'] > 30
        else:
            healthy = pd.Series([True] * len(df))
        
        features = features[healthy].reset_index(drop=True)
        
    else:
        # AI4I 2020 data
        print("  Detected: AI4I 2020 data")
        
        # Find columns
        feature_mapping = {}
        for col in df.columns:
            cl = col.lower()
            if 'air' in cl and 'temp' in cl:
                feature_mapping['air_temp'] = col
            elif 'process' in cl and 'temp' in cl:
                feature_mapping['proc_temp'] = col
            elif 'rotational' in cl:
                feature_mapping['rot_speed'] = col
            elif 'torque' in cl:
                feature_mapping['torque'] = col
            elif 'tool' in cl and 'wear' in cl:
                feature_mapping['tool_wear'] = col
        
        features = pd.DataFrame()
        if 'air_temp' in feature_mapping:
            features['air_temp_c'] = df[feature_mapping['air_temp']] - 273.15
            features['air_temp_delta'] = features['air_temp_c'].diff().fillna(0)
            features['air_temp_roll'] = features['air_temp_c'].rolling(10, min_periods=1).mean()
            features['air_temp_std'] = features['air_temp_c'].rolling(10, min_periods=1).std().fillna(0)
        
        if 'proc_temp' in feature_mapping:
            features['proc_temp_c'] = df[feature_mapping['proc_temp']] - 273.15
            features['proc_temp_delta'] = features['proc_temp_c'].diff().fillna(0)
            features['temp_diff'] = features.get('air_temp_c', 0) - features['proc_temp_c']
        
        if 'rot_speed' in feature_mapping:
            features['rot_speed'] = df[feature_mapping['rot_speed']]
            features['rot_delta'] = features['rot_speed'].diff().fillna(0)
            features['rot_roll_std'] = features['rot_speed'].rolling(10, min_periods=1).std().fillna(0)
        
        if 'torque' in feature_mapping:
            features['torque'] = df[feature_mapping['torque']]
            features['torque_delta'] = features['torque'].diff().fillna(0)
        
        if 'tool_wear' in feature_mapping:
            features['tool_wear'] = df[feature_mapping['tool_wear']]
            features['wear_rate'] = features['tool_wear'].diff().fillna(0)
        
        if 'rot_speed' in feature_mapping and 'torque' in feature_mapping:
            features['power'] = df[feature_mapping['rot_speed']] * df[feature_mapping['torque']]
            features['efficiency'] = df[feature_mapping['torque']] / (df[feature_mapping['rot_speed']] + 1e-6)
        
        # Pad to 28
        while len(features.columns) < 28:
            features[f'_pad_{len(features.columns)}'] = 0.0
        features = features[features.columns[:28]]
        
        # Filter to healthy samples only
        failure_col = None
        for col in df.columns:
            if col.lower() in ['machine failure', 'machine_failure', 'target']:
                failure_col = col
        
        if failure_col:
            healthy = df[failure_col] == 0
            features = features[healthy].reset_index(drop=True)
    
    print(f"  Healthy samples: {len(features)}")
    
    # NOTE: Do NOT standardize here — standardization happens AFTER train/val split
    # in train_model() to prevent data leakage (BUG-N3 fix).
    X = features.values.astype(np.float32)
    
    return X  # Return raw features (not windowed yet — windowing happens after scaling)


def train_model(epochs=10, batch_size=32, dummy=False, data_path=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Building LSTM-Autoencoder...")
    model = build_lstm_autoencoder(SEQ_LEN, NUM_FEATURES)
    model.summary()
    
    if dummy:
        print("Training on dummy HEALTHY data...")
        X_train = generate_dummy_sequences(5000)
        X_val = generate_dummy_sequences(1000)
    elif data_path:
        X_raw = load_healthy_data(data_path)
        if X_raw is None or len(X_raw) < 100:
            print("ERROR: Not enough healthy data loaded. Use --dummy.")
            return
        
        # ── BUG-N7 FIX (2026-08-12): Correct temporal ordering ──
        # WRONG (old code): np.random.permutation(n) shuffled raw 2D rows BEFORE
        # creating sliding windows, destroying all temporal relationships.
        #
        # CORRECT ORDER:
        #   1. Split raw 2D data CHRONOLOGICALLY (first 80% train, last 20% val)
        #   2. Fit scaler on training split ONLY (prevents data leakage — BUG-N3)
        #   3. Scale both splits
        #   4. Create sliding windows on each scaled split INDEPENDENTLY
        #      (preserves temporal order within each window)
        #   5. Shuffle the resulting 3D WINDOWS for batch randomness
        #      (safe: each window's internal temporal order is preserved)
        
        n = len(X_raw)
        split = int(0.8 * n)
        X_train_raw = X_raw[:split]       # First 80% chronologically
        X_val_raw = X_raw[split:]         # Last 20% chronologically
        
        # Fit scaler on TRAINING data ONLY (BUG-N3 fix)
        from sklearn.preprocessing import StandardScaler
        import joblib
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # Save scaler for inference (BUG-N4 fix)
        scaler_path = os.path.join(MODEL_DIR, 'lstm_ae_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"  LSTM-AE scaler saved to {scaler_path}")
        
        # Create sliding windows AFTER scaling, on each split INDEPENDENTLY
        # This preserves temporal order within each window
        X_train = create_sliding_windows(X_train_scaled, SEQ_LEN)
        X_val = create_sliding_windows(X_val_scaled, SEQ_LEN)
        
        # Shuffle the WINDOWS (not raw rows) for training batch randomness
        # This is safe: each window's internal 10-step temporal sequence is preserved
        train_perm = np.random.permutation(len(X_train))
        X_train = X_train[train_perm]
        
        print(f"Train: {len(X_train)} windows, Val: {len(X_val)} windows")
    else:
        print("ERROR: Provide --data <path> or --dummy.")
        return
        
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'lstm_ae.h5'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    print("Starting training (unsupervised reconstruction)...")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Calculate reconstruction threshold using standardized 97th percentile adaptive quantile
    val_pred = model.predict(X_val, verbose=0)
    mse_per_sample = np.mean(np.square(X_val - val_pred), axis=(1, 2))
    threshold = float(np.percentile(mse_per_sample, 97))
    
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE!")
    print(f"  Best Val Loss (MSE): {min(history.history['val_loss']):.6f}")
    print(f"  Anomaly Threshold (97th percentile): {threshold:.6f}")
    print(f"  (Reconstruction error > threshold = anomaly)")
    print(f"  Model saved to: server/models/lstm_ae.h5")
    print(f"{'='*50}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM-Autoencoder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dummy", action="store_true", help="Train on dummy data")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data CSV")
    args = parser.parse_args()
    
    if not args.dummy and not args.data:
        # Auto-detect
        ai4i_path = os.path.join(DATA_DIR, 'ai4i_2020', 'ai4i2020.csv')
        bb_path = os.path.join(DATA_DIR, 'processed', 'backblaze_ttf.csv')
        
        if os.path.exists(ai4i_path):
            print(f"Auto-detected AI4I 2020 at {ai4i_path}")
            args.data = ai4i_path
        elif os.path.exists(bb_path):
            print(f"Auto-detected Backblaze at {bb_path}")
            args.data = bb_path
        else:
            print("No dataset found. Falling back to dummy training.")
            args.dummy = True
    
    train_model(epochs=args.epochs, batch_size=args.batch_size,
                dummy=args.dummy, data_path=args.data)
