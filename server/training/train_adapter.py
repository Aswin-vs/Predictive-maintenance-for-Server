"""
Transfer Learning Adapter Trainer -- Personalized Model Fine-Tuning.

Trains a lightweight per-server adapter on top of the frozen base 1D-CNN model.
Uses real telemetry data from TimescaleDB, auto-labels it using the self-supervised
labeling engine, and produces a personalized INT8 TFLite model for edge deployment.

Architecture:
    Base Model (frozen): Conv1D(32) -> Conv1D(64) -> MaxPool -> Flatten
    Adapter (trainable): Dense(64, relu) -> Dropout(0.3) -> [ClassHead, TTFHead]

This module MUST run on the LOQ RTX 4060 (requires TensorFlow + GPU).

Author: @claude-architect (Phase D -- Production-Ready Upgrade)
Date: 2026-08-09
"""

import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from server.training.feature_schema import FEATURE_NAMES_V1, FEATURE_COUNT, NUM_CLASSES

# Lazy imports for heavy deps (only available on LOQ)
tf = None
joblib = None

def _lazy_imports():
    global tf, joblib
    if tf is None:
        import tensorflow as _tf
        tf = _tf
    if joblib is None:
        import joblib as _joblib
        joblib = _joblib


MODEL_DIR = os.path.join(BASE_DIR, 'server', 'models')
EDGE_MODEL_DIR = os.path.join(BASE_DIR, 'edge', 'models')


def load_base_model(model_path: str = None):
    """
    Load the base FP32 1D-CNN model and freeze the convolutional layers.
    
    Returns:
        tf.keras.Model with frozen conv layers and trainable adapter layers.
    """
    _lazy_imports()
    
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'classifier_fp32.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model not found: {model_path}")
    
    # Load with compile=False to avoid custom loss issues
    from server.training.asymmetric_loss import register_custom_objects
    register_custom_objects()
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Freeze convolutional layers (keep adapter layers trainable)
    freeze_count = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.MaxPooling1D,
                              tf.keras.layers.BatchNormalization)):
            layer.trainable = False
            freeze_count += 1
        elif isinstance(layer, tf.keras.layers.Flatten):
            layer.trainable = False
            freeze_count += 1
    
    print(f"[ADAPTER] Froze {freeze_count} base layers. "
          f"Trainable params: {sum(p.numpy().size for p in model.trainable_weights)}")
    
    return model


def query_training_data(server_id: str, days: int = 30):
    """
    Query TimescaleDB for a server's telemetry data.
    Returns raw feature DataFrame.
    """
    try:
        from server.deep_analysis.tsdb_writer import (
            TimescaleDBWriter, PSYCOPG2_AVAILABLE,
            DEFAULT_DB_HOST, DEFAULT_DB_PORT, DEFAULT_DB_NAME,
            DEFAULT_DB_USER, DEFAULT_DB_PASS
        )
    except ImportError:
        PSYCOPG2_AVAILABLE = False

    if not PSYCOPG2_AVAILABLE:
        print("[ADAPTER] PostgreSQL/TimescaleDB client not available. Cannot query training data.")
        return None

    try:
        writer = TimescaleDBWriter(
            DEFAULT_DB_HOST, DEFAULT_DB_PORT, DEFAULT_DB_NAME,
            DEFAULT_DB_USER, DEFAULT_DB_PASS
        )
        if not writer.is_connected():
            print("[ADAPTER] TimescaleDB not connected. Cannot query training data.")
            return None

        df = writer.query_training_data(server_id=server_id, days=days)
        if df is not None:
            print(f"[ADAPTER] Loaded {len(df)} samples from TimescaleDB for {server_id}")
        else:
            print(f"[ADAPTER] No data found in TimescaleDB for server {server_id}")
        return df
    except Exception as e:
        print(f"[ADAPTER] TimescaleDB query failed: {e}")
        return None


def train_adapter(server_id: str, reason: str = "scheduled",
                  epochs: int = 20, batch_size: int = 32,
                  data_path: str = None):
    """
    Train a personalized adapter for a specific server.
    
    This is the main entry point called by the retrain_manager.
    
    Args:
        server_id: UUID of the patient server
        reason: Why training was triggered
        epochs: Number of fine-tuning epochs
        batch_size: Training batch size
        data_path: Optional CSV path (fallback if TimescaleDB is unavailable)
    """
    _lazy_imports()
    from sklearn.preprocessing import StandardScaler
    from server.training.auto_labeler import AutoLabeler
    from server.deep_analysis.model_registry import ModelRegistry
    import pandas as pd
    
    print(f"\n{'='*60}")
    print(f"  ADAPTER TRAINING -- Server: {server_id[:12]}...")
    print(f"  Reason: {reason}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")
    
    registry = ModelRegistry()
    
    # Step 1: Load training data
    print("[1/7] Loading training data...")
    df = None
    
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} samples from {data_path}")
    else:
        df = query_training_data(server_id)
    
    if df is None or len(df) < 100:
        print(f"  ERROR: Insufficient data ({0 if df is None else len(df)} samples). Need >= 100.")
        return False
    
    # Step 2: Extract features in canonical order
    print("[2/7] Extracting features (FEATURE_SCHEMA_V1 order)...")
    available_features = [f for f in FEATURE_NAMES_V1 if f in df.columns]
    missing_features = [f for f in FEATURE_NAMES_V1 if f not in df.columns]
    
    if missing_features:
        print(f"  Warning: {len(missing_features)} features missing, filling defaults")
        from server.training.feature_schema import FEATURE_DEFAULTS
        for f in missing_features:
            df[f] = FEATURE_DEFAULTS.get(f, 0.0)
    
    X = df[FEATURE_NAMES_V1].values.astype(np.float32)
    
    # Step 3: Auto-label the data
    print("[3/7] Auto-labeling telemetry...")
    labeler = AutoLabeler()
    y_class, y_ttf = labeler.label_dataframe(df[FEATURE_NAMES_V1])
    
    # Step 4: Train/val split, then fit scaler on TRAIN ONLY
    print("[4/7] Splitting and scaling (no data leakage)...")
    n = len(X)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_class_train, y_class_val = y_class[train_idx], y_class[val_idx]
    y_ttf_train, y_ttf_val = y_ttf[train_idx], y_ttf[val_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).reshape(-1, FEATURE_COUNT, 1).astype(np.float32)
    X_val = scaler.transform(X_val_raw).reshape(-1, FEATURE_COUNT, 1).astype(np.float32)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Step 5: Load base model and fine-tune adapter
    print("[5/7] Loading base model and fine-tuning adapter...")
    model = load_base_model()
    
    from server.training.asymmetric_loss import get_asymmetric_ttf_loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'ttf_output': get_asymmetric_ttf_loss()
        },
        loss_weights={'class_output': 1.0, 'ttf_output': 0.5},
        metrics={'class_output': 'accuracy', 'ttf_output': 'mae'}
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]
    
    history = model.fit(
        X_train,
        {'class_output': y_class_train, 'ttf_output': y_ttf_train},
        validation_data=(X_val, {'class_output': y_class_val, 'ttf_output': y_ttf_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    val_acc = max(history.history.get('val_class_output_accuracy', [0]))
    val_mae = min(history.history.get('val_ttf_output_mae', [999]))
    
    print(f"\n  Val Accuracy: {val_acc:.4f}")
    print(f"  Val TTF MAE:  {val_mae:.4f}")
    
    # Step 6: Save adapter model + scaler
    print("[6/7] Saving adapter model and scaler...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EDGE_MODEL_DIR, exist_ok=True)
    
    short_id = server_id[:8]
    model_filename = f'adapter_{short_id}.h5'
    scaler_filename = f'scaler_{short_id}.pkl'
    
    model_path = os.path.join(MODEL_DIR, model_filename)
    scaler_path = os.path.join(MODEL_DIR, scaler_filename)
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    # Also save scaler to edge models
    edge_scaler_path = os.path.join(EDGE_MODEL_DIR, scaler_filename)
    joblib.dump(scaler, edge_scaler_path)
    
    print(f"  Model: {model_path}")
    print(f"  Scaler: {scaler_path}")
    
    # Step 7: Quantize to INT8 TFLite for edge deployment
    print("[7/7] Quantizing to INT8 TFLite...")
    tflite_filename = f'adapter_{short_id}_int8.tflite'
    tflite_path = os.path.join(EDGE_MODEL_DIR, tflite_filename)
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset from training data
        def rep_dataset():
            for i in range(min(100, len(X_train))):
                yield [X_train[i:i+1]]
        
        converter.representative_dataset = rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"  TFLite: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")
        
    except Exception as e:
        print(f"  Warning: Full INT8 quantization failed ({e}). Using default optimization.")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"  TFLite (dynamic range): {tflite_path} ({len(tflite_model)/1024:.1f} KB)")
    
    # Register model in registry
    version = registry.register_model(
        server_id=server_id,
        model_type="adapter",
        model_path=model_path,
        scaler_path=scaler_path,
        tflite_path=tflite_path,
        training_samples=len(X_train),
        val_accuracy=val_acc,
        val_mae=val_mae,
        notes=f"Adapter trained: {reason}. {epochs} epochs."
    )
    
    print(f"\n{'='*60}")
    print(f"  ADAPTER TRAINING COMPLETE -- Version {version}")
    print(f"  Deploy to edge: python server/deployment/deploy_model.py --server-id {server_id}")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Personalized Adapter")
    parser.add_argument("--server-id", type=str, required=True, help="Server UUID")
    parser.add_argument("--reason", type=str, default="manual", help="Trigger reason")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--data", type=str, default=None, help="CSV data path (fallback)")
    args = parser.parse_args()
    
    train_adapter(args.server_id, args.reason, args.epochs, data_path=args.data)
