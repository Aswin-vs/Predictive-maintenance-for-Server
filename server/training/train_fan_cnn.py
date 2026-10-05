"""
G3.4: 1D-CNN Fan Bearing Detector Training
===========================================
Trains a lightweight 1D-CNN to detect fan bearing degradation from
PWM/RPM correlation data. Uses synthetic training data from
synthetic_failure_gen.py.

Architecture: 3 conv layers (32->64->32), Global Average Pooling, Dense(1, sigmoid)
Input:  (batch, 60, 3) -- [PWM_duty_%, actual_RPM, PWM_RPM_ratio]
Output: Binary -- 0=healthy, 1=bearing degradation
Target: <50 KB INT8 TFLite | <10ms on RPi4
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from synthetic_failure_gen import generate_training_dataset

# ─── Configuration ──────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EDGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "edge", "models")

SEQUENCE_LENGTH = 60   # 60 ticks (e.g., 60 seconds at 1Hz)
N_FEATURES = 3         # PWM%, RPM, ratio
N_HEALTHY = 3000
N_DEGRADING = 3000
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3


# ─── Model ──────────────────────────────────────────────────

def build_fan_cnn_model(input_shape=(60, 3)):
    """
    Lightweight 1D-CNN for fan bearing detection.
    
    Architecture:
    - Conv1D(32, 5) -> BatchNorm -> ReLU -> MaxPool
    - Conv1D(64, 3) -> BatchNorm -> ReLU -> MaxPool
    - Conv1D(32, 3) -> BatchNorm -> ReLU
    - GlobalAveragePooling
    - Dense(16, relu) -> Dropout(0.3)
    - Dense(1, sigmoid)
    """
    inputs = keras.Input(shape=input_shape, name='fan_input')
    
    # Block 1
    x = layers.Conv1D(32, 5, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling1D(2, name='pool1')(x)
    
    # Block 2
    x = layers.Conv1D(64, 3, padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling1D(2, name='pool2')(x)
    
    # Block 3
    x = layers.Conv1D(32, 3, padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    
    # Classification head
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(16, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='bearing_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Fan_Bearing_CNN')
    return model


# ─── Training ───────────────────────────────────────────────

def train_fan_cnn():
    """Full fan bearing CNN training pipeline."""
    
    print("=" * 60)
    print("  G3.4: Fan Bearing 1D-CNN -- Training Pipeline")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    X, y = generate_training_dataset(
        n_healthy=N_HEALTHY,
        n_degrading=N_DEGRADING,
        sequence_length=SEQUENCE_LENGTH,
        save_dir="data/synthetic_fan"
    )
    
    # Step 2: Train/val/test split BEFORE scaling (prevents data leakage -- BUG-N3 pattern)
    n_samples, n_steps, n_feat = X.shape
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Step 3: Fit scaler on TRAINING data ONLY (BUG-N3 fix: no data leakage)
    print("\nNormalizing features (fit on training set only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)
    X_val = scaler.transform(X_val_raw.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)
    X_test = scaler.transform(X_test_raw.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Step 4: Build model
    print("\nBuilding Fan Bearing CNN...")
    model = build_fan_cnn_model(input_shape=(SEQUENCE_LENGTH, N_FEATURES))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    # Step 5: Train
    print("\nTraining Fan Bearing CNN...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=8, mode='max', restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
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
    
    # Step 6: Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC:      {test_auc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")
    
    # Classification report
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Degrading']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Step 7: Save FP32 model
    os.makedirs(MODEL_DIR, exist_ok=True)
    fp32_path = os.path.join(MODEL_DIR, "fan_bearing_cnn_fp32.keras")
    model.save(fp32_path)
    print(f"\nSaved FP32 model: {fp32_path}")
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "fan_bearing_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler: {scaler_path}")
    
    # Step 8: Quantize to INT8 TFLite
    print("\nQuantizing to INT8 TFLite...")
    
    def representative_dataset():
        for i in range(min(300, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32  # Keep output as float for probability
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(MODEL_DIR, "fan_bearing_cnn_int8.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_kb = os.path.getsize(tflite_path) / 1024
    print(f"Saved INT8 TFLite: {tflite_path} ({tflite_size_kb:.1f} KB)")
    
    # Copy to edge models directory
    os.makedirs(EDGE_MODEL_DIR, exist_ok=True)
    import shutil
    edge_tflite_path = os.path.join(EDGE_MODEL_DIR, "fan_bearing_cnn_int8.tflite")
    shutil.copy2(tflite_path, edge_tflite_path)
    
    edge_scaler_path = os.path.join(EDGE_MODEL_DIR, "fan_bearing_scaler.joblib")
    shutil.copy2(scaler_path, edge_scaler_path)
    print(f"Copied to edge: {edge_tflite_path}")
    
    # Step 9: Verify INT8 model accuracy
    print("\nVerifying INT8 model accuracy...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get quantization parameters
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    
    correct = 0
    n_test = min(200, len(X_test))
    for i in range(n_test):
        # Quantize input
        input_data = X_test[i:i+1].astype(np.float32)
        if input_details[0]['dtype'] == np.int8:
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        pred = 1 if output[0][0] > 0.5 else 0
        if pred == int(y_test[i]):
            correct += 1
    
    int8_acc = correct / n_test
    print(f"INT8 Accuracy: {int8_acc:.4f} (vs FP32: {test_acc:.4f})")
    
    # Save metadata
    metadata = {
        "model_name": "Fan Bearing 1D-CNN",
        "phase": "G3.4 + G3.7",
        "architecture": "3x Conv1D (32->64->32) + GAP + Dense(16) + Dense(1)",
        "input_shape": [SEQUENCE_LENGTH, N_FEATURES],
        "features": ["PWM_duty_pct", "actual_RPM", "PWM_RPM_ratio"],
        "output": "Binary (0=healthy, 1=degrading)",
        "training_data": f"Synthetic ({N_HEALTHY} healthy + {N_DEGRADING} degrading)",
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "int8_accuracy": float(int8_acc),
        "fp32_path": fp32_path,
        "tflite_path": tflite_path,
        "tflite_size_kb": float(tflite_size_kb),
        "scaler_path": scaler_path,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epochs_run": len(history.history['loss']),
    }
    
    meta_path = os.path.join(MODEL_DIR, "fan_bearing_cnn_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")
    
    print("\n" + "=" * 60)
    print(f"  Fan Bearing CNN -- TRAINING COMPLETE")
    print(f"  Test Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print(f"  INT8 Acc: {int8_acc:.4f} | Size: {tflite_size_kb:.1f} KB")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    train_fan_cnn()
