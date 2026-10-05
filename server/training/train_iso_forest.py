"""
Isolation Forest Training Module.

Trains an Isolation Forest model to detect anomalies in PSU (voltage)
and Fan (RPM, acoustic) data. Since we don't have labeled anomalies
for these components, we use an unsupervised approach.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'server', 'models')
EDGE_MODEL_DIR = os.path.join(BASE_DIR, 'edge', 'models')

# Features we expect to use for Isolation Forest
IF_FEATURES = [
    'psu_voltage_3v3_dev', 
    'psu_voltage_12v_dev',
    'psu_voltage_stability',
    'fan_rpm_deviation'
]

def generate_dummy_psu_data(n_samples=1000):
    """Generate some normal data and a few anomalies to train IF."""
    np.random.seed(42)
    # Normal data
    normal = {
        'psu_voltage_3v3_dev': np.abs(np.random.normal(0, 1.0, n_samples)),
        'psu_voltage_12v_dev': np.abs(np.random.normal(0, 1.5, n_samples)),
        'psu_voltage_stability': np.random.uniform(0.01, 0.1, n_samples),
        'fan_rpm_deviation': np.random.normal(0, 0.05, n_samples)
    }
    df = pd.DataFrame(normal)
    
    # Inject some anomalies (approx 5%)
    n_anomalies = int(n_samples * 0.05)
    anomalies = {
        'psu_voltage_3v3_dev': np.random.uniform(5.0, 15.0, n_anomalies),
        'psu_voltage_12v_dev': np.random.uniform(5.0, 15.0, n_anomalies),
        'psu_voltage_stability': np.random.uniform(0.5, 2.0, n_anomalies),
        'fan_rpm_deviation': np.random.uniform(-0.5, -0.2, n_anomalies) # stalled fans
    }
    df_anom = pd.DataFrame(anomalies)
    
    return pd.concat([df, df_anom]).sample(frac=1.0).reset_index(drop=True)

def train_isolation_forest(data_path=None, contamination=0.05, save_model=True):
    print("Training Isolation Forest for PSU/Fan Anomaly Detection...")
    
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        # Ensure required features exist
        missing = [f for f in IF_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")
    else:
        print("No real data provided. Generating synthetic training data...")
        df = generate_dummy_psu_data()
        
    X = df[IF_FEATURES]
    
    # Create pipeline with scaler and isolation forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iforest', IsolationForest(
            n_estimators=100, 
            contamination=contamination,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        ))
    ])
    
    print("Fitting model...")
    pipeline.fit(X)
    
    # Evaluate on the training set to see anomaly distribution
    preds = pipeline.predict(X)
    anomaly_count = (preds == -1).sum()
    print(f"Detected {anomaly_count} anomalies out of {len(X)} samples "
          f"({anomaly_count/len(X)*100:.2f}%). Expected ~{contamination*100:.2f}%.")
    
    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(EDGE_MODEL_DIR, exist_ok=True)
        
        model_path = os.path.join(MODEL_DIR, 'iso_forest.pkl')
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")
        
        # Also save to edge models for RPi4 deployment
        edge_model_path = os.path.join(EDGE_MODEL_DIR, 'iso_forest.pkl')
        joblib.dump(pipeline, edge_model_path)
        print(f"Model copied to {edge_model_path}")
        
    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to processed telemetry CSV")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly rate")
    args = parser.parse_args()
    
    train_isolation_forest(args.data, args.contamination)
