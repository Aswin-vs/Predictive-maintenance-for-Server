"""
H3.1: Auto-Retrain Pipeline — Phase H
========================================
On-demand per-server model retraining from collected telemetry data.

Runs on LOQ cloud. Triggered via:
  - POST /api/retrain/{server_id} (from dashboard)
  - Automatic after 72h of data collection
  - Manual CLI: python auto_retrain.py --server-id <id>

Trains:
  1. Per-server Isolation Forest (fast, ~5s)
  2. Per-server LSTM-AE (longer, ~30-60s on GPU)
  3. Updates adaptive thresholds in TimescaleDB

Author: Phase H
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from server.utils.logger import get_logger

logger = get_logger("auto_retrain")

MODELS_DIR = os.path.join(BASE_DIR, "edge", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _get_tsdb_conn():
    """Get TimescaleDB connection."""
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("TSDB_HOST", "localhost"),
        port=int(os.getenv("TSDB_PORT", "5432")),
        dbname=os.getenv("TSDB_DB", "predictive_maintenance"),
        user=os.getenv("TSDB_USER", "pm_user"),
        password=os.getenv("TSDB_PASS", "pm_edge_2026"),
        connect_timeout=5,
    )


def fetch_training_data(server_id: str, hours: int = 72) -> Optional[np.ndarray]:
    """Fetch telemetry data from TimescaleDB for a specific server."""
    try:
        conn = _get_tsdb_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT cpu_temp, cpu_load, gpu_temp, gpu_power, gpu_util,
                   ram_used_gb, ram_percent, fan_rpm, psu_12v_dev,
                   ambient_temp, vibration_rms, disk_usage_pct
            FROM telemetry_raw
            WHERE server_id = %s
              AND time > NOW() - INTERVAL '1 hour' * %s
            ORDER BY time ASC
        """, (server_id, hours))
        rows = cur.fetchall()
        conn.close()

        if len(rows) < 100:
            logger.warning(f"[{server_id}] Only {len(rows)} rows — need at least 100 for training")
            return None

        data = np.array(rows, dtype=np.float32)
        # Replace NaNs with column means
        col_means = np.nanmean(data, axis=0)
        for i in range(data.shape[1]):
            mask = np.isnan(data[:, i])
            data[mask, i] = col_means[i]

        logger.info(f"[{server_id}] Fetched {len(rows)} rows ({data.shape[1]} features) for training")
        return data

    except Exception as e:
        logger.error(f"[{server_id}] Failed to fetch training data: {e}")
        return None


def train_isolation_forest(server_id: str, data: np.ndarray) -> Dict[str, Any]:
    """Train a per-server Isolation Forest model."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    logger.info(f"[{server_id}] Training Isolation Forest ({data.shape[0]} samples, {data.shape[1]} features)...")
    start = time.time()

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Assume 5% anomaly rate
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Save model and scaler
    server_model_dir = os.path.join(MODELS_DIR, server_id.replace("/", "_"))
    os.makedirs(server_model_dir, exist_ok=True)

    model_path = os.path.join(server_model_dir, "iso_forest_adaptive.pkl")
    scaler_path = os.path.join(server_model_dir, "scaler_adaptive.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    elapsed = time.time() - start
    logger.info(f"[{server_id}] Isolation Forest trained in {elapsed:.1f}s → {model_path}")

    return {
        "model": "isolation_forest",
        "server_id": server_id,
        "samples": data.shape[0],
        "features": data.shape[1],
        "training_time_s": round(elapsed, 2),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "contamination": 0.05,
    }


def train_lstm_ae(server_id: str, data: np.ndarray, window_size: int = 10) -> Dict[str, Any]:
    """Train a per-server LSTM Autoencoder for temporal anomaly detection."""
    logger.info(f"[{server_id}] Training LSTM-AE ({data.shape[0]} samples, window={window_size})...")
    start = time.time()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Split chronologically 90/10 BEFORE scaling to prevent data leakage (BUG-03)
        split = int(0.9 * len(data))
        train_raw = data[:split]
        val_raw = data[split:]

        # Normalize: Fit StandardScaler strictly on the train partition
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        val_scaled = scaler.transform(val_raw)

        # Create sliding windows separately to prevent boundary leakage
        def make_windows(arr, w):
            wins = []
            for i in range(len(arr) - w + 1):
                wins.append(arr[i:i + w])
            if not wins:
                return np.empty((0, w, arr.shape[1]), dtype=np.float32)
            return np.array(wins, dtype=np.float32)

        X_train_windows = make_windows(train_scaled, window_size)
        X_val_windows = make_windows(val_scaled, window_size)

        X_train = torch.tensor(X_train_windows)
        X_val = torch.tensor(X_val_windows)

        n_features = X_train.shape[2]

        # Simple LSTM-AE
        class LSTMAE(nn.Module):
            def __init__(self, n_feat, hidden=32):
                super().__init__()
                self.encoder = nn.LSTM(n_feat, hidden, batch_first=True)
                self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
                self.output = nn.Linear(hidden, n_feat)

            def forward(self, x):
                _, (h, c) = self.encoder(x)
                h_repeat = h.repeat(1, x.size(1), 1).transpose(0, 1)
                dec_out, _ = self.decoder(h_repeat)
                return self.output(dec_out)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMAE(n_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_loader = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True)

        # Train for 20 epochs
        for epoch in range(20):
            model.train()
            total_loss = 0
            for (batch,) in train_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Compute reconstruction threshold from validation set
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_errors = torch.mean((val_out - X_val.to(device)) ** 2, dim=(1, 2)).cpu().numpy()
            threshold = float(np.percentile(val_errors, 97))

        # Save
        server_model_dir = os.path.join(MODELS_DIR, server_id.replace("/", "_"))
        os.makedirs(server_model_dir, exist_ok=True)

        model_path = os.path.join(server_model_dir, "lstm_ae_adaptive.pt")
        scaler_path = os.path.join(server_model_dir, "lstm_ae_scaler.pkl")

        torch.save(model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Save metadata
        meta = {
            "threshold": threshold,
            "n_features": n_features,
            "window_size": window_size,
            "hidden_size": 32,
            "training_samples": len(X_train),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(server_model_dir, "lstm_ae_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - start
        logger.info(f"[{server_id}] LSTM-AE trained in {elapsed:.1f}s (threshold={threshold:.4f})")

        return {
            "model": "lstm_ae",
            "server_id": server_id,
            "samples": len(X_train),
            "window_size": window_size,
            "threshold": threshold,
            "training_time_s": round(elapsed, 2),
            "device": str(device),
        }

    except ImportError:
        logger.warning(f"[{server_id}] PyTorch not available — skipping LSTM-AE training")
        return {"model": "lstm_ae", "error": "PyTorch not installed"}
    except Exception as e:
        logger.error(f"[{server_id}] LSTM-AE training failed: {e}")
        return {"model": "lstm_ae", "error": str(e)}


def retrain_server(server_id: str, hours: int = 72) -> Dict[str, Any]:
    """Full retrain pipeline for a server."""
    logger.info(f"[{server_id}] Starting retrain pipeline (last {hours}h of data)...")

    data = fetch_training_data(server_id, hours)
    if data is None:
        return {"status": "error", "message": "Not enough training data"}

    results = {
        "server_id": server_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_shape": list(data.shape),
    }

    # Train Isolation Forest
    results["isolation_forest"] = train_isolation_forest(server_id, data)

    # Train LSTM-AE
    results["lstm_ae"] = train_lstm_ae(server_id, data)

    # Update adaptive thresholds
    try:
        from server.deep_analysis.adaptive_thresholds import AdaptiveThresholdEngine
        engine = AdaptiveThresholdEngine()
        engine.recompute(server_id)
        results["thresholds"] = "updated"
    except Exception as e:
        results["thresholds"] = f"error: {e}"

    results["status"] = "complete"
    logger.info(f"[{server_id}] Retrain pipeline complete")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-retrain models for a server")
    parser.add_argument("--server-id", required=True, help="Server ID to retrain")
    parser.add_argument("--hours", type=int, default=72, help="Hours of data to use")
    args = parser.parse_args()

    result = retrain_server(args.server_id, args.hours)
    print(json.dumps(result, indent=2))
