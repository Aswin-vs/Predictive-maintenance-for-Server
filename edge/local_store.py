"""
Edge SQLite Ring Buffer — Local Telemetry Cache.

Stores the last 24 hours of telemetry per server on the RPi4 edge node.
Provides crash resilience (survives reboots) and supports offline operation
when the cloud backend is unreachable.

Features:
- Auto-creates database and tables on first use
- Auto-purges data older than 24 hours
- Per-server partitioning via server_id
- Lightweight: single SQLite file, no external dependencies

Author: @claude-architect (Phase D — Production-Ready Upgrade)
Date: 2026-08-09
"""

import os
import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

# Default database location
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data',
    'edge_cache.db'
)

# Retention period (hours)
DEFAULT_RETENTION_HOURS = 24


class EdgeLocalStore:
    """SQLite-backed ring buffer for caching telemetry on the edge node."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, retention_hours: int = DEFAULT_RETENTION_HOURS):
        self.db_path = db_path
        self.retention_hours = retention_hours
        self._ensure_db()
        self._purge_count = 0  # Purge every N inserts to avoid overhead
    
    def _ensure_db(self):
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    received_at REAL NOT NULL,
                    raw_payload TEXT NOT NULL,
                    predicted_class INTEGER,
                    confidence REAL,
                    ttf_days REAL,
                    anomaly_detected INTEGER DEFAULT 0,
                    forwarded_to_cloud INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_server_time 
                ON telemetry(server_id, received_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_forwarded 
                ON telemetry(forwarded_to_cloud)
            """)
            
            # Prediction results table (for cloud forwarding queue)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    predicted_class INTEGER,
                    confidence REAL,
                    ttf_days REAL,
                    anomaly_detected INTEGER DEFAULT 0,
                    inference_location TEXT DEFAULT 'edge',
                    model_version TEXT,
                    forwarded_to_cloud INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
        
        print(f"[STORE] Edge cache initialized at {self.db_path}")
    
    @contextmanager
    def _get_conn(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read/write
        try:
            yield conn
        finally:
            conn.close()
    
    def store_telemetry(self, server_id: str, payload: Dict[str, Any],
                        prediction: Optional[Dict] = None):
        """
        Store a telemetry reading and its prediction result.
        
        Args:
            server_id: UUID of the patient server
            payload: Raw telemetry dict
            prediction: Prediction result dict (class, confidence, ttf, etc.)
        """
        now = time.time()
        timestamp = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
        
        pred_class = None
        confidence = None
        ttf_days = None
        anomaly = 0
        
        if prediction:
            preds = prediction.get("predictions", {})
            pred_class = preds.get("class")
            confidence = preds.get("confidence")
            if "ttf_days" in preds and preds["ttf_days"] is not None:
                ttf_days = float(preds["ttf_days"])
            else:
                # Realistic bounded TTF calculation (Paper 2 & Blueprint §5)
                if pred_class == 0 or pred_class is None:
                    ttf_days = 90.0
                else:
                    BASE_FAULT_TTF = {1: 14.0, 2: 2.0, 3: 7.0, 4: 5.0, 5: 3.0, 6: 7.0}
                    base = BASE_FAULT_TTF.get(pred_class, 7.0)
                    conf = float(confidence or 0.85)
                    conf_scale = max(0.2, (1.0 - (conf - 0.7) / 0.3 * 0.7))
                    ttf_days = max(0.5, round(base * conf_scale, 1))
            anomaly = 1 if prediction.get("anomaly_detected", False) else 0
        
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO telemetry 
                (server_id, timestamp, received_at, raw_payload, 
                 predicted_class, confidence, ttf_days, anomaly_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (server_id, timestamp, now, json.dumps(payload),
                  pred_class, confidence, ttf_days, anomaly))
            conn.commit()
            record_id = cursor.lastrowid
        
        # Auto-purge every 100 inserts
        self._purge_count += 1
        if self._purge_count >= 100:
            self.purge_old_data()
            self._purge_count = 0
            
        return record_id
    
    def get_recent(self, server_id: Optional[str] = None, hours: float = 24.0, limit: int = 100) -> List[Dict]:
        """Get recent telemetry for a server or all servers within the last N hours."""
        cutoff = time.time() - (hours * 3600)
        
        with self._get_conn() as conn:
            if server_id and server_id.lower() not in ("all", "none", "", "default"):
                cursor = conn.execute("""
                    SELECT id, timestamp, raw_payload, predicted_class, confidence, 
                           ttf_days, anomaly_detected, server_id
                    FROM telemetry
                    WHERE server_id = ? AND received_at > ?
                    ORDER BY received_at DESC
                    LIMIT ?
                """, (server_id, cutoff, limit))
            else:
                cursor = conn.execute("""
                    SELECT id, timestamp, raw_payload, predicted_class, confidence, 
                           ttf_days, anomaly_detected, server_id
                    FROM telemetry
                    WHERE received_at > ?
                    ORDER BY received_at DESC
                    LIMIT ?
                """, (cutoff, limit))
            
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            try:
                payload_data = json.loads(row[2]) if row[2] else {}
            except Exception:
                payload_data = {}
            results.append({
                "id": row[0],
                "timestamp": row[1],
                "payload": payload_data,
                "predicted_class": row[3],
                "confidence": row[4],
                "ttf_days": row[5],
                "anomaly_detected": bool(row[6]),
                "server_id": row[7],
            })
        
        return results

    def clear_telemetry(self, server_id: Optional[str] = None) -> int:
        """
        Purge telemetry entries from the SQLite cache.
        If server_id is None, 'all', or empty, wipes the entire telemetry table.
        Returns the number of deleted rows.
        """
        with self._get_conn() as conn:
            if server_id and server_id.lower() not in ("all", "none", "", "default"):
                cursor = conn.execute("DELETE FROM telemetry WHERE server_id = ?", (server_id,))
            else:
                cursor = conn.execute("DELETE FROM telemetry")
            deleted = cursor.rowcount
            conn.commit()
        return deleted
    
    def get_unforwarded(self, limit: int = 50) -> List[Dict]:
        """Get telemetry entries that haven't been forwarded to the cloud."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT id, server_id, timestamp, raw_payload, 
                       predicted_class, confidence, ttf_days, anomaly_detected
                FROM telemetry
                WHERE forwarded_to_cloud = 0
                ORDER BY received_at ASC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "server_id": row[1],
                "timestamp": row[2],
                "payload": json.loads(row[3]),
                "predicted_class": row[4],
                "confidence": row[5],
                "ttf_days": row[6],
                "anomaly_detected": bool(row[7]),
            })
        
        return results
    
    def mark_forwarded(self, entry_ids: List[int]):
        """Mark entries as successfully forwarded to the cloud."""
        if not entry_ids:
            return
        
        placeholders = ",".join("?" * len(entry_ids))
        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE telemetry SET forwarded_to_cloud = 1 WHERE id IN ({placeholders})",
                entry_ids
            )
            conn.commit()
    
    def count_data_days(self, server_id: str) -> int:
        """Count the number of distinct days of data for a server."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT date(timestamp)) 
                FROM telemetry 
                WHERE server_id = ?
            """, (server_id,))
            return cursor.fetchone()[0] or 0
    
    def get_server_ids(self) -> List[str]:
        """Get all unique server IDs in the cache."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT server_id FROM telemetry ORDER BY server_id"
            )
            return [row[0] for row in cursor.fetchall()]
    
    def purge_old_data(self):
        """Remove data older than the retention period."""
        cutoff = time.time() - (self.retention_hours * 3600)
        
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM telemetry WHERE received_at < ?", (cutoff,)
            )
            deleted = cursor.rowcount
            conn.commit()
        
        if deleted > 0:
            print(f"[STORE] Purged {deleted} entries older than {self.retention_hours}h")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the cache."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
            servers = conn.execute("SELECT COUNT(DISTINCT server_id) FROM telemetry").fetchone()[0]
            unforwarded = conn.execute(
                "SELECT COUNT(*) FROM telemetry WHERE forwarded_to_cloud = 0"
            ).fetchone()[0]
            anomalies = conn.execute(
                "SELECT COUNT(*) FROM telemetry WHERE anomaly_detected = 1"
            ).fetchone()[0]
            
            # Database file size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        return {
            "total_entries": total,
            "unique_servers": servers,
            "unforwarded": unforwarded,
            "anomalies": anomalies,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "retention_hours": self.retention_hours,
        }


if __name__ == "__main__":
    # Quick test
    store = EdgeLocalStore()
    
    # Insert test data
    test_payload = {"cpu_temp": 65.0, "gpu_temp": 55.0, "timestamp": datetime.now(timezone.utc).isoformat()}
    test_prediction = {"predictions": {"class": 0, "confidence": 0.95, "ttf_normalized": 0.8}, "anomaly_detected": False}
    
    store.store_telemetry("test-server-001", test_payload, test_prediction)
    
    print(f"\nStats: {json.dumps(store.get_stats(), indent=2)}")
    print(f"Recent: {store.get_recent('test-server-001', hours=1)}")
    print(f"Server IDs: {store.get_server_ids()}")
