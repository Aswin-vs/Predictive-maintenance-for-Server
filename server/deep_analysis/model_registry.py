"""
Model Registry — SQLite-backed registry for server and model management.

Tracks:
- Registered patient servers and their hardware profiles
- Model versions, training metrics, and deployment status
- Retraining logs and trigger history

Author: @claude-architect (Phase D — Production-Ready Upgrade)
Date: 2026-08-09
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

# Default database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'server', 'models', 'model_registry.db'
)


class ModelRegistry:
    """SQLite-backed registry for servers, models, and retraining state."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self):
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS server_registry (
                    server_id TEXT PRIMARY KEY,
                    server_name TEXT,
                    hardware_profile TEXT,
                    registered_at TEXT NOT NULL,
                    data_mode TEXT DEFAULT 'production',
                    total_data_days INTEGER DEFAULT 0,
                    last_retrain_at TEXT,
                    active_model_version INTEGER,
                    status TEXT DEFAULT 'active'
                );
                
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    trained_at TEXT NOT NULL,
                    training_samples INTEGER,
                    training_days INTEGER,
                    val_accuracy REAL,
                    val_mae REAL,
                    model_path TEXT NOT NULL,
                    scaler_path TEXT,
                    tflite_path TEXT,
                    status TEXT DEFAULT 'active',
                    deployed_to_edge INTEGER DEFAULT 0,
                    deployed_at TEXT,
                    notes TEXT,
                    FOREIGN KEY (server_id) REFERENCES server_registry(server_id)
                );
                
                CREATE TABLE IF NOT EXISTS retraining_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server_id TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    trigger_reason TEXT,
                    completed_at TEXT,
                    status TEXT DEFAULT 'pending',
                    old_model_version INTEGER,
                    new_model_version INTEGER,
                    improvement_pct REAL,
                    error_log TEXT,
                    FOREIGN KEY (server_id) REFERENCES server_registry(server_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_server 
                ON model_registry(server_id, model_type);
                
                CREATE INDEX IF NOT EXISTS idx_retrain_server 
                ON retraining_log(server_id, triggered_at);
            """)
        
        print(f"[REGISTRY] Model registry initialized at {self.db_path}")
    
    @contextmanager
    def _get_conn(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # --- Server Registration ---
    
    def register_server(self, server_id: str, server_name: str,
                        hardware_profile: Dict, data_mode: str = "production") -> bool:
        """Register a new patient server or update an existing one."""
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT server_id FROM server_registry WHERE server_id = ?",
                (server_id,)
            ).fetchone()
            
            if existing:
                conn.execute("""
                    UPDATE server_registry 
                    SET server_name = ?, hardware_profile = ?, data_mode = ?, status = 'active'
                    WHERE server_id = ?
                """, (server_name, json.dumps(hardware_profile), data_mode, server_id))
                print(f"[REGISTRY] Updated server: {server_id} ({server_name})")
            else:
                conn.execute("""
                    INSERT INTO server_registry 
                    (server_id, server_name, hardware_profile, registered_at, data_mode)
                    VALUES (?, ?, ?, ?, ?)
                """, (server_id, server_name, json.dumps(hardware_profile), now, data_mode))
                print(f"[REGISTRY] Registered new server: {server_id} ({server_name})")
            
            conn.commit()
        return True
    
    def get_server(self, server_id: str) -> Optional[Dict]:
        """Get server details by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM server_registry WHERE server_id = ?", (server_id,)
            ).fetchone()
            
            if row:
                result = dict(row)
                result['hardware_profile'] = json.loads(result.get('hardware_profile', '{}'))
                return result
        return None
    
    def get_all_servers(self) -> List[Dict]:
        """Get all registered servers."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM server_registry WHERE status = 'active' ORDER BY registered_at"
            ).fetchall()
            
            results = []
            for row in rows:
                r = dict(row)
                r['hardware_profile'] = json.loads(r.get('hardware_profile', '{}'))
                results.append(r)
            return results
    
    def update_data_days(self, server_id: str, data_days: int):
        """Update the total data days count for a server."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE server_registry SET total_data_days = ? WHERE server_id = ?",
                (data_days, server_id)
            )
            conn.commit()
    
    # --- Model Management ---
    
    def register_model(self, server_id: str, model_type: str,
                       model_path: str, scaler_path: str = None,
                       tflite_path: str = None,
                       training_samples: int = 0, training_days: int = 0,
                       val_accuracy: float = None, val_mae: float = None,
                       notes: str = None) -> int:
        """
        Register a newly trained model.
        
        Returns:
            int: The version number assigned to this model.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_conn() as conn:
            # Get the next version number for this server + model_type
            row = conn.execute("""
                SELECT COALESCE(MAX(version), 0) + 1 as next_version
                FROM model_registry
                WHERE server_id = ? AND model_type = ?
            """, (server_id, model_type)).fetchone()
            
            version = row['next_version']
            
            # Mark all previous versions as superseded
            conn.execute("""
                UPDATE model_registry 
                SET status = 'superseded'
                WHERE server_id = ? AND model_type = ? AND status = 'active'
            """, (server_id, model_type))
            
            # Insert new model
            conn.execute("""
                INSERT INTO model_registry
                (server_id, model_type, version, trained_at, training_samples,
                 training_days, val_accuracy, val_mae, model_path, scaler_path,
                 tflite_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (server_id, model_type, version, now, training_samples,
                  training_days, val_accuracy, val_mae, model_path, scaler_path,
                  tflite_path, notes))
            
            # Update server's active model version
            conn.execute("""
                UPDATE server_registry 
                SET active_model_version = ?, last_retrain_at = ?
                WHERE server_id = ?
            """, (version, now, server_id))
            
            conn.commit()
        
        print(f"[REGISTRY] Model registered: {model_type} v{version} for {server_id}")
        return version
    
    def get_active_model(self, server_id: str, model_type: str) -> Optional[Dict]:
        """Get the currently active model for a server."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM model_registry
                WHERE server_id = ? AND model_type = ? AND status = 'active'
                ORDER BY version DESC LIMIT 1
            """, (server_id, model_type)).fetchone()
            
            return dict(row) if row else None
    
    def mark_deployed(self, model_id: int):
        """Mark a model as deployed to the edge."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE model_registry 
                SET deployed_to_edge = 1, deployed_at = ?
                WHERE id = ?
            """, (now, model_id))
            conn.commit()
    
    # --- Retraining Logs ---
    
    def log_retrain_start(self, server_id: str, reason: str,
                          old_version: int = None) -> int:
        """Log the start of a retraining job. Returns the log entry ID."""
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO retraining_log
                (server_id, triggered_at, trigger_reason, status, old_model_version)
                VALUES (?, ?, ?, 'running', ?)
            """, (server_id, now, reason, old_version))
            conn.commit()
            return cursor.lastrowid
    
    def log_retrain_complete(self, log_id: int, new_version: int,
                             improvement_pct: float = None):
        """Log the completion of a retraining job."""
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE retraining_log
                SET completed_at = ?, status = 'completed', 
                    new_model_version = ?, improvement_pct = ?
                WHERE id = ?
            """, (now, new_version, improvement_pct, log_id))
            conn.commit()
    
    def log_retrain_failed(self, log_id: int, error: str):
        """Log a failed retraining job."""
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE retraining_log
                SET completed_at = ?, status = 'failed', error_log = ?
                WHERE id = ?
            """, (now, error, log_id))
            conn.commit()
    
    def get_retrain_history(self, server_id: str, limit: int = 10) -> List[Dict]:
        """Get recent retraining history for a server."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM retraining_log
                WHERE server_id = ?
                ORDER BY triggered_at DESC
                LIMIT ?
            """, (server_id, limit)).fetchall()
            return [dict(r) for r in rows]
    
    def needs_retraining(self, server_id: str, min_data_days: int = 30,
                         retrain_interval_days: int = 30) -> tuple:
        """
        Check if a server needs model retraining.
        
        Returns:
            tuple: (needs_retrain: bool, reason: str)
        """
        server = self.get_server(server_id)
        if not server:
            return False, "server_not_found"
        
        data_days = server.get('total_data_days', 0)
        last_retrain = server.get('last_retrain_at')
        
        # Check 1: Enough data accumulated?
        if data_days < min_data_days:
            return False, f"insufficient_data ({data_days}/{min_data_days} days)"
        
        # Check 2: Never retrained?
        if last_retrain is None:
            return True, "first_retraining"
        
        # Check 3: Enough time since last retrain?
        last_dt = datetime.fromisoformat(last_retrain)
        days_since = (datetime.now(timezone.utc) - last_dt).days
        
        if days_since >= retrain_interval_days:
            return True, f"scheduled ({days_since} days since last retrain)"
        
        return False, f"not_due ({days_since}/{retrain_interval_days} days)"


if __name__ == "__main__":
    # Quick test
    registry = ModelRegistry()
    
    # Register a test server
    registry.register_server(
        "test-server-001", "LOQ-Test",
        {"cpu": "Ryzen 7", "gpu": "RTX 4060", "ram_gb": 24},
        data_mode="demo"
    )
    
    # Register a test model
    version = registry.register_model(
        "test-server-001", "classifier",
        model_path="server/models/classifier_fp32.h5",
        scaler_path="server/models/classifier_scaler.pkl",
        val_accuracy=0.9907, val_mae=0.0595,
        notes="Base model trained on AI4I 2020"
    )
    
    print(f"\nRegistered model version: {version}")
    print(f"Active model: {registry.get_active_model('test-server-001', 'classifier')}")
    print(f"All servers: {registry.get_all_servers()}")
    
    needs, reason = registry.needs_retraining("test-server-001")
    print(f"Needs retraining: {needs} ({reason})")
