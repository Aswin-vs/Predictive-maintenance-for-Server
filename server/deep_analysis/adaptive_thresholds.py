"""
G5.1: Adaptive Threshold Engine
=================================
Computes percentile-based thresholds from historical telemetry data.
Updates the `thresholds` table in TimescaleDB per server, per metric.

Thresholds auto-adjust based on the actual distribution of each metric:
  - warning = P85 (85th percentile of recent data)
  - critical = P97 (97th percentile of recent data)
  - p95/p99 for reference

Can be run as:
  1. Standalone CLI: `python adaptive_thresholds.py`
  2. Imported by inference_server or scheduler
  3. Cron job (every hour or daily)
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import psycopg2
import numpy as np
from server.utils.logger import get_logger

logger = get_logger("adaptive_thresholds")

# ─── Default Static Thresholds (fallback when no historical data) ─────────────
STATIC_DEFAULTS = {
    "cpu_temp":         {"warning": 80.0,  "critical": 95.0},
    "cpu_usage":        {"warning": 85.0,  "critical": 98.0},
    "gpu_temp":         {"warning": 85.0,  "critical": 100.0},
    "gpu_usage":        {"warning": 90.0,  "critical": 99.0},
    "ram_usage":        {"warning": 85.0,  "critical": 95.0},
    "disk_usage":       {"warning": 80.0,  "critical": 95.0},
    "fan_rpm":          {"warning": 1000.0, "critical": 500.0},  # Below threshold = warning
    "psu_voltage":      {"warning": 11.4,  "critical": 10.8},   # Below threshold = warning
    "smart_5_raw":      {"warning": 5.0,   "critical": 50.0},
    "smart_187_raw":    {"warning": 10.0,  "critical": 100.0},
    "smart_197_raw":    {"warning": 1.0,   "critical": 10.0},
    "smart_198_raw":    {"warning": 1.0,   "critical": 10.0},
    "vibration_x":      {"warning": 0.5,   "critical": 1.5},
    "vibration_y":      {"warning": 0.5,   "critical": 1.5},
    "vibration_z":      {"warning": 2.0,   "critical": 4.0},
    "ambient_temp":     {"warning": 35.0,  "critical": 45.0},
    "sound_level":      {"warning": 0.7,   "critical": 0.9},
}

# Metrics where LOWER is bad (inverse thresholds)
INVERSE_METRICS = {"fan_rpm", "psu_voltage"}

# ─── Connection ──────────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(
        host=os.getenv("TSDB_HOST", "localhost"),
        port=int(os.getenv("TSDB_PORT", "5432")),
        dbname=os.getenv("TSDB_DB", "predictive_maintenance"),
        user=os.getenv("TSDB_USER", "pm_user"),
        password=os.getenv("TSDB_PASS", "pm_edge_2026"),
        connect_timeout=5,
    )


# ─── Core Engine ─────────────────────────────────────────────────────────────

class AdaptiveThresholdEngine:
    """
    Computes and manages adaptive thresholds from TimescaleDB telemetry data.
    
    Strategy:
      1. Pull last N hours of telemetry for a server
      2. Compute percentile distribution per metric
      3. Set warning=P85, critical=P97 (or inverse for fan_rpm/psu_voltage)
      4. Blend with static defaults (70% data-driven, 30% static) for safety
      5. Upsert into thresholds table
    """
    
    def __init__(self, lookback_hours: int = 168, blend_ratio: float = 0.7):
        """
        Args:
            lookback_hours: How many hours of historical data to consider (default: 168 = 7 days)
            blend_ratio: Weight given to data-driven thresholds (0-1). 
                        0.7 = 70% data-driven, 30% static defaults.
        """
        self.lookback_hours = lookback_hours
        self.blend_ratio = blend_ratio
        self.conn = None
    
    def connect(self):
        try:
            self.conn = _get_conn()
            self.conn.autocommit = True
            return True
        except Exception as e:
            logger.error(f"DB connection failed: {e}")
            return False
    
    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
    
    def get_server_ids(self) -> List[str]:
        """Get all known server IDs from telemetry data."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT server_id FROM telemetry_raw ORDER BY server_id")
            return [r[0] for r in cur.fetchall()]
    
    def get_metric_stats(self, server_id: str) -> Dict[str, Dict]:
        """
        Compute percentile stats for all metrics of a server.
        Returns: {metric_name: {p50, p85, p95, p97, p99, mean, std, count}}
        """
        cutoff = datetime.utcnow() - timedelta(hours=self.lookback_hours)
        
        with self.conn.cursor() as cur:
            # Get dedicated columns + JSONB payload
            cur.execute("""
                SELECT cpu_temp, cpu_load, gpu_temp, gpu_power, 
                       ram_used_gb, ram_total_gb, disk_temp,
                       health_score, confidence, payload
                FROM telemetry_raw 
                WHERE server_id = %s AND time >= %s
                ORDER BY time DESC LIMIT 10000
            """, (server_id, cutoff))
            rows = cur.fetchall()
        
        if not rows:
            logger.info(f"[{server_id}] No telemetry data found")
            return {}
        
        # Collect all numeric values per metric
        metric_values = {}
        col_names = ['cpu_temp', 'cpu_load', 'gpu_temp', 'gpu_power', 
                     'ram_used_gb', 'ram_total_gb', 'disk_temp',
                     'health_score', 'confidence']
        
        for row in rows:
            # Dedicated columns
            for i, col in enumerate(col_names):
                val = row[i]
                if val is not None and not np.isnan(val):
                    if col not in metric_values:
                        metric_values[col] = []
                    metric_values[col].append(float(val))
            
            # JSONB payload (contains all original telemetry)
            payload = row[-1]
            if payload:
                if isinstance(payload, str):
                    import json as _json
                    payload = _json.loads(payload)
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            if key not in metric_values:
                                metric_values[key] = []
                            metric_values[key].append(float(value))
        
        # Compute percentiles
        stats = {}
        for metric, values in metric_values.items():
            if len(values) < 5:  # Need minimum data points
                continue
            arr = np.array(values)
            stats[metric] = {
                "p50": float(np.percentile(arr, 50)),
                "p85": float(np.percentile(arr, 85)),
                "p95": float(np.percentile(arr, 95)),
                "p97": float(np.percentile(arr, 97)),
                "p99": float(np.percentile(arr, 99)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "count": len(values),
            }
        
        return stats
    
    def compute_thresholds(self, server_id: str) -> Dict[str, Dict]:
        """
        Compute adaptive thresholds for a server by blending 
        percentile-based thresholds with static defaults.
        """
        stats = self.get_metric_stats(server_id)
        thresholds = {}
        
        for metric_name, defaults in STATIC_DEFAULTS.items():
            data = stats.get(metric_name, None)
            
            if data and data["count"] >= 10:
                # Data-driven thresholds
                if metric_name in INVERSE_METRICS:
                    # For inverse metrics (lower=bad), use lower percentiles
                    data_warning = data["p50"] - 1.5 * data["std"]
                    data_critical = data["p50"] - 3.0 * data["std"]
                else:
                    # Normal metrics (higher=bad)
                    data_warning = data["p85"]
                    data_critical = data["p97"]
                
                # Blend with static defaults
                r = self.blend_ratio
                warning = r * data_warning + (1 - r) * defaults["warning"]
                critical = r * data_critical + (1 - r) * defaults["critical"]
                
                thresholds[metric_name] = {
                    "warning": round(warning, 2),
                    "critical": round(critical, 2),
                    "p95": round(data["p95"], 2),
                    "p99": round(data["p99"], 2),
                    "data_points": data["count"],
                    "source": "adaptive",
                }
            else:
                # Fall back to static defaults
                thresholds[metric_name] = {
                    "warning": defaults["warning"],
                    "critical": defaults["critical"],
                    "p95": None,
                    "p99": None,
                    "data_points": data["count"] if data else 0,
                    "source": "static",
                }
        
        # Include any metrics from data that aren't in STATIC_DEFAULTS
        for metric_name, data in stats.items():
            if metric_name not in thresholds and data["count"] >= 10:
                thresholds[metric_name] = {
                    "warning": round(data["p85"], 2),
                    "critical": round(data["p97"], 2),
                    "p95": round(data["p95"], 2),
                    "p99": round(data["p99"], 2),
                    "data_points": data["count"],
                    "source": "adaptive_new",
                }
        
        return thresholds
    
    def save_thresholds(self, server_id: str, thresholds: Dict[str, Dict]) -> int:
        """Upsert thresholds into TimescaleDB. Returns count of rows updated."""
        count = 0
        with self.conn.cursor() as cur:
            for metric, vals in thresholds.items():
                cur.execute("""
                    INSERT INTO thresholds (server_id, metric_name, warning_value, critical_value, 
                                           p95_value, p99_value, last_computed, manual_override)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), FALSE)
                    ON CONFLICT (server_id, metric_name)
                    DO UPDATE SET 
                        warning_value = EXCLUDED.warning_value,
                        critical_value = EXCLUDED.critical_value,
                        p95_value = EXCLUDED.p95_value,
                        p99_value = EXCLUDED.p99_value,
                        last_computed = NOW()
                    WHERE thresholds.manual_override = FALSE
                """, (
                    server_id, metric,
                    vals["warning"], vals["critical"],
                    vals.get("p95"), vals.get("p99"),
                ))
                count += 1
        return count
    
    def get_thresholds(self, server_id: str) -> Dict[str, Dict]:
        """Retrieve current thresholds from DB for a server."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT metric_name, warning_value, critical_value, p95_value, p99_value, 
                       last_computed, manual_override
                FROM thresholds WHERE server_id = %s
            """, (server_id,))
            rows = cur.fetchall()
        
        result = {}
        for name, warn, crit, p95, p99, computed, override in rows:
            result[name] = {
                "warning": warn, "critical": crit,
                "p95": p95, "p99": p99,
                "last_computed": computed.isoformat() if computed else None,
                "manual_override": override,
            }
        return result
    
    def run_full_update(self) -> Dict[str, int]:
        """Run adaptive threshold computation for all servers."""
        results = {}
        servers = self.get_server_ids()
        
        if not servers:
            # If no telemetry data, seed with defaults for a default server
            logger.info("No servers in telemetry — seeding defaults")
            defaults = {}
            for m, v in STATIC_DEFAULTS.items():
                defaults[m] = {"warning": v["warning"], "critical": v["critical"], 
                              "p95": None, "p99": None, "data_points": 0, "source": "static"}
            count = self.save_thresholds("default", defaults)
            results["default"] = count
        else:
            for sid in servers:
                thresholds = self.compute_thresholds(sid)
                count = self.save_thresholds(sid, thresholds)
                adaptive = sum(1 for v in thresholds.values() if v["source"] != "static")
                results[sid] = {"total": count, "adaptive": adaptive}
                logger.info(f"[{sid}] Updated {count} thresholds ({adaptive} adaptive)")
        
        return results


# ─── API Endpoint (for FastAPI integration) ──────────────────────────────────

def get_thresholds_for_server(server_id: str) -> Dict:
    """Quick lookup — used by inference_server and dashboard API."""
    try:
        engine = AdaptiveThresholdEngine()
        if not engine.connect():
            return {"error": "DB unavailable", "defaults": STATIC_DEFAULTS}
        result = engine.get_thresholds(server_id)
        engine.close()
        if not result:
            return STATIC_DEFAULTS
        return result
    except Exception as e:
        return {"error": str(e), "defaults": STATIC_DEFAULTS}


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  G5.1: Adaptive Threshold Engine")
    print("=" * 60)
    
    engine = AdaptiveThresholdEngine()
    if not engine.connect():
        print("ERROR: Cannot connect to TimescaleDB")
        sys.exit(1)
    
    print(f"\nLookback: {engine.lookback_hours}h | Blend: {engine.blend_ratio:.0%} data-driven")
    
    results = engine.run_full_update()
    
    print(f"\nResults:")
    for sid, info in results.items():
        if isinstance(info, dict):
            print(f"  [{sid}] {info.get('total', info)} thresholds ({info.get('adaptive', 0)} adaptive)")
        else:
            print(f"  [{sid}] {info} thresholds (static defaults)")
    
    # Show sample thresholds
    servers = engine.get_server_ids() or ["default"]
    for sid in servers[:1]:
        thresholds = engine.get_thresholds(sid)
        print(f"\nThresholds for [{sid}]:")
        for metric, vals in sorted(thresholds.items())[:10]:
            print(f"  {metric:20s} | warn={vals['warning']:>8.2f} | crit={vals['critical']:>8.2f}")
    
    engine.close()
    print(f"\n{'='*60}")
    print("  Adaptive Thresholds DONE")
    print(f"{'='*60}")
