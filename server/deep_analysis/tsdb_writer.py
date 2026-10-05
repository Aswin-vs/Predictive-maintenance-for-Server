"""
MQTT → TimescaleDB Ingestion Service — Phase G2.3

Subscribes to the RPi4 Mosquitto broker, receives all telemetry
and registration messages, and writes them to TimescaleDB.

Runs on LOQ (Hospital/Cloud Node) alongside inference_server.py.

Usage:
    python server/deep_analysis/tsdb_writer.py
    python server/deep_analysis/tsdb_writer.py --broker rpi4-edge.tail1ab139.ts.net --db-host localhost
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Auto-load configs/network.env if present
_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_env_path = os.path.join(_base, "configs", "network.env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"\'')
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

import paho.mqtt.client as mqtt

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("[WARN] psycopg2 not installed. Run: pip install psycopg2-binary")

# ─── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("tsdb_writer")

# ─── Configuration ─────────────────────────────────────────────
DEFAULT_MQTT_BROKER = os.getenv("EDGE_BROKER_HOST", os.getenv("MQTT_BROKER", "localhost"))
DEFAULT_MQTT_PORT = int(os.getenv("EDGE_BROKER_PORT", os.getenv("MQTT_PORT", "1883")))
DEFAULT_DB_HOST = os.getenv("TSDB_HOST", "localhost")
DEFAULT_DB_PORT = int(os.getenv("TSDB_PORT", "5432"))
DEFAULT_DB_NAME = os.getenv("TSDB_DB", "predictive_maintenance")
DEFAULT_DB_USER = os.getenv("TSDB_USER", "pm_user")
DEFAULT_DB_PASS = os.getenv("TSDB_PASS", "pm_edge_2026")

# Batch insert settings
BATCH_SIZE = 50         # Flush after N messages
FLUSH_INTERVAL = 5.0    # Flush every N seconds regardless


class TimescaleDBWriter:
    """Manages connection pool and batch inserts to TimescaleDB."""

    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        self.conn_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }
        self.conn: Optional[psycopg2.extensions.connection] = None
        self._connect()

    def _connect(self):
        """Establish connection to TimescaleDB."""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            self.conn.autocommit = False
            log.info(f"Connected to TimescaleDB at {self.conn_params['host']}:{self.conn_params['port']}")
        except Exception as e:
            log.error(f"Failed to connect to TimescaleDB: {e}")
            self.conn = None

    def _ensure_connected(self) -> bool:
        """Reconnect if connection is lost."""
        if self.conn is None or self.conn.closed:
            log.warning("TimescaleDB connection lost, reconnecting...")
            self._connect()
        return self.conn is not None and not self.conn.closed

    def upsert_server(self, server_id: str, payload: Dict[str, Any]):
        """Insert or update server registry entry."""
        if not self._ensure_connected():
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO servers (server_id, server_name, hostname, os, arch, topology, last_seen_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (server_id) DO UPDATE SET
                        server_name = EXCLUDED.server_name,
                        hostname = EXCLUDED.hostname,
                        os = EXCLUDED.os,
                        arch = EXCLUDED.arch,
                        topology = EXCLUDED.topology,
                        last_seen_at = NOW()
                """, (
                    server_id,
                    payload.get("server_name", ""),
                    payload.get("hostname", ""),
                    payload.get("os", ""),
                    payload.get("arch", ""),
                    json.dumps(payload.get("topology", payload)),
                ))
            self.conn.commit()
            log.info(f"[REGISTER] Server '{server_id}' registered/updated")
        except Exception as e:
            log.error(f"[REGISTER] Failed to upsert server '{server_id}': {e}")
            self.conn.rollback()

    def insert_telemetry_batch(self, rows: List[tuple]):
        """Batch insert telemetry rows."""
        if not rows or not self._ensure_connected():
            return

        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO telemetry_raw
                        (time, server_id, schema_version, payload,
                         cpu_temp, cpu_load, gpu_temp, gpu_power,
                         ram_used_gb, ram_total_gb, disk_temp,
                         health_score, fault_class, confidence)
                    VALUES %s
                    """,
                    rows,
                    template=(
                        "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    ),
                    page_size=100
                )
            self.conn.commit()
            log.info(f"[TELEMETRY] Inserted {len(rows)} rows")
        except Exception as e:
            log.error(f"[TELEMETRY] Batch insert failed ({len(rows)} rows): {e}")
            self.conn.rollback()

    def insert_prediction(self, server_id: str, source: str, model_name: str,
                          fault_class: int, fault_label: str, confidence: float,
                          ttf_hours: float, anomaly_score: float,
                          inference_ms: float, shap_data: dict):
        """Insert a single prediction row."""
        if not self._ensure_connected():
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO predictions
                        (time, server_id, source, model_name, fault_class, fault_label,
                         confidence, ttf_hours, anomaly_score, inference_ms, shap_data)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    server_id, source, model_name, fault_class, fault_label,
                    confidence, ttf_hours, anomaly_score, inference_ms,
                    json.dumps(shap_data) if shap_data else None,
                ))
            self.conn.commit()
        except Exception as e:
            log.error(f"[PREDICTION] Insert failed: {e}")
            self.conn.rollback()

    def is_connected(self) -> bool:
        """Check if TimescaleDB connection is active."""
        return self._ensure_connected()

    def query_data_days(self, server_id: str) -> int:
        """Query span of data in days accumulated for a server."""
        if not self._ensure_connected():
            return 0
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(EXTRACT(DAY FROM (MAX(time) - MIN(time))), 0)
                    FROM telemetry_raw
                    WHERE server_id = %s
                """, (server_id,))
                res = cur.fetchone()
                return int(res[0]) if res and res[0] is not None else 0
        except Exception as e:
            log.error(f"[QUERY] Failed to query data days for '{server_id}': {e}")
            if self.conn:
                self.conn.rollback()
            return 0

    def query_training_data(self, server_id: str, days: int = 30) -> Optional[Any]:
        """Query raw telemetry data for training."""
        if not self._ensure_connected():
            return None
        try:
            import pandas as pd
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT time, server_id, cpu_temp, cpu_load, gpu_temp, gpu_power,
                           ram_used_gb, ram_total_gb, disk_temp, health_score, fault_class, confidence
                    FROM telemetry_raw
                    WHERE server_id = %s AND time >= NOW() - (%s || ' days')::INTERVAL
                    ORDER BY time ASC
                """, (server_id, str(days)))
                rows = cur.fetchall()
                if not rows:
                    return None
                colnames = [desc[0] for desc in cur.description]
                return pd.DataFrame(rows, columns=colnames)
        except Exception as e:
            log.error(f"[QUERY] Failed to query training data for '{server_id}': {e}")
            if self.conn:
                self.conn.rollback()
            return None

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            log.info("TimescaleDB connection closed")


class MQTTIngestionService:
    """Subscribes to MQTT and batches writes to TimescaleDB."""

    def __init__(self, broker: str, port: int, db_writer: TimescaleDBWriter):
        self.broker = broker
        self.port = port
        self.db = db_writer
        self._buffer: List[tuple] = []
        self._buffer_lock = threading.Lock()
        self._last_flush = time.time()
        self._running = True
        self._stats = {
            "telemetry_received": 0,
            "telemetry_written": 0,
            "registrations": 0,
            "errors": 0,
        }

        # MQTT client setup
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except (AttributeError, TypeError):
            try:
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            except (AttributeError, TypeError):
                self.client = mqtt.Client()

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc, *args):
        """Subscribe to all relevant topics on connect."""
        log.info(f"Connected to MQTT broker at {self.broker}:{self.port}")

        # Subscribe to all telemetry and registration topics
        topics = [
            ("server/+/telemetry", 1),   # v2 per-server telemetry
            ("server/+/register", 1),     # v2 registration
            ("server/telemetry", 1),      # v1 legacy telemetry
        ]
        for topic, qos in topics:
            client.subscribe(topic, qos)
        log.info(f"Subscribed to: {[t[0] for t in topics]}")

    def _on_disconnect(self, client, userdata, rc, *args):
        """Handle disconnection — auto-reconnect is handled by paho."""
        if rc != 0:
            log.warning(f"MQTT disconnected unexpectedly (rc={rc}), will auto-reconnect")

    def _on_message(self, client, userdata, msg):
        """Route incoming messages to appropriate handlers."""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode("utf-8", errors="replace"))

            if "/register" in topic:
                self._handle_registration(topic, payload)
            elif "/telemetry" in topic or topic == "server/telemetry":
                self._handle_telemetry(topic, payload)

        except json.JSONDecodeError as e:
            log.warning(f"Invalid JSON on {msg.topic}: {e}")
            self._stats["errors"] += 1
        except Exception as e:
            log.error(f"Error processing message on {msg.topic}: {e}")
            self._stats["errors"] += 1

    def _handle_registration(self, topic: str, payload: Dict[str, Any]):
        """Process server registration messages."""
        # Extract server_id from topic: server/{id}/register
        parts = topic.split("/")
        server_id = parts[1] if len(parts) >= 3 else payload.get("server_id", "unknown")

        self.db.upsert_server(server_id, payload)
        self._stats["registrations"] += 1

    def _handle_telemetry(self, topic: str, payload: Dict[str, Any]):
        """Buffer telemetry for batch insertion."""
        self._stats["telemetry_received"] += 1

        # Extract server_id
        parts = topic.split("/")
        if len(parts) >= 3 and parts[1] != "telemetry":
            server_id = parts[1]
        else:
            server_id = payload.get("server_id", "unknown")

        # Determine schema version
        schema_version = str(payload.get("schema_version", "1.0"))

        # Parse timestamp
        ts_str = payload.get("timestamp_utc") or payload.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        # Extract key metrics for indexed columns
        cpu_temp = self._extract_cpu_temp(payload)
        cpu_load = self._extract_cpu_load(payload)
        gpu_temp = self._extract_gpu_temp(payload)
        gpu_power = self._extract_gpu_power(payload)
        ram_used = self._extract_ram_used(payload)
        ram_total = self._extract_ram_total(payload)
        disk_temp = self._extract_disk_temp(payload)

        row = (
            ts, server_id, schema_version, json.dumps(payload),
            cpu_temp, cpu_load, gpu_temp, gpu_power,
            ram_used, ram_total, disk_temp,
            None, None, None  # health_score, fault_class, confidence (filled by inference)
        )

        with self._buffer_lock:
            self._buffer.append(row)

        # Flush if buffer is full
        if len(self._buffer) >= BATCH_SIZE:
            self._flush_buffer()

    def _extract_cpu_temp(self, p: dict) -> Optional[float]:
        """Extract CPU temperature from v1 or v2 payload."""
        # v2: nested under cpus[0].temp_package or temp_core_c
        cpus = p.get("cpus") or p.get("cpu")
        if isinstance(cpus, list) and cpus:
            return cpus[0].get("temp_package") or cpus[0].get("temp_core_c") or cpus[0].get("temp")
        if isinstance(cpus, dict):
            return cpus.get("temp_package") or cpus.get("temp_core_c") or cpus.get("temp")
        # v1: flat
        return p.get("cpu_temp")

    def _extract_cpu_load(self, p: dict) -> Optional[float]:
        cpus = p.get("cpus") or p.get("cpu")
        if isinstance(cpus, list) and cpus:
            return cpus[0].get("load_percent")
        if isinstance(cpus, dict):
            return cpus.get("load_percent") or cpus.get("load")
        return p.get("cpu_load")

    def _extract_gpu_temp(self, p: dict) -> Optional[float]:
        """Extract GPU temperature from v1 or v2 payload."""
        # F-027 FIX: v2 schema uses temp_core_c for GPU, not temp_package
        gpus = p.get("gpus") or p.get("gpu")
        if isinstance(gpus, list) and gpus:
            return gpus[0].get("temp_core_c") or gpus[0].get("temp")
        if isinstance(gpus, dict):
            return gpus.get("temp_core_c") or gpus.get("temp")
        return p.get("gpu_temp")

    def _extract_gpu_power(self, p: dict) -> Optional[float]:
        gpus = p.get("gpus") or p.get("gpu")
        if isinstance(gpus, list) and gpus:
            return gpus[0].get("power_draw_w")
        if isinstance(gpus, dict):
            return gpus.get("power_draw_w")
        return p.get("gpu_power")

    def _extract_ram_used(self, p: dict) -> Optional[float]:
        mem = p.get("memory") or p.get("ram")
        if isinstance(mem, list) and mem:
            return mem[0].get("used_gb")
        if isinstance(mem, dict):
            return mem.get("used_gb")
        return p.get("ram_used_gb")

    def _extract_ram_total(self, p: dict) -> Optional[float]:
        mem = p.get("memory") or p.get("ram")
        if isinstance(mem, list) and mem:
            return mem[0].get("total_gb")
        if isinstance(mem, dict):
            return mem.get("total_gb")
        return p.get("ram_total_gb")

    def _extract_disk_temp(self, p: dict) -> Optional[float]:
        storage = p.get("storage") or p.get("hdd")
        if isinstance(storage, list) and storage:
            return storage[0].get("temperature_c") or storage[0].get("smart_194_raw")
        if isinstance(storage, dict):
            return storage.get("smart_194_raw")
        return p.get("smart_194_raw")

    def _flush_buffer(self):
        """Flush buffered telemetry rows to TimescaleDB."""
        with self._buffer_lock:
            if not self._buffer:
                return
            rows = self._buffer.copy()
            self._buffer.clear()

        self.db.insert_telemetry_batch(rows)
        self._stats["telemetry_written"] += len(rows)
        self._last_flush = time.time()

    def _flush_loop(self):
        """Background thread that flushes buffer periodically."""
        while self._running:
            time.sleep(1.0)
            elapsed = time.time() - self._last_flush
            if elapsed >= FLUSH_INTERVAL and self._buffer:
                self._flush_buffer()

    def start(self):
        """Connect and start the ingestion loop."""
        log.info("=" * 60)
        log.info("  MQTT → TimescaleDB Ingestion Service")
        log.info("=" * 60)
        log.info(f"  MQTT Broker:  {self.broker}:{self.port}")
        log.info(f"  Database:     {self.db.conn_params['host']}:{self.db.conn_params['port']}/{self.db.conn_params['dbname']}")
        log.info(f"  Batch Size:   {BATCH_SIZE} | Flush Interval: {FLUSH_INTERVAL}s")
        log.info("=" * 60)

        # Start flush thread
        flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        flush_thread.start()

        # Connect to MQTT
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_forever()

    def stop(self):
        """Graceful shutdown."""
        log.info("Shutting down...")
        self._running = False
        self._flush_buffer()  # Flush remaining
        self.client.disconnect()
        self.db.close()

        log.info("--- Session Statistics ---")
        for key, val in self._stats.items():
            log.info(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(description="MQTT -> TimescaleDB Ingestion Service")
    parser.add_argument("--broker", default=DEFAULT_MQTT_BROKER, help="MQTT broker address")
    parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT, help="MQTT port")
    parser.add_argument("--db-host", default=DEFAULT_DB_HOST, help="TimescaleDB host")
    parser.add_argument("--db-port", type=int, default=DEFAULT_DB_PORT, help="TimescaleDB port")
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME, help="Database name")
    parser.add_argument("--db-user", default=DEFAULT_DB_USER, help="Database user")
    parser.add_argument("--db-pass", default=DEFAULT_DB_PASS, help="Database password")
    args = parser.parse_args()

    if not PSYCOPG2_AVAILABLE:
        log.error("psycopg2 is required. Install: pip install psycopg2-binary")
        sys.exit(1)

    # Create DB writer
    db_writer = TimescaleDBWriter(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_pass,
    )

    # Create ingestion service
    service = MQTTIngestionService(
        broker=args.broker,
        port=args.mqtt_port,
        db_writer=db_writer,
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start service
    service.start()


if __name__ == "__main__":
    main()
