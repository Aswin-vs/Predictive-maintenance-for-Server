"""
Edge MQTT Handler v2 — Phase G1.6

Subscribes to server telemetry and registration topics.
Maintains a server registry cache for multi-patient support.
Handles both v1 (flat) and v2 (hierarchical) payloads.

Topics subscribed:
  - server/+/telemetry     (multi-tenant telemetry)
  - server/+/register      (hardware registration)
  - server/telemetry        (legacy v1 topic)

Author: @claude-architect (Phase G)
"""

import os
import json
import time
import argparse
from typing import Callable, Any, Dict, Optional
from edge.edge_logger import get_logger

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

logger = get_logger("mqtt_handler")

DEFAULT_BROKER = os.environ.get("EDGE_BROKER_HOST", os.environ.get("MQTT_BROKER", "127.0.0.1"))
DEFAULT_PORT = int(os.environ.get("EDGE_BROKER_PORT", os.environ.get("MQTT_PORT", "1883")))
LEGACY_TOPIC = "server/telemetry"
MULTI_TENANT_TOPIC = "server/+/telemetry"
REGISTRATION_TOPIC = "server/+/register"


class EdgeMQTTHandler:
    def __init__(self, broker: str = DEFAULT_BROKER, port: int = DEFAULT_PORT):
        self.broker = broker
        self.port = port
        self.client = None
        self.on_telemetry_callback = None
        self.on_registration_callback = None

        # Server registry cache: server_id -> {topology, last_seen, name}
        self.server_registry: Dict[str, Dict] = {}

        if mqtt is not None:
            try:
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            except AttributeError:
                self.client = mqtt.Client()

            self.client.reconnect_delay_set(min_delay=1, max_delay=60)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
        else:
            logger.info("paho-mqtt not installed; EdgeMQTTHandler in offline mode")

    def set_telemetry_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set the function to call when telemetry is received."""
        self.on_telemetry_callback = callback

    def set_registration_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set the function to call when a server registers."""
        self.on_registration_callback = callback

    def get_server_topology(self, server_id: str) -> Optional[Dict]:
        """Get the cached hardware topology for a registered server."""
        entry = self.server_registry.get(server_id)
        return entry.get("topology") if entry else None

    def get_registered_servers(self) -> Dict[str, Dict]:
        """Get all registered servers and their metadata."""
        return dict(self.server_registry)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.broker}")
            # Subscribe to all topics
            self.client.subscribe(LEGACY_TOPIC)
            self.client.subscribe(MULTI_TENANT_TOPIC)
            self.client.subscribe(REGISTRATION_TOPIC)
            logger.info(f"Subscribed: {LEGACY_TOPIC}, {MULTI_TENANT_TOPIC}, {REGISTRATION_TOPIC}")
        else:
            logger.error(f"Failed to connect, return code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnect (rc={rc}). Auto-reconnecting...")

    def _on_message(self, client, userdata, msg):
        try:
            payload_str = msg.payload.decode('utf-8')
            data = json.loads(payload_str)
            topic = msg.topic

            # ── Registration Message ──
            if topic.endswith("/register"):
                self._handle_registration(topic, data)
                return

            # ── Telemetry Message ──
            is_telemetry = (
                topic == LEGACY_TOPIC or
                (topic.startswith("server/") and topic.endswith("/telemetry"))
            )

            if is_telemetry and self.on_telemetry_callback:
                # Extract server_id from topic or payload
                server_id = data.get("server_id")
                if not server_id and "/" in topic:
                    parts = topic.split("/")
                    if len(parts) == 3:
                        server_id = parts[1]

                if server_id:
                    data["server_id"] = server_id

                    # Update last_seen in registry
                    if server_id in self.server_registry:
                        self.server_registry[server_id]["last_seen"] = time.time()

                self.on_telemetry_callback(data)

        except json.JSONDecodeError:
            logger.warning(f"Malformed JSON received on {msg.topic}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _handle_registration(self, topic: str, data: Dict[str, Any]):
        """Handle a server registration message."""
        server_id = data.get("server_id", "unknown")
        server_name = data.get("server_name", server_id)
        topology = data.get("hardware_topology", {})

        self.server_registry[server_id] = {
            "server_name": server_name,
            "topology": topology,
            "data_mode": data.get("data_mode", "production"),
            "schema_version": data.get("schema_version", "1.0"),
            "registered_at": data.get("registration_time", ""),
            "last_seen": time.time(),
        }

        # Log the registration
        hw_summary = self._topology_summary(topology)
        logger.info(f"[REGISTER] Server '{server_name}' ({server_id[:12]}...) registered: {hw_summary}")
        logger.info(f"[REGISTER] Total registered servers: {len(self.server_registry)}")

        # Fire callback if set
        if self.on_registration_callback:
            self.on_registration_callback(data)

    def _topology_summary(self, topology: Dict) -> str:
        """Generate a brief human-readable summary of a hardware topology."""
        if not topology:
            return "No topology data"
        cpus = topology.get("cpus", [])
        gpus = topology.get("gpus", [])
        mem = topology.get("memory", [])
        storage = topology.get("storage", [])

        cpu_str = cpus[0].get("model", "?")[:30] if cpus else "?"
        gpu_str = gpus[0].get("model", "None")[:25] if gpus else "None"
        ram_gb = sum(m.get("size_gb", 0) for m in mem)
        drives = len(storage)

        return f"CPU: {cpu_str} | GPU: {gpu_str} | RAM: {ram_gb}GB | Drives: {drives}"

    def start(self):
        """Start the MQTT loop in a non-blocking background thread."""
        if self.client is None:
            logger.info("MQTT client not initialized; skipping start.")
            return
        logger.info(f"Connecting to MQTT broker: {self.broker}:{self.port}...")
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            logger.info("MQTT background loop started")
        except Exception as e:
            logger.error(f"Could not connect to MQTT broker ({e}). Telemetry pipeline will wait.")

    def publish(self, topic: str, payload: Any, qos: int = 0) -> bool:
        """Publish a message to an MQTT topic."""
        if self.client is None:
            return False
        try:
            if isinstance(payload, (dict, list)):
                payload_str = json.dumps(payload)
            elif isinstance(payload, str):
                payload_str = payload
            else:
                payload_str = str(payload)
            info = self.client.publish(topic, payload_str, qos=qos)
            return True
        except Exception as e:
            logger.warning(f"MQTT publish to {topic} failed: {e}")
            return False

    def stop(self):
        """Gracefully disconnect and stop the MQTT loop."""
        if self.client is not None:
            logger.info("Stopping MQTT client...")
            self.client.loop_stop()
            self.client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge MQTT Subscriber v2")
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER, help="MQTT Broker IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT Broker Port")
    args = parser.parse_args()

    def sample_telemetry_cb(data):
        sid = data.get('server_id', 'unknown')[:12]
        ts = data.get('timestamp', data.get('timestamp_utc', '?'))
        print(f"[TELEMETRY] Server {sid}... at {ts}")

    def sample_register_cb(data):
        print(f"[REGISTER] New server: {data.get('server_name', '?')} ({data.get('server_id', '?')[:12]}...)")

    handler = EdgeMQTTHandler(args.broker, args.port)
    handler.set_telemetry_callback(sample_telemetry_cb)
    handler.set_registration_callback(sample_register_cb)

    try:
        handler.start()
        print("Waiting for messages... (Press Ctrl+C to stop)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        handler.stop()


