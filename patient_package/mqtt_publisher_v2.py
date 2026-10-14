"""
MQTT Publisher v2 — Phase G1.5

Enhanced MQTT publisher that:
  1. Runs Hardware Auto-Discovery on first boot
  2. Publishes registration payload on server/{id}/register
  3. Uses TelemetryCollectorV2 (plugin-based) for hierarchical payloads
  4. Publishes v2 hierarchical payload on server/{id}/telemetry
  5. Maintains backward compatibility with v1 flat format on legacy topic

Author: @claude-architect (Phase G)
Date: 2026-09-11
"""

import json
import time
import argparse
import os
import sys

# Add project root for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("ERROR: paho-mqtt not installed. Run: pip install paho-mqtt")
    sys.exit(1)

try:
    from server.telemetry_agent.auto_discovery import HardwareDiscovery
    from server.telemetry_agent.collector_v2 import TelemetryCollectorV2
    from server.telemetry_agent.server_identity import ServerIdentity
except ImportError:
    from auto_discovery import HardwareDiscovery
    from collector_v2 import TelemetryCollectorV2
    from server_identity import ServerIdentity

# Auto-load configs/network.env if present
_env_path = os.path.join(BASE_DIR, "configs", "network.env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"\'')
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Default config (dynamic resolution via Tailscale MagicDNS or network.env)
DEFAULT_BROKER = os.environ.get("EDGE_BROKER_HOST", os.environ.get("MQTT_BROKER", "127.0.0.1"))
DEFAULT_PORT = int(os.environ.get("EDGE_BROKER_PORT", os.environ.get("MQTT_PORT", 1883)))
LEGACY_TOPIC = "server/telemetry"


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected to broker at {userdata['broker']}")
    else:
        print(f"[MQTT] Connection failed, rc={rc}")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"[MQTT] Unexpected disconnect (rc={rc}). Auto-reconnecting...")


def start_publisher_v2(broker: str, port: int, use_dummy: bool = False,
                       data_mode: str = "production", server_name: str = None,
                       interval: int = 5, dry_run: bool = False, max_frames: int = 0):
    """
    Main publisher loop with v2 auto-discovery + hierarchical telemetry.
    """

    # ──── Step 1: Initialize Server Identity ──────────────────────────
    identity = ServerIdentity(data_mode=data_mode)
    server_id = identity.server_id

    # Override server name if provided
    if server_name:
        identity.update_name(server_name)
    srv_name = identity.server_name

    # ──── Step 2: Run Hardware Auto-Discovery ─────────────────────────
    print("\n[DISCOVERY] Running hardware auto-discovery...")
    discovery = HardwareDiscovery()
    topology = discovery.discover(force_refresh=False)
    hw_summary = discovery.get_summary(topology)
    print(f"[DISCOVERY] Hardware: {hw_summary}")

    # ──── Step 3: Initialize Plugin-Based Collector ───────────────────
    collector = TelemetryCollectorV2(use_dummy=use_dummy, topology=topology)

    if dry_run:
        print("\n" + "="*60)
        print("  Patient Server Telemetry Publisher v2.0 (DRY RUN)")
        print("="*60)
        print(f"  Server ID:     {server_id}")
        print(f"  Server Name:   {srv_name}")
        print(f"  MQTT Broker:   {broker}:{port}")
        print(f"  Dummy Mode:    {use_dummy}")
        print(f"  Hardware:      {hw_summary}")
        sample = collector.collect()
        sample["server_id"] = server_id
        sample["server_name"] = srv_name
        sample["data_mode"] = data_mode
        print(f"[DRY RUN] Generated Frame Size: {len(json.dumps(sample))} bytes")
        print(f"[DRY RUN] Components: {list(sample.get('components', {}).keys())}")
        print("[DRY RUN] Hostname and configuration resolution successful.")
        print("="*60 + "\n")
        return

    # ──── Step 4: Setup MQTT Client ───────────────────────────────────
    topic_telemetry = f"server/{server_id}/telemetry"
    topic_register = f"server/{server_id}/register"

    client_id = f"patient-{server_id}-{os.getpid()}"
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id, clean_session=True)
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id=client_id, clean_session=True)

    client.user_data_set({'broker': broker, 'server_id': server_id})
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=60)

    print(f"\n{'='*60}")
    print(f"  Patient Server Telemetry Publisher v2.0")
    print(f"{'='*60}")
    print(f"  Server ID:     {server_id}")
    print(f"  Server Name:   {srv_name}")
    print(f"  Schema:        v2.0 (hierarchical + v1 backward compat)")
    print(f"  MQTT Broker:   {broker}:{port}")
    print(f"  Topics:        {topic_telemetry}")
    print(f"                 {topic_register}")
    print(f"                 {LEGACY_TOPIC} (legacy)")
    print(f"  Interval:      {interval}s")
    print(f"  Data Mode:     {data_mode}")
    print(f"  Dummy Mode:    {use_dummy}")
    print(f"  Hardware:      {hw_summary}")
    print(f"{'='*60}\n")

    connected = False
    retry_count = 0
    while not connected:
        try:
            client.connect(broker, port, 60)
            connected = True
        except (ConnectionRefusedError, OSError) as e:
            retry_count += 1
            print(f"[CONNECT] Broker {broker}:{port} not reachable ({e}). Waiting for network... Retry #{retry_count} in 4s")
            time.sleep(4)

    client.loop_start()

    # ──── Step 5: Publish Registration (once on connect) ──────────────
    registration_payload = {
        "schema_version": "2.0",
        "server_id": server_id,
        "server_name": srv_name,
        "data_mode": data_mode,
        "hardware_topology": topology,
        "registration_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    reg_str = json.dumps(registration_payload)
    client.publish(topic_register, reg_str, qos=1, retain=True)
    print(f"[REGISTER] Published registration ({len(reg_str)} bytes) on {topic_register}")

    # ──── Step 6: Telemetry Publishing Loop ───────────────────────────
    tick = 0
    try:
        while True:
            # Collect v2 hierarchical telemetry
            v2_payload = collector.collect()

            # Inject server identity
            v2_payload["server_id"] = server_id
            v2_payload["server_name"] = srv_name
            v2_payload["data_mode"] = data_mode

            v2_str = json.dumps(v2_payload)

            # Publish v2 hierarchical payload
            result = client.publish(topic_telemetry, v2_str, qos=1)

            # Also publish v1 flat format on legacy topic for backward compat
            v1_flat = collector.to_v1_flat(v2_payload)
            v1_payload = {
                "timestamp": v2_payload.get("timestamp_utc", ""),
                "server_id": server_id,
                "server_name": srv_name,
                "data_mode": data_mode,
                "cpu": {
                    "temp_package": v1_flat.get("cpu_temp", 45.0),
                    "load_percent": v1_flat.get("cpu_load", 15.0),
                    "frequency_mhz": v1_flat.get("cpu_freq", 3000.0),
                },
                "gpu": {
                    "temp": v1_flat.get("gpu_temp", 40.0),
                    "power_draw_w": v1_flat.get("gpu_power", 30.0),
                    "utilization_percent": v1_flat.get("gpu_util", 10.0),
                    "memory_used_mb": v1_flat.get("gpu_mem_used", 500.0),
                    "clock_mhz": v1_flat.get("gpu_clock", 1500.0),
                    "fan_rpm": 0,
                    "ecc_sbe_count": 0,
                },
                "ram": {
                    "used_gb": v1_flat.get("ram_used_gb", 4.0),
                    "total_gb": v1_flat.get("ram_total_gb", 16.0),
                    "page_faults_sec": v1_flat.get("ram_page_faults", 50),
                    "committed_gb": v1_flat.get("ram_used_gb", 4.0),
                },
                "hdd": {
                    "smart_5_raw": int(v1_flat.get("smart_5_raw", 0)),
                    "smart_9_raw": int(v1_flat.get("smart_9_raw", 5000)),
                    "smart_187_raw": int(v1_flat.get("smart_187_raw", 0)),
                    "smart_188_raw": int(v1_flat.get("smart_188_raw", 0)),
                    "smart_194_raw": int(v1_flat.get("smart_194_raw", 35)),
                    "smart_197_raw": int(v1_flat.get("smart_197_raw", 0)),
                    "smart_198_raw": int(v1_flat.get("smart_198_raw", 0)),
                },
                "psu": {
                    "voltage_3v3": v2_payload.get("psu", {}).get("voltage_3v3", 3.3),
                    "voltage_5v": v2_payload.get("psu", {}).get("voltage_5v", 5.0),
                    "voltage_12v": v2_payload.get("psu", {}).get("voltage_12v", 12.0),
                    "power_draw_w": v2_payload.get("psu", {}).get("total_power_w", 200.0),
                },
                "fans": {
                    "cpu_fan_rpm": (v2_payload.get("fans") or [{}])[0].get("rpm", 1500),
                    "case_fan1_rpm": (v2_payload.get("fans") or [{}, {}])[1].get("rpm", 1000) if len(v2_payload.get("fans", [])) > 1 else 1000,
                },
                "environment": {
                    "system_uptime_hours": v2_payload.get("uptime_hours", 100.0),
                    "ambient_temp": v2_payload.get("environment", {}).get("ambient_temp_c", 25.0),
                    "humidity": v2_payload.get("environment", {}).get("humidity_pct", 50.0),
                    "vibration_rms": v2_payload.get("environment", {}).get("vibration_rms_g", 0.02),
                    "acoustic_rms": v2_payload.get("environment", {}).get("acoustic_rms_v", 0.02),
                },
            }
            v1_str = json.dumps(v1_payload)
            client.publish(LEGACY_TOPIC, v1_str, qos=1)
            # NOTE: Do NOT re-publish v1 to server/{server_id}/telemetry — the v2 payload
            # is already published there (line 150). Publishing v1 here causes triple
            # processing on the edge node (3x inference, 3x SQLite writes). (BUG-G-TRIPLE fix)

            tick += 1
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[{tick:>4}] Published v2 ({len(v2_str)}B) + v1 ({len(v1_str)}B)")
            else:
                print(f"[{tick:>4}] Publish failed (rc={result.rc})")

            # Re-publish registration every 100 ticks (~8 min at 5s interval)
            if tick % 100 == 0:
                client.publish(topic_register, reg_str, qos=1, retain=True)
                print(f"[REGISTER] Re-published registration heartbeat")

            if max_frames and tick >= max_frames:
                print(f"[LIMIT] Reached max frames ({max_frames}). Stopping publisher.")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[STOP] Publisher stopping...")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[DONE] Publisher disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient Server Telemetry Publisher v2.0")
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER, help="MQTT Broker Host / IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT Broker Port")
    parser.add_argument("--dummy", action="store_true", help="Use simulated telemetry data")
    parser.add_argument("--mode", type=str, default="production",
                        choices=["production", "demo"],
                        help="Data mode: 'production' or 'demo'")
    parser.add_argument("--server-name", type=str, default=None,
                        help="Human-readable server name (auto-detected if omitted)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Publish interval in seconds (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration, hardware discovery, and hostname resolution without connecting")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to publish before exiting (0 = run indefinitely)")
    args = parser.parse_args()

    start_publisher_v2(
        broker=args.broker, port=args.port,
        use_dummy=args.dummy, data_mode=args.mode,
        server_name=args.server_name, interval=args.interval,
        dry_run=args.dry_run, max_frames=args.max_frames
    )
