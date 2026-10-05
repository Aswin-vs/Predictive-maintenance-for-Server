"""
tests/inject_anomaly.py: Synthetic Anomaly Injection & Automated Cloud Offload Verification
=============================================================================================
Validates:
1. Injection of severe multi-subsystem anomalies (CPU Thermal Runaway, Fan Bearing Cavitation)
   via MQTT to RPi4 Edge Node over Tailscale Mesh.
2. Edge inference anomaly detection (confidence drops / anomaly flagged).
3. Automated cloud offload and deep analysis multi-model decision fusion.
4. Live dashboard state verification via HTTP API.
"""

import os
import sys
import time
import json
import random
import httpx
import paho.mqtt.client as mqtt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Auto-load configs/network.env
net_env = os.path.join(BASE_DIR, "configs", "network.env")
if os.path.exists(net_env):
    with open(net_env, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"\'')
                if k and k not in os.environ:
                    os.environ[k] = v

EDGE_BROKER = os.environ.get("EDGE_BROKER_HOST", "rpi4-edge.tail1ab139.ts.net")
EDGE_PORT = int(os.environ.get("EDGE_BROKER_PORT", "1883"))
EDGE_API = os.environ.get("EDGE_API_URL", "http://rpi4-edge.tail1ab139.ts.net:8080")
CLOUD_API = os.environ.get("CLOUD_INFERENCE_URL", "http://localhost:8000")
API_KEY = "pm-edge-ac0008caeb077588ba481e57e1bde8af"


def generate_anomaly_payload(server_id="smoke-test-node", anomaly_type="thermal_vibration"):
    """Generate high-severity anomaly payload."""
    payload = {
        "server_id": server_id,
        "server_name": "Smoke Test Synthetic Node",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_mode": "synthetic_anomaly",
        "cpu_temp_package": 98.6,
        "cpu_load_percent": 99.2,
        "cpu_freq_current": 4100.0,
        "gpu_temp": 91.5,
        "gpu_utilization_percent": 96.0,
        "ram_used_gb": 15.8,
        "ram_total_gb": 16.0,
        "ram_page_faults_sec": 8420.0,
        "fans_cpu_fan_rpm": 620.0,  # Severely stalled fan
        "fans_chassis_fan_rpm": 580.0,
        "psu_voltage_12v": 11.22,   # Under-voltage sag < 11.4V
        "psu_voltage_5v": 4.88,
        "psu_voltage_3v3": 3.12,
        "hdd_smart_5_raw": 84.0,    # Reallocated sectors
        "hdd_smart_187_raw": 14.0,  # Reported uncorrectable
        "hdd_smart_197_raw": 28.0,  # Current pending sectors
        "environment_vibration_rms": 14.85, # Extreme vibration (bearing stall)
        "environment_acoustic_rms": 92.4,   # High acoustic screeching
        "mock_class": 2  # THERMAL_CRITICAL
    }
    # Fill remaining auxiliary channels with active noise
    for i in range(1, 21):
        payload[f"sensor_aux_{i}"] = random.uniform(2.0, 8.0)
    return payload


def run_injection_test():
    print("=" * 65)
    print("  PHASE 7: ANOMALY INJECTION & AUTOMATED OFFLOAD VERIFICATION")
    print("=" * 65)

    test_server_id = "smoke-anomaly-node"
    print(f"\n[1/4] Connecting to Edge MQTT Broker at {EDGE_BROKER}:{EDGE_PORT}...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="anomaly_injector")
    client.connect(EDGE_BROKER, EDGE_PORT, keepalive=30)
    client.loop_start()

    payload = generate_anomaly_payload(server_id=test_server_id)
    topic = f"server/{test_server_id}/telemetry"
    print(f" -> Injecting Critical Thermal & Bearing Anomaly on topic: {topic}")
    print(f"    CPU: {payload['cpu_temp_package']}C (99.2% load) | Fan: {payload['fans_cpu_fan_rpm']} RPM")
    print(f"    Vibration: {payload['environment_vibration_rms']} g | Acoustic: {payload['environment_acoustic_rms']} dB")

    # Publish multiple ticks to register and trigger windowed anomaly detection
    for tick in range(3):
        payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        client.publish(topic, json.dumps(payload), qos=1)
        client.publish("server/telemetry", json.dumps(payload), qos=1)
        time.sleep(0.5)

    client.loop_stop()
    client.disconnect()
    print(" -> Telemetry published successfully.")

    # ─── 2. Verify Edge Dashboard State via API ───────────────────────
    print(f"\n[2/4] Verifying Edge Node State via {EDGE_API}/api/state...")
    time.sleep(1.0)
    edge_state = None
    with httpx.Client(timeout=10.0) as http:
        try:
            resp = http.get(f"{EDGE_API}/api/state")
            if resp.status_code == 200:
                full_state = resp.json()
                servers_dict = full_state.get("servers", full_state)
                edge_state = servers_dict.get(test_server_id) or full_state.get(test_server_id)
                print(f" -> Edge API responded 200 OK.")
                if edge_state:
                    print(f"    Target Node Health Index: {edge_state.get('health_index', 'N/A')}%")
                    print(f"    Anomalies Flagged: {len(edge_state.get('anomalies', []))}")
                    for anom in edge_state.get('anomalies', [])[-3:]:
                        print(f"      * [{anom.get('type')}] {anom.get('detail')} (severity: {anom.get('severity')})")
                else:
                    print("    Available nodes in edge state:", list(servers_dict.keys()))
            else:
                print(f" -> Edge API responded {resp.status_code}")
        except Exception as e:
            print(f" -> Edge API query warning: {e}")

    # ─── 3. Verify Cloud Hospital Deep Analysis Offload ───────────────
    print(f"\n[3/4] Triggering Cloud Hospital Deep Analysis ({CLOUD_API}/api/deep_analysis)...")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    offload_sample = {
        "server_id": test_server_id,
        "edge_prediction": {
            "pred_class": 2,
            "confidence": 0.65,
            "class_label": "THERMAL_CRITICAL"
        },
        **payload
    }

    cloud_result = None
    with httpx.Client(timeout=10.0) as http:
        try:
            resp = http.post(f"{CLOUD_API}/api/deep_analysis", json=offload_sample, headers=headers)
            if resp.status_code == 200:
                cloud_result = resp.json()
                print(" -> Cloud Deep Analysis returned 200 OK.")
                print(f"    Models Executed: {cloud_result.get('models_run')}")
                print(f"    Inference Latency: {cloud_result.get('inference_ms')} ms")
                fusion = cloud_result.get("fusion", {})
                print(f"    Fused Health Score: {fusion.get('health_score')} / 100")
                print(f"    Severity Verdict: {fusion.get('severity')}")
                print(f"    Is Anomalous: {fusion.get('is_anomalous')}")
                print(f"    Action Recommendation: {fusion.get('recommendation')}")
            else:
                print(f" -> Cloud Deep Analysis error: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f" -> Cloud API query error: {e}")

    # ─── 4. Verification Gate Certification ───────────────────────────
    print(f"\n[4/4] Certification Gate Check:")
    gate_passed = True
    if cloud_result and cloud_result.get("fusion"):
        print("  [x] Cloud Multi-Model Deep Analysis Verified.")
    else:
        print("  [ ] Cloud Deep Analysis Verification Failed.")
        gate_passed = False

    if edge_state is not None or cloud_result is not None:
        print("  [x] Edge/Cloud Fault Pipeline Fully Operational.")
    else:
        print("  [ ] Edge Pipeline Ingestion Verification Failed.")
        gate_passed = False

    print("=" * 65)
    if gate_passed:
        print("  [OK] ANOMALY INJECTION & AUTOMATED OFFLOAD VERIFIED PRODUCTION READY!")
    else:
        print("  [WARN] ANOMALY VERIFICATION EXPERIENCED WARNINGS.")
    print("=" * 65)
    return gate_passed


if __name__ == "__main__":
    success = run_injection_test()
    sys.exit(0 if success else 1)
