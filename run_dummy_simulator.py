import paho.mqtt.client as mqtt
import time
import json
import random
import sys

import os
sys.stdout.reconfigure(encoding="utf-8")

# Auto-load configs/network.env if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "network.env")
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

BROKER = os.environ.get("EDGE_BROKER_HOST", os.environ.get("MQTT_BROKER", "127.0.0.1"))
PORT = int(os.environ.get("EDGE_BROKER_PORT", os.environ.get("MQTT_PORT", 1883)))
TOPIC = "server/telemetry"

def generate_telemetry(anomaly_type=None):
    # Generates a realistic 28-feature vector (expected by the models)
    
    if anomaly_type == "cpu":
        print("[ANOMALY] CPU Thermal Runaway!")
        data = {"mock_class": 2, "cpu_temp_package": random.uniform(95.0, 105.0), "cpu_load_percent": random.uniform(95.0, 100.0), "gpu_temp": random.uniform(40.0, 50.0), "ram_used_gb": random.uniform(4.0, 8.0), "environment_vibration_rms": random.uniform(0.1, 0.5), "psu_voltage_12v": random.uniform(11.8, 12.2)}
    elif anomaly_type == "gpu":
        print("[ANOMALY] GPU Overheating!")
        data = {"mock_class": 6, "cpu_temp_package": random.uniform(40.0, 50.0), "gpu_temp": random.uniform(95.0, 110.0), "gpu_utilization_percent": random.uniform(95.0, 100.0), "ram_used_gb": random.uniform(4.0, 8.0), "environment_vibration_rms": random.uniform(0.1, 0.5), "psu_voltage_12v": random.uniform(11.8, 12.2)}
    elif anomaly_type == "ram":
        print("[ANOMALY] RAM Leak / Failure!")
        data = {"mock_class": 5, "cpu_temp_package": random.uniform(40.0, 50.0), "gpu_temp": random.uniform(40.0, 50.0), "ram_used_gb": random.uniform(15.5, 16.0), "ram_page_faults_sec": random.uniform(5000.0, 10000.0), "environment_vibration_rms": random.uniform(0.1, 0.5), "psu_voltage_12v": random.uniform(11.8, 12.2)}
    elif anomaly_type == "hdd":
        print("[ANOMALY] HDD Failure / Overheating!")
        data = {"mock_class": 1, "cpu_temp_package": random.uniform(40.0, 50.0), "gpu_temp": random.uniform(40.0, 50.0), "hdd_smart_194_raw": random.uniform(65.0, 80.0), "hdd_smart_5_raw": random.uniform(10.0, 50.0), "environment_vibration_rms": random.uniform(0.1, 0.5), "psu_voltage_12v": random.uniform(11.8, 12.2)}
    elif anomaly_type == "psu":
        print("[ANOMALY] PSU Power Surge!")
        data = {"mock_class": 3, "cpu_temp_package": random.uniform(50.0, 60.0), "gpu_temp": random.uniform(50.0, 60.0), "psu_voltage_12v": random.uniform(14.0, 15.0), "psu_voltage_3v3": random.uniform(4.0, 5.0), "environment_vibration_rms": random.uniform(0.1, 0.5)}
    elif anomaly_type == "fans":
        print("[ANOMALY] Fans Failing!")
        data = {"mock_class": 4, "cpu_temp_package": random.uniform(70.0, 80.0), "gpu_temp": random.uniform(70.0, 80.0), "environment_vibration_rms": random.uniform(20.0, 30.0), "environment_acoustic_rms": random.uniform(80.0, 95.0), "psu_voltage_12v": random.uniform(11.8, 12.2)}
    else:
        data = {
            "mock_class": 0,
            "cpu_temp_package": random.uniform(35.0, 55.0),
            "gpu_temp": random.uniform(40.0, 60.0),
            "cpu_load_percent": random.uniform(10.0, 40.0),
            "gpu_utilization_percent": random.uniform(10.0, 40.0),
            "ram_used_gb": random.uniform(2.0, 6.0),
            "hdd_smart_194_raw": random.uniform(30.0, 45.0),
            "psu_voltage_12v": random.uniform(11.9, 12.1),
            "environment_vibration_rms": random.uniform(0.1, 0.5),
            "environment_acoustic_rms": random.uniform(20.0, 40.0)
        }
        
    # Fill remaining canonical features with realistic defaults
    try:
        from server.training.feature_schema import FEATURE_NAMES_V1, FEATURE_DEFAULTS
        telem_key_map = {
            'cpu_temp': 'cpu_temp_package', 'cpu_load': 'cpu_load_percent',
            'gpu_util': 'gpu_utilization_percent', 'smart_5_raw': 'hdd_smart_5_raw',
            'smart_194_raw': 'hdd_smart_194_raw', 'vibration_rms': 'environment_vibration_rms',
            'acoustic_rms': 'environment_acoustic_rms',
        }
        for feat in FEATURE_NAMES_V1:
            telem_key = telem_key_map.get(feat, feat)
            if telem_key not in data and feat not in data:
                data[telem_key] = FEATURE_DEFAULTS.get(feat, 0.0) + random.uniform(-0.1, 0.1)
    except ImportError:
        pass

    return data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dummy Patient Telemetry Simulator")
    parser.add_argument("--broker", type=str, default=BROKER, help="MQTT Broker IP")
    parser.add_argument("--port", type=int, default=PORT, help="MQTT Broker Port")
    parser.add_argument("--server-id", type=str, default="dummy-simulator",
                        help="Server ID to use in payloads")
    parser.add_argument("--server-name", type=str, default="Anomaly Simulator",
                        help="Human-readable server name")
    parser.add_argument("--no-mock", action="store_true",
                        help="Omit mock_class to test real AI inference pipeline")
    args = parser.parse_args()

    broker = args.broker
    port = args.port
    server_id = args.server_id
    server_name = args.server_name

    print(f"[START] Starting Dummy Patient Simulator -> target: {broker}:{port}")
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()
    
    try:
        client.connect(broker, port, 60)
        print("[OK] Connected to RPi4 Edge Broker successfully!\n")
    except Exception as e:
        print(f"[ERROR] Failed to connect to broker at {broker}: {e}")
        print("Make sure START_ALL.bat has been run and the RPi4 is online.")
        sys.exit(1)
        
    client.loop_start()
    
    counter = 0
    try:
        anomaly_types = ["cpu", "gpu", "ram", "hdd", "psu", "fans"]
        anomaly_index = 0
        
        while True:
            # Inject an anomaly every ~5 seconds
            if counter > 0 and counter % 5 == 0:
                current_anomaly = anomaly_types[anomaly_index % len(anomaly_types)]
                payload = generate_telemetry(anomaly_type=current_anomaly)
                anomaly_index += 1
            else:
                payload = generate_telemetry(anomaly_type=None)
            # Include server_id and server_name for multi-server dashboard support
            payload["server_id"] = server_id
            payload["server_name"] = server_name
            if args.no_mock and "mock_class" in payload:
                del payload["mock_class"]
            client.publish(TOPIC, json.dumps(payload))
            
            print(f"[{time.strftime('%H:%M:%S')}] Published {len(payload)} features | CPU: {payload.get('cpu_temp_package', 0.0):.1f}C")
            
            counter += 1
            time.sleep(1) # Send 1 per second
            
    except KeyboardInterrupt:
        print("\n[STOP] Simulator stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
