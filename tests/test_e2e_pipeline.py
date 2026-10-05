"""
End-to-end simulation of the edge ingestion and dashboard state pipeline.
"""
import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from edge.main import on_telemetry_received
from edge.api_server import current_state, state_lock, local_store


class MockMessage:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload


def test_full_pipeline():
    print("[E2E TEST] Simulating telemetry messages through edge/main.py pipeline...")

    # 1. Normal telemetry tick
    payload1 = {
        'server_id': 'srv-edge-prod-01',
        'timestamp': '2026-08-15T20:00:00Z',
        'cpu_temp_package': 48.2,
        'cpu_load_percent': 22.0,
        'cpu_freq_current': 3200.0,
        'gpu_temp': 42.0,
        'gpu_utilization_percent': 12.0,
        'ram_used_gb': 4.5,
        'ram_total_gb': 16.0,
        'fans_cpu_fan_rpm': 1520,
        'psu_voltage_12v': 12.02,
        'psu_voltage_3v3': 3.31,
        'mock_class': 0
    }
    on_telemetry_received(payload1)

    with state_lock:
        srv = current_state.get('srv-edge-prod-01')
        assert srv is not None, 'Server not found in current_state'
        print(f"Tick 1: Health={srv['health_index']}%, CPU Temp={srv['components']['cpu']['metrics']['temp']}C")
        assert len(srv['history']) >= 1

    # 2. Critical Thermal Anomaly tick
    payload2 = {
        'server_id': 'srv-edge-prod-01',
        'timestamp': '2026-08-15T20:00:03Z',
        'cpu_temp_package': 96.5,
        'cpu_load_percent': 98.0,
        'cpu_freq_current': 3800.0,
        'gpu_temp': 58.0,
        'gpu_utilization_percent': 25.0,
        'ram_used_gb': 6.0,
        'ram_total_gb': 16.0,
        'fans_cpu_fan_rpm': 900,
        'psu_voltage_12v': 11.95,
        'psu_voltage_3v3': 3.28,
        'mock_class': 2
    }
    on_telemetry_received(payload2)

    with state_lock:
        srv = current_state.get('srv-edge-prod-01')
        assert (srv['components']['cpu']['status'] in ['WARNING', 'CRITICAL'] or 
                srv['components']['fans']['status'] in ['WARNING', 'CRITICAL'])
        assert len(srv['anomalies']) >= 1

    # 3. Check SQLite persistence
    records = local_store.get_recent('srv-edge-prod-01', hours=1.0, limit=10)
    print(f"SQLite Cached Records: {len(records)} entries found")
    assert len(records) >= 2
    print("SUCCESS: Full E2E Pipeline successfully validated!")


if __name__ == "__main__":
    test_full_pipeline()
