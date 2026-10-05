"""
Interactive Test Dashboard Launcher with Multi-Server Simulated Telemetry Feeder.
Starts FastAPI server on port 8080 and streams live telemetry into memory and SQLite database.
"""

import sys
import os
import time
import threading
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from edge.api_server import app, current_state, state_lock, local_store, _default_server_state


def populate_initial_state():
    """Populate rich initial data for 2 servers so all tabs and charts have full data."""
    with state_lock:
        # Server 1: Healthy Production Server
        current_state["srv-alpha-01"] = _default_server_state()
        srv1 = current_state["srv-alpha-01"]
        srv1["health_index"] = 96.4
        srv1["inference"] = {
            "predicted_class": 0,
            "fault_label": "HEALTHY",
            "confidence": 0.985,
            "ttf_days": 365.0,
            "offloaded": False,
            "location": "EDGE",
            "latency_ms": 14.2
        }
        srv1["components"]["cpu"]["metrics"] = {"temp": 52.4, "load": 28.5, "freq": 3200.0, "delta": 24.1}
        srv1["components"]["gpu"]["metrics"] = {"temp": 46.0, "util": 15.0, "power": 35.0, "vram_mb": 1200}
        srv1["components"]["ram"]["metrics"] = {"percent": 34.2, "used_gb": 5.4, "total_gb": 16.0, "page_faults": 42}
        srv1["components"]["psu"]["metrics"] = {"status": "STABLE", "dev_3v3": 0.4, "dev_12v": 0.5, "stability": 0.03}
        srv1["components"]["fans"]["metrics"] = {"rpm": 1850, "deviation": 2.1, "status": "OPTIMAL"}

        # Server 2: Degraded Thermal Runaway Server
        current_state["srv-beta-02"] = _default_server_state()
        srv2 = current_state["srv-beta-02"]
        srv2["health_index"] = 48.0
        srv2["inference"] = {
            "predicted_class": 2,
            "fault_label": "CPU_THERMAL_RUNAWAY",
            "confidence": 0.932,
            "ttf_days": 3.8,
            "offloaded": True,
            "location": "CLOUD",
            "latency_ms": 48.6
        }
        srv2["components"]["cpu"]["status"] = "CRITICAL"
        srv2["components"]["cpu"]["ttf_days"] = 3.8
        srv2["components"]["cpu"]["metrics"] = {"temp": 94.8, "load": 96.2, "freq": 3950.0, "delta": 68.4}
        srv2["components"]["fans"]["status"] = "WARNING"
        srv2["components"]["fans"]["ttf_days"] = 5.2
        srv2["components"]["fans"]["metrics"] = {"rpm": 820, "deviation": -54.0, "status": "FAILING"}
        srv2["anomalies"].append({
            "time": "2026-08-15T21:00:00Z",
            "fault": "CPU_THERMAL_RUNAWAY",
            "component": "cpu",
            "confidence": 0.932
        })
        srv2["recent_shap_data"] = {
            "top_features": [
                {"feature": "cpu_temp_package", "shap_value": 0.68, "impact": 0.68},
                {"feature": "fan_rpm_deviation", "shap_value": 0.54, "impact": 0.54},
                {"feature": "cpu_load_percent", "shap_value": 0.32, "impact": 0.32},
                {"feature": "psu_voltage_12v_dev", "shap_value": -0.15, "impact": -0.15},
                {"feature": "smart_5_raw", "shap_value": -0.22, "impact": -0.22},
                {"feature": "environment_vibration_rms", "shap_value": -0.35, "impact": -0.35}
            ]
        }

        # Populate rolling sparklines for both
        now = time.time()
        for i in range(30):
            t_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - (30 - i) * 3))
            srv1["history"].append({
                "time": t_str,
                "cpu_temp": 50.0 + (i % 5) * 0.8,
                "cpu_load": 25.0 + (i % 4) * 2.0,
                "gpu_temp": 44.0 + (i % 3) * 0.5,
                "gpu_util": 12.0,
                "ram_used_gb": 5.2,
                "ram_percent": 32.5,
                "fan_rpm": 1800 + (i % 6) * 15,
                "psu_12v_dev": 0.4,
                "ambient_temp": 24.5,
                "vibration_rms": 0.01,
                "health_index": 96.0 + (i % 3) * 0.5
            })

            srv2["history"].append({
                "time": t_str,
                "cpu_temp": 80.0 + i * 0.5,
                "cpu_load": 90.0 + (i % 3) * 2.0,
                "gpu_temp": 62.0 + (i % 4) * 1.0,
                "gpu_util": 40.0,
                "ram_used_gb": 8.0,
                "ram_percent": 50.0,
                "fan_rpm": 950 - i * 4,
                "psu_12v_dev": 0.8,
                "ambient_temp": 26.0,
                "vibration_rms": 0.04,
                "health_index": max(30.0, 75.0 - i * 1.0)
            })

    # Also persist records into SQLite for Database tab testing
    for i in range(15):
        t_hist = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - (15 - i) * 60))
        local_store.store_telemetry("srv-alpha-01", {
            "timestamp": t_hist,
            "cpu_temp": 52.0 + (i % 3),
            "cpu_load": 28.0,
            "fan_rpm": 1850
        }, {
            "predictions": {"class": 0, "confidence": 0.98, "ttf_normalized": 1.0},
            "anomaly_detected": False
        })

        local_store.store_telemetry("srv-beta-02", {
            "timestamp": t_hist,
            "cpu_temp": 92.0 + (i % 4),
            "cpu_load": 95.0,
            "fan_rpm": 820
        }, {
            "predictions": {"class": 2, "confidence": 0.93, "ttf_normalized": 0.01},
            "anomaly_detected": True
        })


def telemetry_feeder_loop():
    """Background thread to slowly tick telemetry so SSE pushes live updates."""
    while True:
        time.sleep(2.0)
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with state_lock:
            for sid, srv in current_state.items():
                srv["last_seen"] = now_str
                # Keep rolling history trimmed to 60
                if len(srv["history"]) > 60:
                    srv["history"].pop(0)


if __name__ == "__main__":
    populate_initial_state()
    t = threading.Thread(target=telemetry_feeder_loop, daemon=True)
    t.start()
    print("Starting FastAPI Dashboard Server on http://127.0.0.1:8080...")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
