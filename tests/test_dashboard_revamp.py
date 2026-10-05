"""
Automated Verification Suite for Revamped Production Dashboard
Tests the dual-mode operator & hardware mechanic dashboard endpoints,
SQLite persistence, telemetry feature extraction, and state transitions.
"""

import os
import sys
import json
import time
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from edge.api_server import app, current_state, state_lock, _default_server_state
from edge.local_store import EdgeLocalStore
from server.training.feature_schema import FEATURE_NAMES_V1, FEATURE_DEFAULTS

client = TestClient(app)


def test_dashboard_html_served():
    """Verify that root endpoint serves the revamped dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    # Verify core sections and tabs are in the DOM
    assert "Simple Overview" in html
    assert "Telemetry Trends" in html
    assert "Root Cause & XAI" in html
    assert "Database & History" in html
    assert "Platform Health" in html
    assert "mainTrendsChart" in html
    assert "shapImpactChart" in html
    assert "dbHistoryTable" in html
    assert "AI Assistant" in html


def test_api_state_default():
    """Verify /api/state structure and default values."""
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert "servers" in data
    assert "timestamp" in data


def test_state_update_and_sparkline():
    """Simulate receiving telemetry and updating state."""
    server_id = "test-prod-server-01"
    
    with state_lock:
        current_state[server_id] = _default_server_state()
        srv = current_state[server_id]
        srv["health_index"] = 92.5
        srv["components"]["cpu"]["metrics"]["temp"] = 68.4
        srv["components"]["cpu"]["status"] = "HEALTHY"
        srv["history"].append({
            "time": "2026-08-15T20:00:00Z",
            "cpu_temp": 68.4,
            "cpu_load": 35.0,
            "gpu_temp": 52.0,
            "gpu_util": 20.0,
            "ram_used_gb": 6.2,
            "ram_percent": 38.7,
            "fan_rpm": 2200,
            "psu_12v_dev": 0.4,
            "ambient_temp": 24.0,
            "vibration_rms": 0.015,
            "health_index": 92.5
        })

    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert server_id in data["servers"]
    srv_data = data["servers"][server_id]
    assert srv_data["health_index"] == 92.5
    assert srv_data["components"]["cpu"]["metrics"]["temp"] == 68.4
    assert len(srv_data["history"]) == 1


def test_sqlite_persistence_and_query():
    """Verify EdgeLocalStore can persist and query telemetry history."""
    store = EdgeLocalStore()
    test_server = "test-store-server"
    raw_payload = {"cpu_temp": 82.0, "fan_rpm": 2900, "psu_voltage_12v_dev": 0.6}
    inference_result = {
        "predictions": {"class": 1, "confidence": 0.94, "ttf_normalized": 0.05},
        "anomaly_detected": True,
        "offloaded": False,
        "latency_ms": 17.8
    }

    record_id = store.store_telemetry(test_server, raw_payload, inference_result)
    assert record_id is not None

    # Query history endpoint
    resp = client.get(f"/api/history?server_id={test_server}&hours=1.0&limit=10")
    assert resp.status_code == 200
    hist_data = resp.json()
    assert hist_data["server_id"] == test_server
    assert len(hist_data["records"]) >= 1
    rec = hist_data["records"][0]
    assert rec["anomaly_detected"] == 1
    assert rec["predicted_class"] == 1

    # Query database anomalies endpoint
    anom_resp = client.get(f"/api/database/anomalies?server_id={test_server}&limit=10")
    assert anom_resp.status_code == 200
    anom_data = anom_resp.json()
    assert anom_data["count"] >= 1
    assert anom_data["anomalies"][0]["fault_class"] == 1


def test_platform_status_endpoint():
    """Verify platform health status endpoint returns subsystem matrix."""
    resp = client.get("/api/platform/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "edge_node" in data
    assert data["edge_node"]["sqlite_cache"] == "connected"
    assert "mqtt_broker" in data
    assert "cloud_backend" in data


def test_chat_endpoint_fallback():
    """Verify /api/chat responds intelligently in offline rule-based fallback mode."""
    # Query system status
    resp = client.post("/api/chat", json={"message": "What is the system health status?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "Health" in data["response"] or "health" in data["response"]

    # Query failure risk
    resp_fail = client.post("/api/chat", json={"message": "Are there any components failing?"})
    assert resp_fail.status_code == 200
    assert "response" in resp_fail.json()


if __name__ == "__main__":
    print("Running Dashboard Revamp Test Suite...")
    test_dashboard_html_served()
    print("PASS: test_dashboard_html_served")
    test_api_state_default()
    print("PASS: test_api_state_default")
    test_state_update_and_sparkline()
    print("PASS: test_state_update_and_sparkline")
    test_sqlite_persistence_and_query()
    print("PASS: test_sqlite_persistence_and_query")
    test_platform_status_endpoint()
    print("PASS: test_platform_status_endpoint")
    test_chat_endpoint_fallback()
    print("PASS: test_chat_endpoint_fallback")
    print("ALL TESTS PASSED SUCCESSFULLY!")
