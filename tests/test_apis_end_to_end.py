"""
End-to-End Cloud Backend Test Script.

Tests the Inference Server (port 8000) and Chat API (port 8001) on the LOQ RTX 4060.
Sends dummy telemetry matching the collector.py payload schema and validates responses.

Usage:
    python tests/test_apis_end_to_end.py
    python tests/test_apis_end_to_end.py --inference-url http://192.168.1.100:8000/api/offload
"""

import requests
import json
import time
import sys
import argparse

sys.stdout.reconfigure(encoding='utf-8')

# Default URLs (configurable via CLI args)
DEFAULT_INFERENCE_URL = "http://127.0.0.1:8000/api/offload"
DEFAULT_CHAT_URL = "http://127.0.0.1:8001/api/chat"
DEFAULT_API_KEY = "pm-edge-ac0008caeb077588ba481e57e1bde8af"

def run_tests(inference_url: str, chat_url: str, api_key: str = DEFAULT_API_KEY):
    print("--- END-TO-END CLOUD BACKEND TEST ---\n")
    
    # Payload matches the nested structure from collector.py
    dummy_telemetry = {
        "timestamp": "2026-08-03T06:00:00Z",
        "cpu": {
            "temp_package": 82.5,
            "temp_cores": [80.1, 81.3, 83.0, 82.5],
            "load_percent": 75.0,
            "frequency_mhz": 3800,
            "throttle_count": 0,
            "context_switches": 12000
        },
        "gpu": {
            "temp": 88.0,
            "power_draw_w": 180.0,
            "memory_used_mb": 6500,
            "memory_total_mb": 8192,
            "clock_mhz": 1800,
            "fan_rpm": 2200,
            "ecc_sbe_count": 0,
            "utilization_percent": 95
        },
        "ram": {
            "used_gb": 12.5,
            "total_gb": 16.0,
            "page_faults_sec": 150,
            "committed_gb": 13.2
        },
        "hdd": {
            "smart_5_raw": 10,
            "smart_9_raw": 8500,
            "smart_187_raw": 3,
            "smart_188_raw": 0,
            "smart_197_raw": 2,
            "smart_198_raw": 0,
            "smart_194_raw": 42,
            "read_error_rate": 0,
            "temperature": 42
        },
        "psu": {
            "voltage_3v3": 3.28,
            "voltage_5v": 4.95,
            "voltage_12v": 11.85,
            "power_draw_w": 320.0
        },
        "fans": {
            "cpu_fan_rpm": 1200,
            "case_fan1_rpm": 900,
            "case_fan2_rpm": 850
        },
        "environment": {
            "system_uptime_hours": 720.5,
            "ambient_temp": 28.0,
            "humidity": 55.0,
            "vibration_rms": 0.04,
            "acoustic_rms": 0.03
        }
    }
    
    passed = 0
    failed = 0
    
    # --- Test 1: Inference Server ---
    print(f"[1] Sending telemetry to Inference Server ({inference_url})...")
    print(f"    Payload size: {len(json.dumps(dummy_telemetry))} bytes")
    
    try:
        headers = {"X-API-Key": api_key} if api_key else {}
        inf_response = requests.post(inference_url, json=dummy_telemetry, headers=headers, timeout=10)
        inf_response.raise_for_status()
        result = inf_response.json()
        print("\n    [PASS] Inference Result Received:")
        print(f"    Source: {result.get('source', 'N/A')}")
        print(f"    Predictions: {json.dumps(result.get('predictions', {}), indent=6)}")
        if result.get('shap_explanation'):
            print(f"    SHAP: {json.dumps(result['shap_explanation'], indent=6)}")
        passed += 1
    except requests.exceptions.ConnectionError:
        print(f"\n    [FAIL] Cannot connect to {inference_url}. Is inference_server.py running?")
        result = None
        failed += 1
    except Exception as e:
        print(f"\n    [FAIL] Inference Server Error: {e}")
        result = None
        failed += 1
    
    # --- Test 2: Chat API ---
    print(f"\n[2] Asking Chat API to explain prediction ({chat_url})...")
    
    # Build payload matching ChatRequest Pydantic model
    chat_payload = {
        "message": "Why is the TTF so low? Explain the anomalies.",
        "current_health_state": {
            "health_index": 65.0,
            "components": {
                "cpu": {"status": "WARNING", "ttf_days": 45},
                "gpu": {"status": "CRITICAL", "ttf_days": 12},
                "hdd": {"status": "WARNING", "ttf_days": 28},
                "psu": {"status": "HEALTHY", "ttf_days": 999},
                "ram": {"status": "HEALTHY", "ttf_days": 999},
                "fans": {"status": "HEALTHY", "ttf_days": 999}
            }
        },
        "recent_shap_data": result.get("shap_explanation") if result else None
    }
    
    try:
        headers = {"X-API-Key": api_key} if api_key else {}
        chat_response = requests.post(chat_url, json=chat_payload, headers=headers, timeout=60)
        chat_response.raise_for_status()
        chat_result = chat_response.json()
        llm_response = chat_result.get('response', 'No response field found')
        print(f"\n    [PASS] Chatbot Response (first 500 chars):")
        print(f"    {llm_response[:500]}")
        passed += 1
    except requests.exceptions.ConnectionError:
        print(f"\n    [FAIL] Cannot connect to {chat_url}. Is chat_api.py running?")
        failed += 1
    except requests.exceptions.ReadTimeout:
        print(f"\n    [WARN] Chat API timed out (60s). Ollama may still be loading the model.")
        failed += 1
    except Exception as e:
        print(f"\n    [FAIL] Chat API Error: {e}")
        failed += 1
    
    # --- Test 3: Edge API (if accessible) ---
    print(f"\n[3] Testing Edge Dashboard API (http://127.0.0.1:8080/api/state)...")
    try:
        edge_response = requests.get("http://127.0.0.1:8080/api/state", timeout=5)
        edge_response.raise_for_status()
        edge_state = edge_response.json()
        print(f"    [PASS] Edge Health Index: {edge_state.get('health_index')}%")
        passed += 1
    except requests.exceptions.ConnectionError:
        print(f"    [SKIP] Edge API not reachable (expected if not running on this machine)")
    except Exception as e:
        print(f"    [FAIL] Edge API Error: {e}")
        failed += 1
    
    # --- Summary ---
    print(f"\n{'=' * 40}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 40}")
    
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Cloud Backend Test")
    parser.add_argument("--inference-url", type=str, default=DEFAULT_INFERENCE_URL)
    parser.add_argument("--chat-url", type=str, default=DEFAULT_CHAT_URL)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY,
                        help="X-API-Key for authenticating with cloud backend")
    args = parser.parse_args()
    
    success = run_tests(args.inference_url, args.chat_url, args.api_key)
    sys.exit(0 if success else 1)
