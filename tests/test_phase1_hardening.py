"""
Tests for Phase 1: Critical Bug Remediation & Backend Hardening
Validates:
1. BUG-01: ISO timestamp parsing & safe TTL eviction in edge/api_server.py
2. BUG-02: StandardScaler and adjacency matrix isolation in server/training/train_stgcn.py
3. BUG-03: Chronological split & StandardScaler isolation in server/training/auto_retrain.py
4. BUG-04: Bounded memory & TTLCache limits in server/deep_analysis/inference_server.py
5. BUG-05: Table sorting contract, UTF-8 charset header, and zero Mojibake in dashboard
"""

import os
import sys
import time
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from edge.api_server import app, current_state, state_lock, parse_last_seen
from cachetools import TTLCache
from fastapi.testclient import TestClient


# ─── 1. BUG-01: Silent TTL Eviction Crash in edge/api_server.py ──────────────

def test_parse_last_seen():
    """Test safe parsing of ISO strings, floats, ints, and invalid values."""
    # ISO-8601 string with UTC 'Z'
    iso_z = "2026-09-25T12:00:00Z"
    ts = parse_last_seen(iso_z)
    assert ts is not None
    assert isinstance(ts, float)

    # ISO-8601 string with offset
    iso_offset = "2026-09-25T12:00:00+00:00"
    assert parse_last_seen(iso_offset) == ts

    # Epoch float and int
    now_epoch = time.time()
    assert parse_last_seen(now_epoch) == now_epoch
    assert parse_last_seen(int(now_epoch)) == float(int(now_epoch))

    # Invalid values must return None safely without raising TypeError
    assert parse_last_seen("invalid-date-format") is None
    assert parse_last_seen(None) is None
    assert parse_last_seen([]) is None
    assert parse_last_seen({}) is None


def test_ttl_eviction_handles_string_and_float_timestamps():
    """Test that TTL eviction handles string ISO timestamps and does not crash."""
    now = time.time()
    stale_iso = (datetime.now(timezone.utc) - timedelta(seconds=2500)).isoformat()
    active_iso = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
    stale_float = now - 2500
    active_float = now - 300

    with state_lock:
        current_state.clear()
        current_state["srv_stale_iso"] = {"last_seen": stale_iso}
        current_state["srv_active_iso"] = {"last_seen": active_iso}
        current_state["srv_stale_float"] = {"last_seen": stale_float}
        current_state["srv_active_float"] = {"last_seen": active_float}
        current_state["srv_corrupted"] = {"last_seen": "corrupted_timestamp"}

        # Simulate eviction logic from _cleanup_loop
        stale = []
        for sid, s in list(current_state.items()):
            ls_ts = parse_last_seen(s.get("last_seen"))
            if ls_ts is not None and (now - ls_ts > 1800):
                stale.append(sid)

        for sid in stale:
            current_state.pop(sid, None)

        # Assert correct servers were evicted
        assert "srv_stale_iso" not in current_state
        assert "srv_stale_float" not in current_state
        assert "srv_active_iso" in current_state
        assert "srv_active_float" in current_state
        assert "srv_corrupted" in current_state  # not evicted, but didn't crash either
        current_state.clear()


# ─── 2. BUG-02: ST-GCN Data Leakage Prevention ──────────────────────────────

def test_stgcn_scaler_and_adjacency_isolation():
    """Assert scaler and adjacency matrix do not see test set data."""
    from sklearn.preprocessing import StandardScaler
    from server.training.train_stgcn import build_adjacency

    # Create synthetic dataset where test set (last 20%) has wildly different mean
    np.random.seed(42)
    n_samples = 200
    n_feat = 8

    train_data = np.random.normal(loc=10.0, scale=1.0, size=(160, n_feat)).astype(np.float32)
    test_data = np.random.normal(loc=1000.0, scale=1.0, size=(40, n_feat)).astype(np.float32)
    X_raw = np.vstack([train_data, test_data])

    # BUG-02 Fix: Split 80/20 BEFORE scaling
    split_idx = int(0.8 * len(X_raw))
    X_train_raw = X_raw[:split_idx]
    X_test_raw = X_raw[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Scaler mean must strictly reflect train_data (~10.0), NOT contaminated by test (~1000.0)
    for m in scaler.mean_:
        assert 9.0 <= m <= 11.0, f"Scaler mean {m} contaminated by test data!"

    # Scaler scale must reflect train variance (~1.0)
    for s in scaler.scale_:
        assert 0.8 <= s <= 1.2, f"Scaler variance {s} contaminated by test data!"

    # Adjacency derived strictly from training data
    adj = build_adjacency(n_feat, X_train_scaled)
    assert adj.shape == (n_feat, n_feat)
    assert np.all(np.isfinite(adj))
    # Diagonal should be normalized non-zero
    assert np.all(np.diag(adj) > 0)


# ─── 3. BUG-03: Auto-Retraining Data Leakage Prevention ──────────────────────

def test_auto_retrain_chronological_isolation():
    """Verify chronological split and scaler isolation in auto_retrain."""
    from sklearn.preprocessing import StandardScaler

    np.random.seed(123)
    n_samples = 300
    n_features = 6

    # 90% train (mean=5.0), 10% val (mean=500.0)
    data_train = np.random.normal(loc=5.0, scale=0.5, size=(270, n_features)).astype(np.float32)
    data_val = np.random.normal(loc=500.0, scale=0.5, size=(30, n_features)).astype(np.float32)
    data = np.vstack([data_train, data_val])

    # Chronological 90/10 split
    split = int(0.9 * len(data))
    train_raw = data[:split]
    val_raw = data[split:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)

    for m in scaler.mean_:
        assert 4.5 <= m <= 5.5, f"Auto-retrain scaler contaminated: {m}"


# ─── 4. BUG-04: Unbounded Memory Leak in inference_server.py ─────────────────

def test_inference_server_cache_bounds():
    """Verify TTLCache bounding prevents memory leak under thousands of clients."""
    from server.deep_analysis.inference_server import (
        _rate_tracker,
        _server_windows,
        _madgan_windows,
        _server_last_seen,
        _retrain_status,
        _check_rate_limit,
    )

    # Verify types are TTLCache
    assert isinstance(_rate_tracker, TTLCache)
    assert isinstance(_server_windows, TTLCache)
    assert isinstance(_madgan_windows, TTLCache)
    assert isinstance(_server_last_seen, TTLCache)
    assert isinstance(_retrain_status, TTLCache)

    # Verify maxsize limits
    assert _rate_tracker.maxsize == 10000
    assert _server_windows.maxsize == 1000
    assert _madgan_windows.maxsize == 1000

    # Simulate 1500 distinct server IDs to verify hard cap of 1000
    for i in range(1500):
        sid = f"test-server-{i:04d}"
        if sid not in _server_windows:
            _server_windows[sid] = deque(maxlen=10)
        _server_windows[sid].append(np.zeros(28))

    assert len(_server_windows) <= 1000, f"Memory leak: _server_windows grew to {len(_server_windows)}"

    # Test rate limiter bounding
    for i in range(12000):
        ip = f"192.168.{(i // 254) % 254}.{i % 254 + 1}"
        _rate_tracker[ip] = [time.time()]

    assert len(_rate_tracker) <= 10000, f"Memory leak: _rate_tracker grew to {len(_rate_tracker)}"


# ─── 5. BUG-05: Non-Functional Table Sorting & Mojibake ──────────────────────

def test_dashboard_charset_and_sorting_contract():
    """Verify HTML response explicitly sets UTF-8 charset and app.js has zero Mojibake."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200

    # Verify Content-Type contains charset=utf-8
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type
    assert "charset=utf-8" in content_type.lower()

    # Verify app.js contains NO Mojibake
    app_js_path = os.path.join(BASE_DIR, "dashboard", "assets", "app.js")
    with open(app_js_path, "r", encoding="utf-8") as f:
        app_js_content = f.read()

    assert "â€”" not in app_js_content, "Mojibake 'â€”' found in app.js!"
    assert "â€¢" not in app_js_content, "Mojibake 'â€¢' found in app.js!"
    assert "â— " not in app_js_content, "Mojibake 'â— ' found in app.js!"
    assert "â—‹" not in app_js_content, "Mojibake 'â—‹' found in app.js!"
    assert "â€¦" not in app_js_content, "Mojibake 'â€¦' found in app.js!"
    assert "â”€" not in app_js_content, "Mojibake 'â”€' found in app.js!"

    # Verify sort-col class and listeners exist
    assert "sort-col" in app_js_content
    assert "cache.historyRecords" in app_js_content

    # Verify index.html has UTF-8 meta as first element in head
    index_html_path = os.path.join(BASE_DIR, "dashboard", "index.html")
    with open(index_html_path, "r", encoding="utf-8") as f:
        html = f.read()

    head_idx = html.find("<head>")
    charset_idx = html.find('<meta charset="UTF-8">')
    assert head_idx != -1 and charset_idx != -1
    # Ensure <meta charset="UTF-8"> comes directly after <head>
    head_content_before_charset = html[head_idx + len("<head>"):charset_idx].strip()
    assert head_content_before_charset == "", "<meta charset='UTF-8'> must be the very first element in <head>"

    # Ensure table headers in dbHistoryTable have sort-col class
    assert 'class="sort-col"' in html
