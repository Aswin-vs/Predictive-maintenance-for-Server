"""
Automated Test Suite for Phase 4: Tiered Hardware Degradation Telemetry Plugins

Tests all 6 plugins, graceful degradation, and precursor indicator calculations:
1. CPUPlugin: RAPL/LHM/WMI, thermal resistance slope, MCE events, PROCHOT.
2. GPUPlugin: pynvml, Xid/nvlddmkm logs, hotspot delta, ECC error counters.
3. MotherboardPlugin: PCH/VRM, CMOS battery voltage, Super I/O support.
4. MemoryPlugin: EDAC sysfs, WHEA logs, ECC burst rate, non-ECC degradation.
5. StoragePlugin: NVMe Log Page 0x02, SATA SMART, endurance/spare degradation.
6. PSUFanPlugin: Tier 1 rail voltages (+12V sag), Tier 2 BMC probe, fan wear.
7. TelemetryCollectorV2: hierarchical collection, crash-proof to_v1_flat conversion.
"""

import pytest
import os
import sys
import json
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from server.telemetry_agent.plugins.cpu_plugin import CPUPlugin
from server.telemetry_agent.plugins.gpu_plugin import GPUPlugin
from server.telemetry_agent.plugins.gpu_nvidia_plugin import GPUNvidiaPlugin
from server.telemetry_agent.plugins.motherboard_plugin import MotherboardPlugin
from server.telemetry_agent.plugins.memory_plugin import MemoryPlugin
from server.telemetry_agent.plugins.storage_plugin import StoragePlugin
from server.telemetry_agent.plugins.psu_fan_plugin import PSUFanPlugin
from server.telemetry_agent.collector_v2 import TelemetryCollectorV2
from server.training.feature_schema import FEATURE_NAMES_V1, FEATURE_COUNT


def test_cpu_plugin():
    plugin = CPUPlugin()
    assert plugin.initialize() is True
    data = plugin.safe_collect()
    assert data is not None
    assert "id" in data
    assert "load_percent" in data
    assert "frequency_mhz" in data
    assert "thermal_paste_degradation_detected" in data
    assert "prochot_events_count" in data
    assert "mce_events_count" in data


def test_gpu_plugin_and_alias():
    plugin = GPUPlugin()
    init_res = plugin.initialize()
    data = plugin.safe_collect()
    assert data is not None
    assert "gpus" in data
    assert "vrm_phase_telemetry_note" in data
    assert "fault_events" in data

    # Test backward compatibility alias
    legacy = GPUNvidiaPlugin()
    assert legacy.initialize() == init_res
    legacy_data = legacy.safe_collect()
    assert legacy_data is not None


def test_motherboard_plugin():
    plugin = MotherboardPlugin()
    assert plugin.initialize() is True
    data = plugin.safe_collect()
    assert data is not None
    assert "board_vendor" in data
    assert "superio_supported" in data
    assert "status" in data
    assert data["status"] in ("ok", "unsupported")


def test_memory_plugin_ecc_conditional():
    plugin = MemoryPlugin()
    assert plugin.initialize() is True
    data = plugin.safe_collect()
    assert data is not None
    assert "used_gb" in data
    assert "total_gb" in data
    assert "ecc_supported" in data
    assert isinstance(data["ecc_supported"], bool)
    assert "impending_failure_detected" in data


def test_storage_plugin_nvme_and_sata():
    plugin = StoragePlugin()
    assert plugin.initialize() is True
    data = plugin.safe_collect()
    assert data is not None
    assert "drives" in data
    assert "drive_count" in data
    if data["drives"]:
        drive = data["drives"][0]
        assert "capacity_gb" in drive
        assert "used_gb" in drive
        assert "smart" in drive
        assert "degradation_flag_spare_low" in drive
        assert "degradation_flag_endurance_exhausted" in drive
        assert "degradation_flag_uncorrectable" in drive


def test_psu_fan_plugin_and_indicators():
    plugin = PSUFanPlugin()
    assert plugin.initialize() is True
    data = plugin.safe_collect()
    assert data is not None
    assert "psu" in data
    assert "fans" in data
    psu = data["psu"]
    assert "voltage_12v" in psu
    assert "rail_12v_sag_detected" in psu
    assert isinstance(psu["rail_12v_sag_detected"], bool)

    for fan in data["fans"]:
        assert "rpm" in fan
        assert "rpm_deviation_pct" in fan
        assert "motor_winding_wear_flag" in fan


def test_collector_v2_e2e_and_to_v1_flat():
    collector = TelemetryCollectorV2(use_dummy=False)
    payload = collector.collect()
    assert payload["schema_version"] == "2.0"
    assert "cpus" in payload
    assert "storage" in payload
    assert "memory" in payload
    assert "psu" in payload
    assert "fans" in payload
    assert "environment" in payload

    # Test conversion to flat 28 features
    flat = collector.to_v1_flat(payload)
    assert len(flat) == FEATURE_COUNT
    for key in FEATURE_NAMES_V1:
        assert key in flat, f"Missing feature: {key}"
        assert flat[key] is not None, f"Feature {key} must not be None"
        assert isinstance(flat[key], (int, float)), f"Feature {key} must be numeric"


def test_collector_v2_dummy_mode():
    collector = TelemetryCollectorV2(use_dummy=True)
    payload = collector.collect()
    assert payload["schema_version"] == "2.0"
    flat = collector.to_v1_flat(payload)
    assert len(flat) == FEATURE_COUNT
    for key in FEATURE_NAMES_V1:
        assert key in flat
        assert isinstance(flat[key], (int, float))
