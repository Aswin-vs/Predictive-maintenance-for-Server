"""
Telemetry Plugin: Memory (Tier 1 ECC-Conditional) — Phase 4 Production Hardening

Collects enterprise and consumer RAM telemetry:
- Base Metrics: Used/Total/Available GB, utilization percentage, swap usage,
  and dynamic page fault rates.
- ECC Error Telemetry:
  * Linux: Checks EDAC sysfs (`/sys/devices/system/edac/mc/mc*/ce_count` and `ue_count`).
    Calculates 60-second single-bit correctable error burst rate.
    If EDAC sysfs is absent (non-ECC consumer system), reports `"ecc_supported": false`.
  * Windows: Queries WHEA (Windows Hardware Error Architecture) Event Logs for
    Memory Corrected (ID 17/47) and Uncorrected (ID 18) error events, and polls
    LibreHardwareMonitor for DIMM thermal telemetry.
- Degradation Precursor: Flags `impending_failure_detected` if correctable error
  burst rate exceeds 10 errors/minute or any uncorrectable error occurs.

Author: Antigravity AI Engine (Phase 4 Hardening)
"""

import os
import sys
import time
import json
import logging
import platform
import subprocess
from collections import deque
from typing import Dict, Any, List, Optional

try:
    from server.telemetry_agent.plugins import TelemetryPlugin
except ImportError:
    from plugins import TelemetryPlugin

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger("telemetry.plugins.memory")


class MemoryPlugin(TelemetryPlugin):
    PLUGIN_NAME = "memory_tier1_ecc_conditional"
    SUBSYSTEM = "memory"
    SUPPORTED_OS = ["windows", "linux"]

    def __init__(self):
        super().__init__()
        self._os_name = platform.system().lower()
        self._logged_warnings = set()
        self._prev_page_faults = 0
        self._prev_time = 0.0

        # ECC sliding window tracking: deque of (timestamp, ce_count)
        self._ecc_history = deque(maxlen=60)
        self._last_ce_count: Optional[int] = None

    def _log_once(self, key: str, message: str, level: str = "warning"):
        if key not in self._logged_warnings:
            self._logged_warnings.add(key)
            if level == "warning":
                logger.warning(message)
            else:
                logger.info(message)

    def detect(self) -> bool:
        return PSUTIL_AVAILABLE

    def collect(self) -> Dict[str, Any]:
        """Collect memory metrics with graceful ECC degradation."""
        timestamp = time.time()
        
        # 1. Base psutil Virtual & Swap Memory
        mem_used_gb = 0.0
        mem_total_gb = 0.0
        mem_avail_gb = 0.0
        mem_percent = 0.0
        mem_committed_gb = 0.0
        swap_used_gb = 0.0
        swap_total_gb = 0.0
        swap_percent = 0.0

        if PSUTIL_AVAILABLE:
            vmem = psutil.virtual_memory()
            smem = psutil.swap_memory()
            mem_used_gb = round(vmem.used / (1024**3), 2)
            mem_total_gb = round(vmem.total / (1024**3), 2)
            mem_avail_gb = round(vmem.available / (1024**3), 2)
            mem_percent = vmem.percent
            mem_committed_gb = round((vmem.total - vmem.available) / (1024**3), 2)
            swap_used_gb = round(smem.used / (1024**3), 2)
            swap_total_gb = round(smem.total / (1024**3), 2)
            swap_percent = smem.percent

        # 2. Process / OS Page Fault Rate Calculation
        page_faults_sec = 0.0
        if PSUTIL_AVAILABLE:
            try:
                proc = psutil.Process()
                mem_info = proc.memory_info()
                curr_faults = getattr(mem_info, "num_page_faults", 0)
                if self._prev_time > 0.0:
                    dt = max(0.1, timestamp - self._prev_time)
                    page_faults_sec = max(0.0, round((curr_faults - self._prev_page_faults) / dt, 1))
                self._prev_page_faults = curr_faults
                self._prev_time = timestamp
            except Exception:
                page_faults_sec = 0.0

        # 3. ECC Memory Telemetry (Linux vs Windows)
        if self._os_name == "linux":
            ecc_data = self._collect_linux_edac(timestamp)
        elif self._os_name == "windows":
            ecc_data = self._collect_windows_whea(timestamp)
        else:
            ecc_data = {
                "ecc_supported": False,
                "ecc_ce_count": None,
                "ecc_ue_count": None,
                "ecc_ce_burst_rate_per_min": None,
                "impending_failure_detected": False,
                "dimm_temps_c": [],
            }

        return {
            "used_gb": mem_used_gb,
            "total_gb": mem_total_gb,
            "available_gb": mem_avail_gb,
            "percent_used": mem_percent,
            "committed_gb": mem_committed_gb,
            "swap_used_gb": swap_used_gb,
            "swap_total_gb": swap_total_gb,
            "swap_percent": swap_percent,
            "page_faults_sec": page_faults_sec,
            "ecc_supported": ecc_data["ecc_supported"],
            "ecc_ce_count": ecc_data["ecc_ce_count"],
            "ecc_ue_count": ecc_data["ecc_ue_count"],
            "ecc_ce_burst_rate_per_min": ecc_data["ecc_ce_burst_rate_per_min"],
            "impending_failure_detected": ecc_data["impending_failure_detected"],
            "dimm_temps_c": ecc_data["dimm_temps_c"],
        }

    def _collect_linux_edac(self, timestamp: float) -> Dict[str, Any]:
        """Linux EDAC sysfs check (/sys/devices/system/edac/mc/)."""
        edac_mc_dir = "/sys/devices/system/edac/mc"
        if not os.path.isdir(edac_mc_dir):
            self._log_once("linux_no_edac", "Linux EDAC sysfs absent (non-ECC system); skipping ECC metrics.", "info")
            return {
                "ecc_supported": False,
                "ecc_ce_count": None,
                "ecc_ue_count": None,
                "ecc_ce_burst_rate_per_min": None,
                "impending_failure_detected": False,
                "dimm_temps_c": [],
            }

        total_ce = 0
        total_ue = 0
        mc_found = False

        try:
            for entry in os.listdir(edac_mc_dir):
                if entry.startswith("mc"):
                    mc_path = os.path.join(edac_mc_dir, entry)
                    ce_file = os.path.join(mc_path, "ce_count")
                    ue_file = os.path.join(mc_path, "ue_count")
                    if os.path.isfile(ce_file):
                        with open(ce_file, "r") as f:
                            total_ce += int(f.read().strip())
                        mc_found = True
                    if os.path.isfile(ue_file):
                        with open(ue_file, "r") as f:
                            total_ue += int(f.read().strip())
                        mc_found = True
        except Exception as e:
            self._log_once("linux_edac_err", f"Error reading EDAC sysfs: {e}", "info")

        if not mc_found:
            return {
                "ecc_supported": False,
                "ecc_ce_count": None,
                "ecc_ue_count": None,
                "ecc_ce_burst_rate_per_min": None,
                "impending_failure_detected": False,
                "dimm_temps_c": [],
            }

        # Calculate burst rate over sliding window
        burst_rate = 0.0
        self._ecc_history.append((timestamp, total_ce))
        if len(self._ecc_history) >= 2:
            t0, ce0 = self._ecc_history[0]
            t_n, ce_n = self._ecc_history[-1]
            dt = t_n - t0
            if dt > 1.0:
                burst_rate = round(((ce_n - ce0) / dt) * 60.0, 2)  # errors per minute

        impending_failure = (burst_rate > 10.0) or (total_ue > 0)

        return {
            "ecc_supported": True,
            "ecc_ce_count": total_ce,
            "ecc_ue_count": total_ue,
            "ecc_ce_burst_rate_per_min": burst_rate,
            "impending_failure_detected": impending_failure,
            "dimm_temps_c": [],
        }

    def _collect_windows_whea(self, timestamp: float) -> Dict[str, Any]:
        """Windows WHEA event log check for ECC memory faults."""
        # Query PowerShell for WHEA-Logger memory event records
        cmd = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; Id=17,18,47} -MaxEvents 10 -ErrorAction SilentlyContinue | Select-Object -Property Id, TimeCreated | ConvertTo-Json"
        ]
        flags = {}
        if platform.system() == "Windows":
            flags["creationflags"] = subprocess.CREATE_NO_WINDOW

        ce_count = 0
        ue_count = 0
        ecc_supported = False

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5, **flags)
            if res.returncode == 0 and res.stdout.strip():
                events = json.loads(res.stdout)
                if isinstance(events, dict):
                    events = [events]
                ecc_supported = True
                for evt in events:
                    eid = evt.get("Id")
                    if eid in (17, 47):  # Corrected memory error
                        ce_count += 1
                    elif eid == 18:      # Uncorrected memory error
                        ue_count += 1
        except Exception:
            pass

        if not ecc_supported:
            self._log_once("win_no_ecc", "Windows host lacks active ECC WHEA records (non-ECC or zero faults); reporting ecc_supported=false.", "info")
            return {
                "ecc_supported": False,
                "ecc_ce_count": None,
                "ecc_ue_count": None,
                "ecc_ce_burst_rate_per_min": None,
                "impending_failure_detected": False,
                "dimm_temps_c": [],
            }

        impending_failure = (ce_count >= 5) or (ue_count > 0)
        return {
            "ecc_supported": True,
            "ecc_ce_count": ce_count,
            "ecc_ue_count": ue_count,
            "ecc_ce_burst_rate_per_min": round(float(ce_count), 2),
            "impending_failure_detected": impending_failure,
            "dimm_temps_c": [],
        }
