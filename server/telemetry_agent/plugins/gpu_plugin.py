"""
Telemetry Plugin: GPU (Tier 1 Cross-Platform) — Phase 4 Production Hardening

Collects enterprise and consumer GPU telemetry across NVIDIA and AMD hardware:
- NVIDIA Primary: High-speed native `pynvml` / `nvidia-ml-py` query with fallback
  to `nvidia-smi` subprocess. Captures core temperature, junction/hotspot temperature,
  board power draw, VRAM usage, utilization rates, and ECC memory error counters.
- Fault Tracking:
  * Linux: Parses kernel log (`journalctl -k`) for `NVRM: Xid` events (Xid 79, 63, 48, 45, 31).
  * Windows: Queries System Event Log for Source=`nvlddmkm` via PowerShell / win32evtlog.
- AMD Secondary: Fallback to `rocm-smi` for AMD Radeon / Instinct accelerators.
- Physical Degradation Indicator: Calculates Hotspot Delta (T_junction - T_edge).
  A delta exceeding 30°C flags physical thermal pad deterioration / pump-out.
- VRM Current Proxy Note: Per-VRM-phase current telemetry is unavailable without
  proprietary vendor SDKs; board-level power draw is used as a proxy.

Author: Antigravity AI Engine (Phase 4 Hardening)
"""

import os
import sys
import json
import logging
import platform
import subprocess
from typing import Dict, Any, List, Optional

try:
    from server.telemetry_agent.plugins import TelemetryPlugin
except ImportError:
    from plugins import TelemetryPlugin

# Try pynvml / nvidia-ml-py
PYNVML_AVAILABLE = False
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    try:
        import nvidia_ml_py as pynvml
        PYNVML_AVAILABLE = True
    except ImportError:
        pynvml = None

logger = logging.getLogger("telemetry.plugins.gpu")


def _run_cmd_quiet(cmd: List[str], timeout: float = 3.0) -> Optional[str]:
    """Run a subprocess quietly without popping console windows on Windows."""
    try:
        flags = {}
        if platform.system() == "Windows":
            flags["creationflags"] = subprocess.CREATE_NO_WINDOW
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **flags)
        if res.returncode == 0:
            return res.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


class GPUPlugin(TelemetryPlugin):
    PLUGIN_NAME = "gpu_tier1_cross_platform"
    SUBSYSTEM = "gpu"
    SUPPORTED_OS = ["windows", "linux"]

    VRM_PROXY_NOTE = "Per-VRM-phase current telemetry is unavailable without proprietary vendor SDKs; board-level power draw is used as proxy."

    def __init__(self):
        super().__init__()
        self._os_name = platform.system().lower()
        self._logged_warnings = set()
        self._nvml_initialized = False
        self._gpu_vendor = "none"  # "nvidia", "amd", "none"
        self._gpu_count = 0

    def _log_once(self, key: str, message: str, level: str = "warning"):
        if key not in self._logged_warnings:
            self._logged_warnings.add(key)
            if level == "warning":
                logger.warning(message)
            else:
                logger.info(message)

    def detect(self) -> bool:
        """Probe for NVIDIA NVML or AMD ROCm GPUs."""
        # 1. Probe NVIDIA NVML
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_count = pynvml.nvmlDeviceGetCount()
                if self._gpu_count > 0:
                    self._nvml_initialized = True
                    self._gpu_vendor = "nvidia"
                    return True
            except Exception as e:
                self._log_once("nvml_init_fail", f"NVML init failed ({e}); falling back to nvidia-smi", "info")

        # 2. Probe NVIDIA CLI (nvidia-smi)
        out = _run_cmd_quiet(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"])
        if out:
            try:
                self._gpu_count = int(out.strip().split("\n")[0])
                if self._gpu_count > 0:
                    self._gpu_vendor = "nvidia"
                    return True
            except ValueError:
                pass

        # 3. Probe AMD ROCm (rocm-smi)
        rocm_out = _run_cmd_quiet(["rocm-smi", "--showid"])
        if rocm_out and "GPU[" in rocm_out:
            self._gpu_vendor = "amd"
            self._gpu_count = rocm_out.count("GPU[")
            return True

        self._log_once("no_gpu_detected", "No dedicated NVIDIA or AMD GPU detected on host.", "info")
        return False

    def collect(self) -> Dict[str, Any]:
        """Collect GPU metrics, fault logs, and degradation indicators."""
        if self._gpu_vendor == "nvidia":
            gpus = self._collect_nvidia()
        elif self._gpu_vendor == "amd":
            gpus = self._collect_amd()
        else:
            gpus = []

        # Query OS-specific GPU fault events (Xid on Linux, nvlddmkm on Windows)
        fault_events = self._collect_gpu_fault_events()

        return {
            "gpus": gpus,
            "gpu_count": len(gpus),
            "gpu_vendor": self._gpu_vendor,
            "vrm_phase_telemetry_note": self.VRM_PROXY_NOTE,
            "fault_events": fault_events,
        }

    def _collect_nvidia(self) -> List[Dict[str, Any]]:
        """Collect metrics from all detected NVIDIA GPUs."""
        gpus = []

        # Try NVML first if initialized
        if self._nvml_initialized and PYNVML_AVAILABLE:
            try:
                for i in range(self._gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_data = self._read_nvml_device(handle, i)
                    gpus.append(gpu_data)
                return gpus
            except Exception as e:
                self._log_once("nvml_read_err", f"NVML device read failed: {e}; falling back to nvidia-smi", "info")

        # Fallback: nvidia-smi subprocess
        return self._collect_nvidia_smi()

    def _read_nvml_device(self, handle, index: int) -> Dict[str, Any]:
        """Query detailed metrics from NVML device handle."""
        name = "NVIDIA GPU"
        try:
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        except Exception:
            pass

        # Temperature: Core (Edge)
        temp_core = None
        try:
            temp_core = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass

        # Temperature: Junction / Hotspot (Sensor 1 or 2 if supported)
        temp_junction = None
        try:
            # 1 corresponds to NVML_TEMPERATURE_GPU_HOTSPOT on architectures that support it
            temp_junction = float(pynvml.nvmlDeviceGetTemperature(handle, 1))
        except Exception:
            pass

        # Power draw
        power_w = None
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = round(power_mw / 1000.0, 2)
        except Exception:
            pass

        # Utilization & Clocks
        sm_util = None
        mem_util = None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            sm_util = util.gpu
            mem_util = util.memory
        except Exception:
            pass

        clock_mhz = None
        try:
            clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        except Exception:
            pass

        # VRAM
        vram_used_mb = 0
        vram_total_mb = 0
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_mb = int(mem_info.used // (1024 * 1024))
            vram_total_mb = int(mem_info.total // (1024 * 1024))
        except Exception:
            pass

        # Fan speed
        fan_speed_pct = None
        try:
            fan_speed_pct = pynvml.nvmlDeviceGetFanSpeed(handle)
        except Exception:
            pass

        # PCIe link info
        pcie_width = None
        pcie_gen = None
        try:
            pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        except Exception:
            pass

        # ECC Error Counters
        ecc_supported = False
        ecc_sbe = None
        ecc_dbe = None
        try:
            ecc_mode = pynvml.nvmlDeviceGetEccMode(handle)
            # ECC mode tuple: (current_mode, pending_mode)
            ecc_supported = bool(ecc_mode[0] if isinstance(ecc_mode, tuple) else ecc_mode)
            if ecc_supported:
                # Corrected Single-Bit Errors (SBE)
                ecc_sbe = pynvml.nvmlDeviceGetMemoryErrorCounter(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    pynvml.NVML_VOLATILE_ECC,
                    pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
                )
                # Uncorrected Double-Bit Errors (DBE)
                ecc_dbe = pynvml.nvmlDeviceGetMemoryErrorCounter(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_VOLATILE_ECC,
                    pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
                )
        except Exception:
            ecc_supported = False

        # Physical Precursor: Hotspot Delta (T_junction - T_edge)
        hotspot_delta = None
        thermal_pad_degradation = False
        if temp_junction is not None and temp_core is not None:
            hotspot_delta = round(temp_junction - temp_core, 1)
            if hotspot_delta > 30.0:
                thermal_pad_degradation = True

        return {
            "id": f"GPU{index}",
            "name": name,
            "vendor": "NVIDIA",
            "temp_core_c": temp_core,
            "temp_junction_c": temp_junction,
            "hotspot_delta_c": hotspot_delta,
            "thermal_pad_degradation_detected": thermal_pad_degradation,
            "power_draw_w": power_w,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "clock_mhz": clock_mhz,
            "sm_utilization_pct": sm_util,
            "mem_utilization_pct": mem_util,
            "fan_speed_pct": fan_speed_pct,
            "pcie_link_width": pcie_width,
            "pcie_gen": pcie_gen,
            "ecc_supported": ecc_supported,
            "ecc_sbe_count": ecc_sbe,
            "ecc_dbe_count": ecc_dbe,
        }

    def _collect_nvidia_smi(self) -> List[Dict[str, Any]]:
        """Fallback querying nvidia-smi subprocess."""
        gpus = []
        fields = (
            "index,name,temperature.gpu,temperature.memory,power.draw,"
            "memory.used,memory.total,clocks.current.graphics,"
            "utilization.gpu,utilization.memory,fan.speed,"
            "pcie.link.width.current,pcie.link.gen.current"
        )
        out = _run_cmd_quiet(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        if not out:
            return gpus

        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 11:
                idx = parts[0]
                temp_core = self._safe_float(parts[2])
                temp_junc = self._safe_float(parts[3])
                
                hotspot_delta = None
                thermal_pad_deg = False
                if temp_junc is not None and temp_core is not None:
                    hotspot_delta = round(temp_junc - temp_core, 1)
                    if hotspot_delta > 30.0:
                        thermal_pad_deg = True

                gpu = {
                    "id": f"GPU{idx}",
                    "name": parts[1] if len(parts) > 1 else "NVIDIA GPU",
                    "vendor": "NVIDIA",
                    "temp_core_c": temp_core,
                    "temp_junction_c": temp_junc,
                    "hotspot_delta_c": hotspot_delta,
                    "thermal_pad_degradation_detected": thermal_pad_deg,
                    "power_draw_w": self._safe_float(parts[4]),
                    "vram_used_mb": self._safe_int(parts[5]),
                    "vram_total_mb": self._safe_int(parts[6]),
                    "clock_mhz": self._safe_int(parts[7]),
                    "sm_utilization_pct": self._safe_int(parts[8]),
                    "mem_utilization_pct": self._safe_int(parts[9]),
                    "fan_speed_pct": self._safe_int(parts[10]),
                    "pcie_link_width": self._safe_int(parts[11]) if len(parts) > 11 else None,
                    "pcie_gen": self._safe_int(parts[12]) if len(parts) > 12 else None,
                    "ecc_supported": False,
                    "ecc_sbe_count": None,
                    "ecc_dbe_count": None,
                }
                gpus.append(gpu)

        # Separate query for ECC if supported
        ecc_out = _run_cmd_quiet([
            "nvidia-smi",
            "--query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits"
        ])
        if ecc_out:
            for line in ecc_out.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx = self._safe_int(parts[0])
                    if idx is not None and idx < len(gpus):
                        sbe = self._safe_int(parts[1])
                        dbe = self._safe_int(parts[2])
                        if sbe is not None or dbe is not None:
                            gpus[idx]["ecc_supported"] = True
                            gpus[idx]["ecc_sbe_count"] = sbe
                            gpus[idx]["ecc_dbe_count"] = dbe

        return gpus

    def _collect_amd(self) -> List[Dict[str, Any]]:
        """Collect telemetry from AMD GPUs using rocm-smi."""
        gpus = []
        out = _run_cmd_quiet(["rocm-smi", "--showtemp", "--showuse", "--showpower", "--json"])
        if not out:
            return gpus

        try:
            data = json.loads(out)
            for card_id, card_data in data.items():
                if not card_id.startswith("card"):
                    continue
                temp_edge = self._safe_float(card_data.get("Temperature (Sensor edge) (C)"))
                temp_junc = self._safe_float(card_data.get("Temperature (Sensor junction) (C)"))
                
                hotspot_delta = None
                thermal_pad_deg = False
                if temp_junc is not None and temp_edge is not None:
                    hotspot_delta = round(temp_junc - temp_edge, 1)
                    if hotspot_delta > 30.0:
                        thermal_pad_deg = True

                gpu = {
                    "id": card_id,
                    "name": "AMD Radeon/Instinct Accelerator",
                    "vendor": "AMD",
                    "temp_core_c": temp_edge,
                    "temp_junction_c": temp_junc,
                    "hotspot_delta_c": hotspot_delta,
                    "thermal_pad_degradation_detected": thermal_pad_deg,
                    "power_draw_w": self._safe_float(card_data.get("Average Graphics Package Power (W)")),
                    "vram_used_mb": None,
                    "vram_total_mb": None,
                    "clock_mhz": None,
                    "sm_utilization_pct": self._safe_int(card_data.get("GPU use (%)")),
                    "mem_utilization_pct": self._safe_int(card_data.get("GPU Memory use (%)")),
                    "fan_speed_pct": self._safe_int(card_data.get("Fan speed (%)")),
                    "pcie_link_width": None,
                    "pcie_gen": None,
                    "ecc_supported": False,
                    "ecc_sbe_count": None,
                    "ecc_dbe_count": None,
                }
                gpus.append(gpu)
        except Exception as e:
            self._log_once("rocm_parse_err", f"Failed to parse rocm-smi output: {e}", "info")

        return gpus

    def _collect_gpu_fault_events(self) -> Dict[str, Any]:
        """Track GPU driver crashes, Xid events, and reset events."""
        fault_summary = {
            "source": "none",
            "xid_events": [],
            "fatal_bus_drop_detected": False,      # Xid 79
            "ecc_row_remap_exhausted": False,      # Xid 63
            "ecc_double_bit_detected": False,      # Xid 48
            "recent_error_count": 0,
        }

        if self._os_name == "linux":
            # Query Linux journalctl for NVRM: Xid
            out = _run_cmd_quiet(["journalctl", "-k", "--grep=NVRM: Xid", "--no-pager", "-n", "10", "--output=cat"])
            if out:
                fault_summary["source"] = "journalctl"
                for line in out.strip().split("\n"):
                    if "Xid" in line:
                        fault_summary["xid_events"].append(line.strip())
                        if "79" in line:
                            fault_summary["fatal_bus_drop_detected"] = True
                        if "63" in line:
                            fault_summary["ecc_row_remap_exhausted"] = True
                        if "48" in line:
                            fault_summary["ecc_double_bit_detected"] = True
                fault_summary["recent_error_count"] = len(fault_summary["xid_events"])

        elif self._os_name == "windows":
            # Query Windows System Event Log for nvlddmkm
            cmd = [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'} -MaxEvents 5 -ErrorAction SilentlyContinue | Select-Object -Property Id, TimeCreated | ConvertTo-Json"
            ]
            out = _run_cmd_quiet(cmd, timeout=2.5)
            if out:
                fault_summary["source"] = "windows_event_log"
                try:
                    events = json.loads(out)
                    if isinstance(events, dict):
                        events = [events]
                    for evt in events:
                        evt_id = evt.get("Id")
                        fault_summary["xid_events"].append(f"nvlddmkm Event ID {evt_id}")
                        if evt_id == 153 or evt_id == 79:
                            # 153: The description for Event ID 153 from source nvlddmkm cannot be found (GPU Reset)
                            fault_summary["fatal_bus_drop_detected"] = True
                    fault_summary["recent_error_count"] = len(events)
                except Exception:
                    pass

        return fault_summary

    def _safe_float(self, val: Any) -> Optional[float]:
        try:
            if val is None or val in ("[N/A]", "N/A", "[Not Supported]", "None", ""):
                return None
            return float(str(val).replace("W", "").replace("C", "").replace("%", "").strip())
        except (ValueError, TypeError):
            return None

    def _safe_int(self, val: Any) -> Optional[int]:
        try:
            if val is None or val in ("[N/A]", "N/A", "[Not Supported]", "None", ""):
                return None
            return int(float(str(val).replace("MB", "").replace("%", "").strip()))
        except (ValueError, TypeError):
            return None
