"""
Telemetry Plugin: Motherboard & Chipset (Tier 1 Cross-Platform) — Phase 4 Production Hardening

Collects motherboard, chipset, and VRM thermal telemetry:
- Linux: Reads PCH (Platform Controller Hub), VRM MOSFET, and ambient system
  temperatures via `lm-sensors` / sysfs hwmon drivers.
- Windows: Polls LibreHardwareMonitor Motherboard sensor group, falling back
  to WMI for board identification.
- Graceful Degradation: Reports "unsupported" with null metrics if consumer
  motherboard lacks exposed Super I/O thermal diode interfaces.

Author: Antigravity AI Engine (Phase 4 Hardening)
"""

import os
import sys
import json
import logging
import platform
import subprocess
import urllib.request
from typing import Dict, Any, List, Optional

try:
    from server.telemetry_agent.plugins import TelemetryPlugin
except ImportError:
    from plugins import TelemetryPlugin

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    wmi = None
    WMI_AVAILABLE = False

logger = logging.getLogger("telemetry.plugins.motherboard")


class MotherboardPlugin(TelemetryPlugin):
    PLUGIN_NAME = "motherboard_tier1"
    SUBSYSTEM = "motherboard"
    SUPPORTED_OS = ["windows", "linux"]

    def __init__(self):
        super().__init__()
        self._os_name = platform.system().lower()
        self._logged_warnings = set()
        self._vendor = "Unknown"
        self._model = "Generic Motherboard"
        self._bios_version = "Unknown"

    def _log_once(self, key: str, message: str, level: str = "warning"):
        if key not in self._logged_warnings:
            self._logged_warnings.add(key)
            if level == "warning":
                logger.warning(message)
            else:
                logger.info(message)

    def detect(self) -> bool:
        """Detect motherboard hardware information."""
        if self._os_name == "windows":
            if WMI_AVAILABLE:
                try:
                    w_obj = wmi.WMI()
                    for b in w_obj.Win32_BaseBoard():
                        self._vendor = getattr(b, "Manufacturer", "Unknown")
                        self._model = getattr(b, "Product", "Generic Motherboard")
                    for bios in w_obj.Win32_BIOS():
                        self._bios_version = getattr(bios, "SMBIOSBIOSVersion", "Unknown")
                    return True
                except Exception:
                    pass
            return True

        elif self._os_name == "linux":
            # Read DMI sysfs
            try:
                if os.path.exists("/sys/class/dmi/id/board_vendor"):
                    with open("/sys/class/dmi/id/board_vendor", "r") as f:
                        self._vendor = f.read().strip()
                if os.path.exists("/sys/class/dmi/id/board_name"):
                    with open("/sys/class/dmi/id/board_name", "r") as f:
                        self._model = f.read().strip()
                if os.path.exists("/sys/class/dmi/id/bios_version"):
                    with open("/sys/class/dmi/id/bios_version", "r") as f:
                        self._bios_version = f.read().strip()
            except Exception:
                pass
            return True

        return False

    def collect(self) -> Dict[str, Any]:
        """Collect motherboard temperatures, voltage, and Super I/O metrics."""
        if self._os_name == "linux":
            return self._collect_linux()
        elif self._os_name == "windows":
            return self._collect_windows()
        else:
            return {
                "board_vendor": self._vendor,
                "board_model": self._model,
                "bios_version": self._bios_version,
                "pch_temp_c": None,
                "vrm_temp_c": None,
                "system_temp_c": None,
                "cmos_battery_voltage": None,
                "superio_supported": False,
                "source": "unsupported_os",
                "status": "unsupported",
            }

    def _collect_linux(self) -> Dict[str, Any]:
        """Query lm-sensors JSON output for PCH, VRM, and Super I/O sensors."""
        out = {
            "board_vendor": self._vendor,
            "board_model": self._model,
            "bios_version": self._bios_version,
            "pch_temp_c": None,
            "vrm_temp_c": None,
            "system_temp_c": None,
            "cmos_battery_voltage": None,
            "superio_supported": False,
            "source": "lm-sensors",
            "status": "unsupported",
        }

        try:
            res = subprocess.run(["sensors", "-j"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and res.stdout.strip():
                sensors_data = json.loads(res.stdout)
                for chip_name, chip_data in sensors_data.items():
                    chip_lower = chip_name.lower()
                    
                    # Super I/O chips
                    if any(sio in chip_lower for sio in ["it87", "nct67", "w83627", "f718", "asb"]):
                        out["superio_supported"] = True

                    # PCH chipset
                    if "pch" in chip_lower:
                        out["superio_supported"] = True
                        for sensor_label, sensor_values in chip_data.items():
                            if isinstance(sensor_values, dict):
                                for key, val in sensor_values.items():
                                    if "input" in key and isinstance(val, (int, float)):
                                        out["pch_temp_c"] = float(val)

                    # Inspect sensor labels for VRM / Motherboard / System / Battery
                    for sensor_label, sensor_values in chip_data.items():
                        if not isinstance(sensor_values, dict):
                            continue
                        label_lower = sensor_label.lower()
                        for key, val in sensor_values.items():
                            if "input" in key and isinstance(val, (int, float)):
                                f_val = float(val)
                                if "vrm" in label_lower or "mos" in label_lower:
                                    out["vrm_temp_c"] = f_val
                                elif "pch" in label_lower or "chipset" in label_lower:
                                    out["pch_temp_c"] = f_val
                                elif any(k in label_lower for k in ["motherboard", "system", "temp1", "board"]):
                                    out["system_temp_c"] = f_val
                                elif "vbat" in label_lower or "battery" in label_lower:
                                    out["cmos_battery_voltage"] = f_val

                if out["pch_temp_c"] is not None or out["vrm_temp_c"] is not None or out["system_temp_c"] is not None:
                    out["status"] = "ok"
                else:
                    self._log_once("linux_sio_missing", "Motherboard lacks exposed Super I/O sensors; reporting unsupported.", "info")

        except Exception as e:
            self._log_once("linux_mb_sensors_err", f"Motherboard sensors query failed: {e}", "info")

        return out

    def _collect_windows(self) -> Dict[str, Any]:
        """Poll LibreHardwareMonitor for Motherboard group with WMI fallback."""
        out = {
            "board_vendor": self._vendor,
            "board_model": self._model,
            "bios_version": self._bios_version,
            "pch_temp_c": None,
            "vrm_temp_c": None,
            "system_temp_c": None,
            "cmos_battery_voltage": None,
            "superio_supported": False,
            "source": "unsupported",
            "status": "unsupported",
        }

        # Try LibreHardwareMonitor Remote Web Server (:8085/data.json)
        try:
            req = urllib.request.Request("http://localhost:8085/data.json", headers={"User-Agent": "AntigravityTelemetry/1.0"})
            with urllib.request.urlopen(req, timeout=0.6) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    self._parse_lhm_motherboard(payload, out)
                    if out["pch_temp_c"] is not None or out["vrm_temp_c"] is not None or out["system_temp_c"] is not None:
                        out["superio_supported"] = True
                        out["source"] = "librehardwaremonitor"
                        out["status"] = "ok"
                        return out
        except Exception:
            pass

        self._log_once("win_sio_missing", "Motherboard Super I/O sensors unavailable via LibreHardwareMonitor; reporting unsupported.", "info")
        return out

    def _parse_lhm_motherboard(self, node: Dict[str, Any], out: Dict[str, Any]):
        """Recursively search for Motherboard sensor nodes in LibreHardwareMonitor tree."""
        text = node.get("Text", "")
        children = node.get("Children", [])
        sensor_type = node.get("Type", "")
        value_str = node.get("Value", "")

        if sensor_type == "Temperature":
            try:
                val = float(value_str.replace("°C", "").replace(",", ".").strip())
                text_lower = text.lower()
                if "vrm" in text_lower or "mos" in text_lower:
                    out["vrm_temp_c"] = val
                elif "pch" in text_lower or "chipset" in text_lower:
                    out["pch_temp_c"] = val
                elif any(k in text_lower for k in ["system", "motherboard", "ambient"]):
                    out["system_temp_c"] = val
            except (ValueError, AttributeError):
                pass
        elif sensor_type == "Voltage":
            try:
                val = float(value_str.replace("V", "").replace(",", ".").strip())
                if "vbat" in text.lower() or "battery" in text.lower():
                    out["cmos_battery_voltage"] = val
            except (ValueError, AttributeError):
                pass

        for child in children:
            self._parse_lhm_motherboard(child, out)
