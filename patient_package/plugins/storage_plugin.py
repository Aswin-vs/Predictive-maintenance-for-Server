"""
Telemetry Plugin: Storage (Tier 1 NVMe & SATA) — Phase 4 Production Hardening

Collects enterprise and consumer storage telemetry across NVMe and SATA drives:
- NVMe SSDs: Queries `nvme smart-log` (Linux) or `smartctl -a` (Windows/Linux)
  to extract NVMe Log Page 0x02 attributes: Available Spare %, Percentage Used,
  Media and Data Integrity Errors, Critical Warnings, and Thermal Throttle duration.
- SATA Drives: Queries `smartctl -a` for SMART 5 (Reallocated Sectors),
  187 (Reported Uncorrectable), 188 (Command Timeout), 197 (Current Pending),
  and 198 (Offline Uncorrectable).
- Degradation Precursors: Flags NAND flash block exhaustion (Available Spare < 10%),
  write endurance limit reached (Percentage Used > 100%), and pending/uncorrectable sectors.
- Graceful Degradation: If smartctl or nvme-cli is absent, uses psutil and WMI
  to report drive metrics, logging once without throwing unhandled exceptions.

Author: Antigravity AI Engine (Phase 4 Hardening)
"""

import os
import re
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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    wmi = None
    WMI_AVAILABLE = False

logger = logging.getLogger("telemetry.plugins.storage")


def _run_cmd_quiet(cmd: List[str], timeout: float = 3.0) -> Optional[str]:
    """Execute a subprocess quietly with CREATE_NO_WINDOW on Windows."""
    try:
        flags = {}
        if platform.system() == "Windows":
            flags["creationflags"] = subprocess.CREATE_NO_WINDOW
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **flags)
        # Note: smartctl returns non-zero exit codes for valid drive states
        return res.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


class StoragePlugin(TelemetryPlugin):
    PLUGIN_NAME = "storage_tier1_smart"
    SUBSYSTEM = "storage"
    SUPPORTED_OS = ["windows", "linux"]

    def __init__(self):
        super().__init__()
        self._os_name = platform.system().lower()
        self._logged_warnings = set()
        self.smartctl_available = False
        self.nvme_cli_available = False

    def _log_once(self, key: str, message: str, level: str = "warning"):
        if key not in self._logged_warnings:
            self._logged_warnings.add(key)
            if level == "warning":
                logger.warning(message)
            else:
                logger.info(message)

    def detect(self) -> bool:
        """Detect storage collection tools and drives."""
        # Probe smartctl
        smart_out = _run_cmd_quiet(["smartctl", "--version"])
        if smart_out and "smartctl" in smart_out:
            self.smartctl_available = True

        # Probe nvme-cli (Linux)
        if self._os_name == "linux":
            nvme_out = _run_cmd_quiet(["nvme", "version"])
            if nvme_out and "nvme" in nvme_out:
                self.nvme_cli_available = True

        if not self.smartctl_available and not self.nvme_cli_available:
            self._log_once("smartctl_missing", "smartctl/nvme-cli not found; using psutil and WMI fallback.", "info")

        return PSUTIL_AVAILABLE or self.smartctl_available

    def collect(self) -> Dict[str, Any]:
        """Collect storage telemetry with NVMe Log Page 0x02 and SATA SMART."""
        drives = []
        if not PSUTIL_AVAILABLE:
            return {"drives": [], "drive_count": 0}

        io_counters = psutil.disk_io_counters(perdisk=True) or {}
        partitions = psutil.disk_partitions(all=False)

        # Map physical drives on Windows via WMI if available
        wmi_disks = self._get_windows_wmi_disks() if self._os_name == "windows" else {}

        seen_devices = set()
        for part in partitions:
            device = part.device
            device_norm = device.rstrip("0123456789").rstrip("\\")
            if device_norm in seen_devices:
                continue
            seen_devices.add(device_norm)

            drive: Dict[str, Any] = {
                "id": device,
                "mountpoint": part.mountpoint,
                "fstype": part.fstype,
                "model": "Unknown Drive",
                "type": "Unknown",
                "capacity_gb": 0.0,
                "used_gb": 0.0,
                "used_pct": 0.0,
                "read_bytes": 0,
                "write_bytes": 0,
                "read_count": 0,
                "write_count": 0,
                "read_time_ms": None,
                "write_time_ms": None,
                "smart_supported": False,
                "smart_source": "none",
                # NVMe Log Page 0x02 Attributes
                "available_spare_pct": None,
                "available_spare_threshold_pct": None,
                "percentage_used": None,
                "media_errors": None,
                "critical_warnings": None,
                "power_loss_count": None,
                "thermal_throttle_duration_mins": None,
                "thermal_throttle_count": None,
                "temperature_c": None,
                "degradation_flag_spare_low": False,
                "degradation_flag_endurance_exhausted": False,
                # SATA SMART Attributes
                "smart": {
                    "reallocated_sectors": None,
                    "power_on_hours": None,
                    "reported_uncorrectable": None,
                    "command_timeout": None,
                    "temperature_c": None,
                    "current_pending_sector": None,
                    "offline_uncorrectable": None,
                },
                "degradation_flag_uncorrectable": False,
            }

            # 1. Capacity & Usage via psutil
            try:
                usage = psutil.disk_usage(part.mountpoint)
                drive["capacity_gb"] = round(usage.total / (1024**3), 1)
                drive["used_gb"] = round(usage.used / (1024**3), 1)
                drive["used_pct"] = usage.percent
            except (PermissionError, OSError):
                pass

            # 2. I/O counters
            for disk_name, counters in io_counters.items():
                if disk_name.lower() in device.lower() or device_norm.lower().endswith(disk_name.lower()):
                    drive["read_bytes"] = counters.read_bytes
                    drive["write_bytes"] = counters.write_bytes
                    drive["read_count"] = counters.read_count
                    drive["write_count"] = counters.write_count
                    if hasattr(counters, "read_time"):
                        drive["read_time_ms"] = counters.read_time
                        drive["write_time_ms"] = counters.write_time
                    break

            # 3. Model identification from WMI / PowerShell (Windows)
            clean_dev = device.rstrip("\\").upper()
            if not clean_dev.endswith(":"):
                clean_dev += ":"
            w_info = wmi_disks.get(clean_dev) or wmi_disks.get(device) or wmi_disks.get(device.rstrip("\\"))
            if w_info:
                drive["model"] = w_info.get("model", drive["model"])
                drive["type"] = w_info.get("type", drive["type"])

            if drive["type"] == "Unknown" or not drive.get("type"):
                cap = drive.get("capacity_gb", 0)
                if cap >= 800:
                    drive["type"] = "Mechanical HDD"
                    if drive["model"] == "Unknown Drive":
                        drive["model"] = "1TB Mechanical HDD"
                elif cap > 0 and cap <= 600:
                    drive["type"] = "SATA SSD"
                    if drive["model"] == "Unknown Drive":
                        drive["model"] = "Solid State Drive (SSD)"

            # 4. SMART Collection (smartctl or nvme-cli)
            if self.smartctl_available:
                self._collect_smartctl(device, drive)
            elif self.nvme_cli_available and drive.get("type") == "NVMe":
                self._collect_nvme_cli(device, drive)

            # Ensure backward-compatible defaults in smart dict if null
            for k in ["reallocated_sectors", "power_on_hours", "reported_uncorrectable", "command_timeout", "temperature_c", "current_pending_sector", "offline_uncorrectable"]:
                if drive["smart"].get(k) is None:
                    drive["smart"][k] = 0

            drives.append(drive)

        return {"drives": drives, "drive_count": len(drives)}

    def _collect_smartctl(self, device: str, drive: Dict[str, Any]):
        """Query smartctl for NVMe Log Page 0x02 or SATA SMART."""
        raw_out = _run_cmd_quiet(["smartctl", "-x", "-j", device])
        if raw_out:
            try:
                data = json.loads(raw_out)
                drive["smart_supported"] = True
                drive["smart_source"] = "smartctl_json"

                # Detect model & drive family
                drive["model"] = data.get("model_name", drive["model"])
                dev_type = data.get("device", {}).get("type", "")
                if "nvme" in dev_type.lower():
                    drive["type"] = "NVMe"
                elif "ata" in dev_type.lower() or "scsi" in dev_type.lower():
                    drive["type"] = "SSD" if "solid" in data.get("rotation_rate", "").lower() else "HDD"

                # NVMe Log Page 0x02 extraction
                nvme_log = data.get("nvme_smart_health_information_log", {})
                if nvme_log:
                    drive["type"] = "NVMe"
                    spare = nvme_log.get("available_spare")
                    thresh = nvme_log.get("available_spare_threshold")
                    pct_used = nvme_log.get("percentage_used")
                    media_err = nvme_log.get("media_errors")
                    crit_warn = nvme_log.get("critical_warning")
                    temp_k = nvme_log.get("temperature")

                    drive["available_spare_pct"] = spare
                    drive["available_spare_threshold_pct"] = thresh
                    drive["percentage_used"] = pct_used
                    drive["media_errors"] = media_err
                    drive["critical_warnings"] = crit_warn
                    drive["temperature_c"] = (temp_k - 273) if temp_k else None

                    # Thermal throttling count/time
                    t1_trans = nvme_log.get("thermal_management_temp_1_transition_count", 0)
                    t2_trans = nvme_log.get("thermal_management_temp_2_transition_count", 0)
                    t1_time = nvme_log.get("total_time_for_thermal_management_temp_1", 0)
                    drive["thermal_throttle_count"] = t1_trans + t2_trans
                    drive["thermal_throttle_duration_mins"] = round(t1_time / 60.0, 1)

                    # Degradation Precursor Flags
                    if spare is not None and spare < 10:
                        drive["degradation_flag_spare_low"] = True
                    if pct_used is not None and pct_used > 100:
                        drive["degradation_flag_endurance_exhausted"] = True

                # SATA ATA SMART table extraction
                ata_attrs = data.get("ata_smart_attributes", {}).get("table", [])
                for attr in ata_attrs:
                    aid = attr.get("id")
                    val = attr.get("raw", {}).get("value", 0)
                    if aid == 5:
                        drive["smart"]["reallocated_sectors"] = val
                    elif aid == 9:
                        drive["smart"]["power_on_hours"] = val
                    elif aid == 187:
                        drive["smart"]["reported_uncorrectable"] = val
                    elif aid == 188:
                        drive["smart"]["command_timeout"] = val
                    elif aid == 194:
                        drive["smart"]["temperature_c"] = val
                    elif aid == 197:
                        drive["smart"]["current_pending_sector"] = val
                    elif aid == 198:
                        drive["smart"]["offline_uncorrectable"] = val

                # Flag uncorrectable / sector issues
                if (drive["smart"]["reallocated_sectors"] or 0) > 0 or \
                   (drive["smart"]["reported_uncorrectable"] or 0) > 0 or \
                   (drive["smart"]["current_pending_sector"] or 0) > 0 or \
                   (drive["smart"]["offline_uncorrectable"] or 0) > 0:
                    drive["degradation_flag_uncorrectable"] = True

                return
            except Exception:
                pass

        # Text fallback if JSON parsing fails
        text_out = _run_cmd_quiet(["smartctl", "-a", device])
        if text_out:
            drive["smart_supported"] = True
            drive["smart_source"] = "smartctl_text"
            self._parse_smartctl_text(text_out, drive)

    def _parse_smartctl_text(self, text: str, drive: Dict[str, Any]):
        """Parse raw text output of smartctl."""
        for line in text.split("\n"):
            line_strip = line.strip()
            if "Available Spare:" in line_strip:
                match = re.search(r"(\d+)%", line_strip)
                if match:
                    drive["available_spare_pct"] = int(match.group(1))
                    if drive["available_spare_pct"] < 10:
                        drive["degradation_flag_spare_low"] = True
            elif "Percentage Used:" in line_strip:
                match = re.search(r"(\d+)%", line_strip)
                if match:
                    drive["percentage_used"] = int(match.group(1))
                    if drive["percentage_used"] > 100:
                        drive["degradation_flag_endurance_exhausted"] = True
            elif "Media and Data Integrity Errors:" in line_strip:
                parts = line_strip.split(":")
                if len(parts) > 1:
                    drive["media_errors"] = int(parts[1].strip().replace(",", ""))
            elif "Critical Warning:" in line_strip:
                match = re.search(r"(0x[0-9a-fA-F]+)", line_strip)
                if match:
                    drive["critical_warnings"] = int(match.group(1), 16)

            # SATA Attribute Lines (ID# ATTRIBUTE_NAME ... RAW_VALUE)
            parts = line_strip.split()
            if len(parts) >= 10 and parts[0].isdigit():
                try:
                    aid = int(parts[0])
                    val = int(parts[9].split("/")[0])
                    key_map = {
                        5: "reallocated_sectors",
                        9: "power_on_hours",
                        187: "reported_uncorrectable",
                        188: "command_timeout",
                        194: "temperature_c",
                        197: "current_pending_sector",
                        198: "offline_uncorrectable",
                    }
                    if aid in key_map:
                        drive["smart"][key_map[aid]] = val
                except (ValueError, IndexError):
                    pass

        # Check SATA degradation flags
        if (drive["smart"].get("reallocated_sectors") or 0) > 0 or \
           (drive["smart"].get("reported_uncorrectable") or 0) > 0 or \
           (drive["smart"].get("current_pending_sector") or 0) > 0 or \
           (drive["smart"].get("offline_uncorrectable") or 0) > 0:
            drive["degradation_flag_uncorrectable"] = True

    def _collect_nvme_cli(self, device: str, drive: Dict[str, Any]):
        """Linux nvme-cli fallback."""
        out = _run_cmd_quiet(["nvme", "smart-log", device, "-o", "json"])
        if out:
            try:
                log = json.loads(out)
                drive["smart_supported"] = True
                drive["smart_source"] = "nvme_cli"
                drive["available_spare_pct"] = log.get("avail_spare")
                drive["available_spare_threshold_pct"] = log.get("spare_thresh")
                drive["percentage_used"] = log.get("percent_used")
                drive["media_errors"] = log.get("media_errors")
                drive["critical_warnings"] = log.get("critical_warning")
                drive["temperature_c"] = log.get("temperature") - 273 if log.get("temperature") else None
            except Exception:
                pass

    def _get_windows_wmi_disks(self) -> Dict[str, Dict[str, Any]]:
        """Map Windows drive letters to physical disk properties via PowerShell, WMI, and partition mapping."""
        disks = {}
        # 1. Query PhysicalDisks via PowerShell
        pdisks_by_id = {}
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Select-Object DeviceId, FriendlyName, MediaType, BusType, Size | ConvertTo-Json -Compress"]
            out = _run_cmd_quiet(cmd, timeout=4.0)
            if out:
                pdata = json.loads(out)
                if isinstance(pdata, dict):
                    pdata = [pdata]
                for p in pdata:
                    did = str(p.get("DeviceId"))
                    pdisks_by_id[did] = p
        except Exception:
            pass

        # 2. Query Win32_DiskDrive via WMI / PowerShell
        ddrives_by_index = {}
        try:
            dcmd = ["powershell", "-NoProfile", "-Command", "Get-WmiObject Win32_DiskDrive | Select-Object Index, Model, InterfaceType, MediaType, Size | ConvertTo-Json -Compress"]
            dout = _run_cmd_quiet(dcmd, timeout=4.0)
            if dout:
                ddata = json.loads(dout)
                if isinstance(ddata, dict):
                    ddata = [ddata]
                for d in ddata:
                    idx = str(d.get("Index"))
                    ddrives_by_index[idx] = d
        except Exception:
            pass

        # 3. Query Win32_LogicalDiskToPartition for exact Drive Letter -> Disk Index mapping
        partition_map = {}
        try:
            lpcmd = ["powershell", "-NoProfile", "-Command", "Get-WmiObject Win32_LogicalDiskToPartition | Select-Object Antecedent, Dependent | ConvertTo-Json -Compress"]
            lpout = _run_cmd_quiet(lpcmd, timeout=4.0)
            if lpout:
                lpdata = json.loads(lpout)
                if isinstance(lpdata, dict):
                    lpdata = [lpdata]
                for item in lpdata:
                    ant = str(item.get("Antecedent", ""))
                    dep = str(item.get("Dependent", ""))
                    disk_match = re.search(r'Disk #(\d+)', ant)
                    let_match = re.search(r'DeviceID=\"([A-Za-z]:)\"', dep)
                    if disk_match and let_match:
                        partition_map[let_match.group(1).upper()] = disk_match.group(1)
        except Exception:
            pass

        # 3b. Backup: Get-Partition
        if not partition_map:
            try:
                part_cmd = ["powershell", "-NoProfile", "-Command", "Get-Partition | Select-Object DiskNumber, DriveLetter | ConvertTo-Json -Compress"]
                part_out = _run_cmd_quiet(part_cmd, timeout=4.0)
                if part_out:
                    parts = json.loads(part_out)
                    if isinstance(parts, dict):
                        parts = [parts]
                    for p in parts:
                        dl = str(p.get("DriveLetter") or "").strip().upper()
                        dn = p.get("DiskNumber")
                        if dl and dn is not None:
                            partition_map[f"{dl}:"] = str(dn)
            except Exception:
                pass

        # 4. Map each drive letter to physical hardware details
        for letter, disk_idx in partition_map.items():
            pdisk = pdisks_by_id.get(disk_idx, {})
            ddrive = ddrives_by_index.get(disk_idx, {})

            name = str(pdisk.get("FriendlyName") or ddrive.get("Model") or f"Disk {disk_idx}").strip()
            btype = str(pdisk.get("BusType") or ddrive.get("InterfaceType") or "").upper()
            mtype = str(pdisk.get("MediaType") or ddrive.get("MediaType") or "").upper()
            size_b = pdisk.get("Size") or ddrive.get("Size") or 0

            if "NVME" in btype or "NVME" in name.upper() or "NVME" in mtype:
                dtype = "NVMe SSD"
            elif "SSD" in mtype or "SSD" in name.upper() or "SATA" in btype:
                dtype = "SATA SSD"
            elif "USB" in btype or "EXTERNAL" in mtype or "HDD" in mtype or size_b >= 800_000_000_000:
                dtype = "Mechanical HDD"
            elif size_b > 0 and size_b <= 600_000_000_000:
                dtype = "SATA SSD"
            else:
                dtype = "Mechanical HDD" if size_b >= 800_000_000_000 else "SSD"

            disks[letter] = {
                "model": name,
                "type": dtype,
                "bus": btype,
                "media": mtype,
                "size_bytes": size_b
            }

        return disks

