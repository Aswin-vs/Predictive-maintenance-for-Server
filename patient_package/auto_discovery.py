"""
Hardware Auto-Discovery Daemon — Phase G1.1

Profiles the patient server's hardware topology on first boot and generates
a hardware_topology.json manifest. This manifest is:
  1. Persisted locally (alongside server_id.json)
  2. Published on MQTT topic server/{server_id}/register
  3. Used by the Edge node to select appropriate AI models

Supports Windows (wmic, nvidia-smi, smartctl) and Linux (dmidecode,
lm-sensors, smartctl, lspci).

Author: @claude-architect (Phase G)
Date: 2026-09-04
"""

import os
import sys
import json
import platform
import socket
import subprocess
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add project root for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

try:
    import psutil
except ImportError:
    psutil = None

# Persistent topology file location (alongside server_id.json)
DEFAULT_TOPOLOGY_DIR = os.path.expanduser("~/.predictive_maintenance")
DEFAULT_TOPOLOGY_FILE = os.path.join(DEFAULT_TOPOLOGY_DIR, "hardware_topology.json")

SCHEMA_VERSION = "2.0"


def _run_cmd(cmd: List[str], timeout: int = 10) -> Optional[str]:
    """Run a shell command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _run_wmic(wmic_class: str, fields: List[str]) -> List[Dict[str, str]]:
    """Run a WMIC query (Windows only) and return a list of dicts."""
    if platform.system() != "Windows":
        return []
    cmd = ["wmic", wmic_class, "get", ",".join(fields), "/format:csv"]
    output = _run_cmd(cmd)
    if not output:
        return []
    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return []
    # First non-empty line is the header
    headers = [h.strip() for h in lines[0].split(",")]
    results = []
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(headers):
            results.append(dict(zip(headers, vals)))
    return results


class HardwareDiscovery:
    """Auto-detects and profiles the hardware topology of the patient server."""

    def __init__(self, topology_path: str = DEFAULT_TOPOLOGY_FILE):
        self.topology_path = topology_path
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux"

    def discover(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run full hardware discovery and return topology dict.
        Caches to disk. Use force_refresh=True to re-scan.
        """
        if not force_refresh and os.path.exists(self.topology_path):
            try:
                with open(self.topology_path, 'r') as f:
                    cached = json.load(f)
                if cached.get("schema_version") == SCHEMA_VERSION:
                    print(f"[DISCOVERY] Loaded cached topology: {self.topology_path}")
                    return cached
            except (json.JSONDecodeError, KeyError):
                pass

        print("[DISCOVERY] Running full hardware auto-discovery...")
        topology = {
            "schema_version": SCHEMA_VERSION,
            "hostname": socket.gethostname(),
            "os": f"{platform.system()} {platform.release()} {platform.version()}",
            "os_family": platform.system().lower(),
            "arch": platform.machine(),
            "python_version": platform.python_version(),
            "discovered_at": datetime.now(timezone.utc).isoformat(),
            "cpus": self._discover_cpus(),
            "gpus": self._discover_gpus(),
            "storage": self._discover_storage(),
            "memory": self._discover_memory(),
            "psu": self._discover_psu(),
            "fans": self._discover_fans(),
            "network": self._discover_network(),
            "motherboard": self._discover_motherboard(),
        }

        # Save to disk
        self._save(topology)
        return topology

    def _discover_cpus(self) -> List[Dict[str, Any]]:
        """Detect CPU(s) — model, cores, threads, frequency."""
        cpus = []
        cpu_info = {
            "id": "CPU0",
            "model": platform.processor() or "Unknown",
            "physical_cores": psutil.cpu_count(logical=False) if psutil else 0,
            "logical_cores": psutil.cpu_count(logical=True) if psutil else 0,
        }

        # Get max frequency
        if psutil:
            freq = psutil.cpu_freq()
            if freq:
                cpu_info["max_freq_mhz"] = round(freq.max) if freq.max else round(freq.current)

        # Windows: Get better CPU name via wmic
        if self.is_windows:
            wmic_data = _run_wmic("cpu", ["Name", "NumberOfCores", "NumberOfLogicalProcessors", "MaxClockSpeed"])
            for i, entry in enumerate(wmic_data):
                info = {
                    "id": f"CPU{i}",
                    "model": entry.get("Name", cpu_info["model"]),
                    "physical_cores": int(entry.get("NumberOfCores", cpu_info["physical_cores"])),
                    "logical_cores": int(entry.get("NumberOfLogicalProcessors", cpu_info["logical_cores"])),
                    "max_freq_mhz": int(entry.get("MaxClockSpeed", 0)),
                }
                cpus.append(info)

        # Linux: Parse /proc/cpuinfo
        elif self.is_linux:
            output = _run_cmd(["cat", "/proc/cpuinfo"])
            if output:
                model = ""
                for line in output.split("\n"):
                    if "model name" in line:
                        model = line.split(":")[1].strip()
                        break
                if model:
                    cpu_info["model"] = model

        if not cpus:
            cpus.append(cpu_info)

        return cpus

    def _discover_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPU(s) via nvidia-smi."""
        gpus = []

        # Try NVIDIA
        output = _run_cmd([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,pcie.link.width.current",
            "--format=csv,noheader,nounits"
        ])
        if output:
            for line in output.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu = {
                        "id": f"GPU{parts[0]}",
                        "vendor": "NVIDIA",
                        "model": parts[1],
                        "vram_mb": int(float(parts[2])) if parts[2] != "[N/A]" else 0,
                        "driver_version": parts[3],
                        "pcie_link_width": int(parts[4]) if len(parts) > 4 and parts[4] != "[N/A]" else 16,
                        "has_ecc": False,  # Will be updated below
                    }
                    gpus.append(gpu)

            # Check ECC support
            ecc_output = _run_cmd(["nvidia-smi", "--query-gpu=ecc.mode.current", "--format=csv,noheader"])
            if ecc_output:
                for i, line in enumerate(ecc_output.strip().split("\n")):
                    if i < len(gpus):
                        gpus[i]["has_ecc"] = line.strip().lower() == "enabled"

        # If no NVIDIA GPU, check for integrated/AMD
        if not gpus:
            if self.is_windows:
                wmic_data = _run_wmic("path", ["win32_VideoController", "get", "Name,AdapterRAM"])
                # Fallback: just report what psutil can see
                gpus.append({
                    "id": "GPU0",
                    "vendor": "Unknown",
                    "model": "Integrated/No dedicated GPU detected",
                    "vram_mb": 0,
                    "driver_version": "N/A",
                    "pcie_link_width": 0,
                    "has_ecc": False,
                })

        return gpus

    def _discover_storage(self) -> List[Dict[str, Any]]:
        """Detect storage devices — type (HDD/SSD/NVMe), SMART capability."""
        drives = []

        if psutil:
            partitions = psutil.disk_partitions(all=False)
            seen_devices = set()
            for part in partitions:
                device = part.device.rstrip("0123456789").rstrip("\\")  # Normalize
                if device in seen_devices:
                    continue
                seen_devices.add(device)

                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    drive_info = {
                        "id": os.path.basename(device) or device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "capacity_gb": round(usage.total / (1024**3), 1),
                        "type": "Unknown",
                        "smart_capable": False,
                    }
                except (PermissionError, OSError):
                    drive_info = {
                        "id": os.path.basename(device) or device,
                        "mountpoint": part.mountpoint,
                        "fstype": part.fstype,
                        "capacity_gb": 0,
                        "type": "Unknown",
                        "smart_capable": False,
                    }

                # Try smartctl to detect drive type
                smart_output = _run_cmd(["smartctl", "-i", device])
                if smart_output:
                    drive_info["smart_capable"] = True
                    if "Solid State" in smart_output or "SSD" in smart_output:
                        drive_info["type"] = "SSD"
                    elif "NVMe" in smart_output or "nvme" in smart_output.lower():
                        drive_info["type"] = "NVMe"
                    elif "Rotation Rate" in smart_output:
                        if "Solid State Device" in smart_output:
                            drive_info["type"] = "SSD"
                        else:
                            drive_info["type"] = "HDD"
                    # Extract model
                    for line in smart_output.split("\n"):
                        if "Device Model" in line or "Model Number" in line:
                            drive_info["model"] = line.split(":")[1].strip()
                            break
                else:
                    # Heuristic: if no SMART, assume SSD (modern default)
                    drive_info["type"] = "SSD"

                drives.append(drive_info)

        # If no drives found via psutil, add a placeholder
        if not drives:
            drives.append({
                "id": "disk0",
                "type": "Unknown",
                "capacity_gb": 0,
                "smart_capable": False,
            })

        return drives

    def _discover_memory(self) -> List[Dict[str, Any]]:
        """Detect memory sticks — total, type, ECC status."""
        sticks = []

        if self.is_windows:
            wmic_data = _run_wmic("memorychip", [
                "BankLabel", "Capacity", "Speed", "MemoryType",
                "FormFactor", "Manufacturer"
            ])
            for entry in wmic_data:
                capacity_bytes = int(entry.get("Capacity", 0) or 0)
                stick = {
                    "id": entry.get("BankLabel", "DIMM_Unknown"),
                    "size_gb": round(capacity_bytes / (1024**3), 1) if capacity_bytes else 0,
                    "speed_mhz": int(entry.get("Speed", 0) or 0),
                    "manufacturer": entry.get("Manufacturer", "Unknown"),
                    "ecc": False,  # Consumer Windows RAM is typically non-ECC
                }
                if stick["size_gb"] > 0:
                    sticks.append(stick)

        elif self.is_linux:
            output = _run_cmd(["sudo", "dmidecode", "--type", "memory"])
            # Parse dmidecode output for memory sticks
            if output:
                current_stick = {}
                for line in output.split("\n"):
                    line = line.strip()
                    if line.startswith("Size:") and "No Module" not in line:
                        size_str = line.split(":")[1].strip()
                        try:
                            if "GB" in size_str:
                                current_stick["size_gb"] = float(size_str.replace("GB", "").strip())
                            elif "MB" in size_str:
                                current_stick["size_gb"] = float(size_str.replace("MB", "").strip()) / 1024
                        except ValueError:
                            pass
                    elif line.startswith("Speed:"):
                        try:
                            current_stick["speed_mhz"] = int(re.search(r'\d+', line).group())
                        except (ValueError, AttributeError):
                            pass
                    elif line.startswith("Type:"):
                        current_stick["type"] = line.split(":")[1].strip()
                    elif line.startswith("Locator:"):
                        current_stick["id"] = line.split(":")[1].strip()
                    elif line == "" and current_stick.get("size_gb", 0) > 0:
                        sticks.append(current_stick)
                        current_stick = {}

        # Fallback: use psutil for total RAM
        if not sticks and psutil:
            mem = psutil.virtual_memory()
            sticks.append({
                "id": "SYSTEM_TOTAL",
                "size_gb": round(mem.total / (1024**3), 1),
                "speed_mhz": 0,
                "ecc": False,
            })

        return sticks

    def _discover_psu(self) -> Dict[str, Any]:
        """Detect PSU info — mostly unavailable on consumer hardware."""
        psu = {
            "type": "unknown",
            "wattage": 0,
            "rails": ["3.3V", "5V", "12V"],
            "monitoring_available": False,
        }

        # Check if IPMI is available (servers only)
        ipmi_output = _run_cmd(["ipmitool", "sensor", "list"])
        if ipmi_output:
            psu["monitoring_available"] = True
            psu["type"] = "server_ipmi"
            # Parse voltage sensors from IPMI output
            for line in ipmi_output.split("\n"):
                if "Watt" in line or "PS" in line:
                    psu["type"] = "server_redundant"
                    break

        # Laptops: detect battery
        if psutil:
            battery = psutil.sensors_battery()
            if battery is not None:
                psu["type"] = "laptop_battery"
                psu["has_battery"] = True
            else:
                psu["type"] = "desktop_atx"

        return psu

    def _discover_fans(self) -> List[Dict[str, Any]]:
        """Detect fan information."""
        fans = []

        if psutil:
            try:
                fan_data = psutil.sensors_fans()
                if fan_data:
                    for name, entries in fan_data.items():
                        for i, entry in enumerate(entries):
                            fans.append({
                                "id": f"{name}_{i}",
                                "label": entry.label or f"{name}_fan_{i}",
                                "current_rpm": entry.current,
                            })
            except (AttributeError, NotImplementedError):
                pass

        # If no fans detected, report as unknown
        if not fans:
            fans.append({
                "id": "fan_unknown",
                "label": "System fans (not individually detectable)",
                "current_rpm": 0,
            })

        return fans

    def _discover_network(self) -> List[Dict[str, Any]]:
        """Detect network interfaces."""
        interfaces = []

        if psutil:
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            for name, addr_list in addrs.items():
                # Skip loopback and virtual adapters
                if name.lower().startswith(("lo", "veth", "docker", "br-")):
                    continue

                iface = {
                    "id": name,
                    "type": "unknown",
                    "speed_mbps": 0,
                    "is_up": False,
                }

                if name in stats:
                    s = stats[name]
                    iface["speed_mbps"] = s.speed
                    iface["is_up"] = s.isup

                # Determine type heuristic
                name_lower = name.lower()
                if "eth" in name_lower or "en" in name_lower:
                    iface["type"] = "ethernet"
                elif "wlan" in name_lower or "wi-fi" in name_lower or "wireless" in name_lower:
                    iface["type"] = "wifi"

                # Get IPv4 address
                for addr in addr_list:
                    if addr.family.name == "AF_INET":
                        iface["ipv4"] = addr.address
                        break

                interfaces.append(iface)

        return interfaces

    def _discover_motherboard(self) -> Dict[str, Any]:
        """Detect motherboard/BIOS info."""
        mb = {"manufacturer": "Unknown", "product": "Unknown", "bios_version": "Unknown"}

        if self.is_windows:
            wmic_data = _run_wmic("baseboard", ["Manufacturer", "Product"])
            if wmic_data:
                mb["manufacturer"] = wmic_data[0].get("Manufacturer", "Unknown")
                mb["product"] = wmic_data[0].get("Product", "Unknown")

            bios_data = _run_wmic("bios", ["SMBIOSBIOSVersion", "ReleaseDate"])
            if bios_data:
                mb["bios_version"] = bios_data[0].get("SMBIOSBIOSVersion", "Unknown")

        elif self.is_linux:
            output = _run_cmd(["cat", "/sys/devices/virtual/dmi/id/board_vendor"])
            if output:
                mb["manufacturer"] = output
            output = _run_cmd(["cat", "/sys/devices/virtual/dmi/id/board_name"])
            if output:
                mb["product"] = output
            output = _run_cmd(["cat", "/sys/devices/virtual/dmi/id/bios_version"])
            if output:
                mb["bios_version"] = output

        return mb

    def _save(self, topology: Dict[str, Any]):
        """Persist topology to disk."""
        parent_dir = os.path.dirname(self.topology_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self.topology_path, 'w') as f:
            json.dump(topology, f, indent=2)
        print(f"[DISCOVERY] Saved topology to {self.topology_path}")

    def get_summary(self, topology: Dict[str, Any]) -> str:
        """Return a human-readable summary of the discovered hardware."""
        cpus = topology.get("cpus", [])
        gpus = topology.get("gpus", [])
        storage = topology.get("storage", [])
        memory = topology.get("memory", [])

        cpu_str = ", ".join(c.get("model", "?") for c in cpus) if cpus else "Unknown"
        gpu_str = ", ".join(g.get("model", "?") for g in gpus) if gpus else "None"
        total_ram = sum(m.get("size_gb", 0) for m in memory)
        storage_str = ", ".join(
            f"{s.get('type', '?')} {s.get('capacity_gb', 0)}GB"
            for s in storage
        ) if storage else "Unknown"

        return (
            f"CPU: {cpu_str} | "
            f"GPU: {gpu_str} | "
            f"RAM: {total_ram}GB | "
            f"Storage: {storage_str} | "
            f"OS: {topology.get('os', '?')}"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hardware Auto-Discovery Daemon")
    parser.add_argument("--force", action="store_true", help="Force re-scan even if cached topology exists")
    parser.add_argument("--output", type=str, default=DEFAULT_TOPOLOGY_FILE, help="Output file path")
    args = parser.parse_args()

    discovery = HardwareDiscovery(topology_path=args.output)
    topology = discovery.discover(force_refresh=args.force)

    summary = discovery.get_summary(topology)
    print(f"\n{'='*60}")
    print(f"HARDWARE TOPOLOGY SUMMARY")
    print(f"{'='*60}")
    print(summary)
    print(f"\nFull topology saved to: {args.output}")
    print(f"\nJSON output:")
    print(json.dumps(topology, indent=2))
