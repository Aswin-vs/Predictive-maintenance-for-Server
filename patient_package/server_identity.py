"""
Server Identity Module — Assigns a unique UUID to each Patient Server.

Each patient server is uniquely identified by a UUID v4 that is:
1. Auto-generated on first run (if no identity file exists)
2. Persisted locally to ~/.predictive_maintenance/server_id.json
3. Included in every MQTT telemetry payload
4. Registered in the cloud backend's server registry

This module is imported by mqtt_publisher.py to tag telemetry data.

Author: @claude-architect (Phase D — Production-Ready Upgrade)
Date: 2026-08-09
"""

import os
import sys
import json
import uuid
import platform
import socket
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Default path for the identity file
DEFAULT_IDENTITY_DIR = os.path.expanduser("~/.predictive_maintenance")
DEFAULT_IDENTITY_FILE = os.path.join(DEFAULT_IDENTITY_DIR, "server_id.json")


class ServerIdentity:
    """Manages a persistent, unique identity for a patient server."""
    
    def __init__(self, identity_path: str = DEFAULT_IDENTITY_FILE, data_mode: str = "production"):
        """
        Load or create a server identity.
        
        Args:
            identity_path: Path to the JSON file storing the server identity.
            data_mode: "production" or "demo". Controls which InfluxDB bucket
                       telemetry is routed to on the cloud backend.
        """
        self.identity_path = identity_path
        self.data_mode = data_mode
        self._identity = self._load_or_create()
    
    def _load_or_create(self) -> Dict[str, Any]:
        """Load identity from disk or create a new one."""
        if os.path.exists(self.identity_path):
            try:
                with open(self.identity_path, 'r') as f:
                    identity = json.load(f)
                print(f"[IDENTITY] Loaded server identity: {identity['server_id']}")
                print(f"           Name: {identity.get('server_name', 'Unnamed')}")
                return identity
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[IDENTITY] Warning: Corrupted identity file ({e}). Regenerating.")
        
        # Generate new identity
        identity = self._generate_identity()
        self._save(identity)
        print(f"[IDENTITY] NEW server identity created: {identity['server_id']}")
        return identity
    
    def _generate_identity(self) -> Dict[str, Any]:
        """Generate a new server identity with hardware profiling."""
        server_id = str(uuid.uuid4())
        hostname = socket.gethostname()
        
        hardware_profile = self._detect_hardware()
        
        return {
            "server_id": server_id,
            "server_name": f"{hostname}-patient",
            "hostname": hostname,
            "hardware_profile": hardware_profile,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "data_mode": self.data_mode,
            "schema_version": "1.0"
        }
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Auto-detect the server's hardware configuration."""
        profile = {
            "cpu": platform.processor() or "Unknown",
            "os": f"{platform.system()} {platform.release()}",
            "arch": platform.machine(),
            "python": platform.python_version(),
        }
        
        # Try to get RAM info
        try:
            import psutil
            ram = psutil.virtual_memory()
            profile["ram_gb"] = round(ram.total / (1024**3), 1)
        except ImportError:
            profile["ram_gb"] = "unknown (psutil not installed)"
        
        # Try to detect GPU
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                profile["gpu"] = result.stdout.strip()
            else:
                profile["gpu"] = "None detected"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            profile["gpu"] = "None detected"
        
        return profile
    
    def _save(self, identity: Dict[str, Any]):
        """Persist identity to disk."""
        os.makedirs(os.path.dirname(self.identity_path), exist_ok=True)
        with open(self.identity_path, 'w') as f:
            json.dump(identity, f, indent=4)
        print(f"[IDENTITY] Saved to {self.identity_path}")
    
    # --- Public Properties ---
    
    @property
    def server_id(self) -> str:
        """The unique UUID for this server."""
        return self._identity["server_id"]
    
    @property
    def server_name(self) -> str:
        """Human-readable name for this server."""
        return self._identity.get("server_name", "Unnamed")
    
    @property
    def hardware_profile(self) -> Dict[str, Any]:
        """Hardware configuration dict."""
        return self._identity.get("hardware_profile", {})
    
    @property
    def registration_payload(self) -> Dict[str, Any]:
        """
        Full identity payload suitable for sending to cloud for registration.
        """
        return {
            "server_id": self.server_id,
            "server_name": self.server_name,
            "hardware_profile": self.hardware_profile,
            "registered_at": self._identity.get("registered_at"),
            "data_mode": self.data_mode,
        }
    
    def update_name(self, new_name: str):
        """Update the server's human-readable name."""
        self._identity["server_name"] = new_name
        self._save(self._identity)
    
    def set_data_mode(self, mode: str):
        """Set data mode to 'production' or 'demo'."""
        if mode not in ("production", "demo"):
            raise ValueError(f"data_mode must be 'production' or 'demo', got '{mode}'")
        self.data_mode = mode
        self._identity["data_mode"] = mode
        self._save(self._identity)


# Convenience: module-level access for simple use cases
_default_identity = None

def get_server_identity(data_mode: str = "production") -> ServerIdentity:
    """Get or create the default server identity (singleton)."""
    global _default_identity
    if _default_identity is None:
        _default_identity = ServerIdentity(data_mode=data_mode)
    return _default_identity


if __name__ == "__main__":
    # Quick test: generate or load identity
    identity = ServerIdentity(data_mode="demo")
    print(f"\n--- Server Identity ---")
    print(f"  ID:       {identity.server_id}")
    print(f"  Name:     {identity.server_name}")
    print(f"  Hardware: {json.dumps(identity.hardware_profile, indent=4)}")
    print(f"  Mode:     {identity.data_mode}")
    print(f"\nRegistration payload:")
    print(json.dumps(identity.registration_payload, indent=2))
