"""
Telemetry Plugins Package -- Phase 4 Production Hardening

All hardware telemetry plugins inherit from TelemetryPlugin base class.
Each plugin collects hardware metrics from a specific subsystem with
mandatory graceful degradation and precursor failure analysis.

Author: Antigravity AI Engine (Phase 4 Hardening)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class TelemetryPlugin(ABC):
    """Base class for all hardware telemetry plugins."""

    # Plugin metadata -- subclasses MUST override these
    PLUGIN_NAME: str = "base"
    SUBSYSTEM: str = "unknown"  # cpu, gpu, storage, memory, psu_fans, motherboard, network
    SUPPORTED_OS: List[str] = ["windows", "linux"]  # Which OS this plugin works on

    def __init__(self):
        self._available = False
        self._error_count = 0
        self._max_errors = 5  # After this many consecutive errors, mark unavailable

    @abstractmethod
    def detect(self) -> bool:
        """
        Detect if this plugin's hardware is present.
        Returns True if the hardware exists and can be read.
        Called once during initialization.
        """
        pass

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """
        Collect current telemetry from this hardware component.
        Returns a dict with the component's current metrics.
        Must handle errors gracefully -- never raise exceptions.
        """
        pass

    @property
    def is_available(self) -> bool:
        """Whether this plugin's hardware is available."""
        return self._available

    def safe_collect(self) -> Optional[Dict[str, Any]]:
        """
        Safely collect telemetry with error tracking.
        Returns None if the plugin has failed too many times.
        """
        if not self._available:
            return None
        try:
            data = self.collect()
            self._error_count = 0  # Reset on success
            return data
        except Exception as e:
            self._error_count += 1
            if self._error_count >= self._max_errors:
                self._available = False
                print(f"[PLUGIN] {self.PLUGIN_NAME} disabled after {self._max_errors} consecutive errors: {e}")
            return None

    def initialize(self) -> bool:
        """Initialize the plugin. Returns True if hardware detected."""
        try:
            self._available = self.detect()
            if self._available:
                print(f"[PLUGIN] {self.PLUGIN_NAME} initialized successfully")
            else:
                print(f"[PLUGIN] {self.PLUGIN_NAME} -- hardware not detected")
            return self._available
        except Exception as e:
            print(f"[PLUGIN] {self.PLUGIN_NAME} initialization failed: {e}")
            self._available = False
            return False


# Re-export plugins
try:
    from .cpu_plugin import CPUPlugin
    from .gpu_plugin import GPUPlugin
    from .gpu_nvidia_plugin import GPUNvidiaPlugin
    from .motherboard_plugin import MotherboardPlugin
    from .memory_plugin import MemoryPlugin
    from .storage_plugin import StoragePlugin
    from .psu_fan_plugin import PSUFanPlugin
except ImportError:
    pass
