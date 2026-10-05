"""
Telemetry Plugin: NVIDIA GPU — Backward Compatibility Wrapper for Phase 4

Imports and aliases GPUPlugin from gpu_plugin.py to ensure zero breakage
for legacy imports while providing full Phase 4 enterprise degradation capabilities.

Author: Antigravity AI Engine (Phase 4 Hardening)
"""

try:
    from server.telemetry_agent.plugins.gpu_plugin import GPUPlugin
except ImportError:
    from plugins.gpu_plugin import GPUPlugin

# Alias for backward compatibility
class GPUNvidiaPlugin(GPUPlugin):
    PLUGIN_NAME = "gpu_nvidia"
    SUBSYSTEM = "gpu"
