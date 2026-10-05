"""
Remote Patient Telemetry Launcher (IdeaPad S145 via SSH from LOQ)
Connects to the IdeaPad S145 patient server over SSH and launches
the telemetry publisher remotely, streaming output live back to the LOQ console.
"""

import sys
import os
import time
import argparse
import paramiko

# Load network config from configs/network.env
def _load_network_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'network.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()

_load_network_env()

# Default connection settings for IdeaPad S145
IDEAPAD_HOST = os.environ.get("IDEAPAD_HOST", os.environ.get("IDEAPAD_IP", "ideapads145.tail1ab139.ts.net"))
IDEAPAD_USER = os.environ.get("IDEAPAD_USER", "Aswin")
IDEAPAD_PASS = os.environ.get("IDEAPAD_PASS")
REMOTE_DIR = r"C:\Users\Aswin\Documents\Predictive_Maintenance_Patient"
DEFAULT_RPI4_BROKER = os.environ.get("EDGE_BROKER_HOST", os.environ.get("RPI4_HOST", os.environ.get("RPI4_IP", "rpi4-edge.tail1ab139.ts.net")))


def run_remote_patient(broker: str, port: int = 1883, use_dummy: bool = False, mode: str = "production", server_name: str = "IdeaPad-Patient", use_v2: bool = True):
    print("=" * 65)
    print("  PATIENT TELEMETRY REMOTE RUNNER (IdeaPad S145 via SSH)")
    print("=" * 65)
    print(f" Connecting to Patient Machine: {IDEAPAD_USER}@{IDEAPAD_HOST}...")
    print(f" Target MQTT Broker:           {broker}:{port}")
    print(f" Publisher Version:            {'v2.0 (Auto-Discovery + Plugins)' if use_v2 else 'v1.0 (Legacy)'}")
    print(f" Mode:                         {'SIMULATED DUMMY' if use_dummy else 'LIVE HARDWARE (psutil + LibreHardware)'}")
    print(f" Remote Directory:             {REMOTE_DIR}")
    print("=" * 65)
    print()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(IDEAPAD_HOST, username=IDEAPAD_USER, password=IDEAPAD_PASS, timeout=8.0, look_for_keys=False, allow_agent=False)
        print(f"[SUCCESS] SSH connection established to {IDEAPAD_HOST}!\n")
    except Exception as e:
        print(f"[ERROR] Failed to connect to IdeaPad at {IDEAPAD_HOST}: {e}")
        print(" -> Check if IdeaPad is powered on, connected to WiFi Hotspot, and OpenSSH Server is running.")
        sys.exit(1)

    # Build remote execution command for PowerShell
    extra_flags = "--dummy" if use_dummy else ""
    if use_v2:
        script_path = f"{REMOTE_DIR}\\mqtt_publisher_v2.py"
        cmd = f"Set-Location -LiteralPath '{REMOTE_DIR}'; python '{script_path}' --broker {broker} --port {port} --mode {mode} --server-name '{server_name}' --interval 5 {extra_flags}"
    else:
        script_path = f"{REMOTE_DIR}\\mqtt_publisher.py"
        cmd = f"Set-Location -LiteralPath '{REMOTE_DIR}'; python '{script_path}' --broker {broker} --port {port} --mode {mode} --server-name '{server_name}' {extra_flags}"

    print(f"[REMOTE CMD] Starting telemetry publisher on IdeaPad...")
    print("Press Ctrl+C in this window at any time to stop remote publisher.\n" + "-" * 65)

    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
        # Stream remote output line-by-line
        for line in iter(stdout.readline, ""):
            print(f"[IdeaPad] {line}", end="")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[STOPPING] User interrupted. Stopping remote telemetry on IdeaPad...")
    finally:
        try:
            # Kill running python publisher on remote IdeaPad
            ssh.exec_command("powershell -Command \"Stop-Process -Name python -Force -ErrorAction SilentlyContinue\"")
        except Exception:
            pass
        ssh.close()
        print("[DISCONNECTED] SSH session closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Patient Telemetry on IdeaPad S145 remotely via SSH from LOQ")
    parser.add_argument("--broker", type=str, default=DEFAULT_RPI4_BROKER, help="MQTT Broker Host (default: from EDGE_BROKER_HOST / network.env)")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker Port (default: 1883)")
    parser.add_argument("--dummy", action="store_true", help="Use simulated dummy telemetry instead of live hardware")
    parser.add_argument("--mode", type=str, default="production", choices=["production", "demo"], help="Data mode")
    parser.add_argument("--server-name", type=str, default="IdeaPad-Patient", help="Human-readable server name")
    parser.add_argument("--v2", action="store_true", default=True, help="Use v2 publisher with auto-discovery and plugins (default: True)")
    parser.add_argument("--v1", action="store_true", help="Force use of legacy v1 publisher")
    args = parser.parse_args()

    use_v2 = not args.v1  # Default is v2 unless --v1 is explicitly set
    run_remote_patient(broker=args.broker, port=args.port, use_dummy=args.dummy, mode=args.mode, server_name=args.server_name, use_v2=use_v2)

