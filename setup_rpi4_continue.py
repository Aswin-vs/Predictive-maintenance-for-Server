"""
RPi4 Final Setup — Fixes from previous run + completes remaining steps.
Handles Unicode encoding safely and uses correct Debian 13 Trixie packages.
"""

import os
import sys
import time
import paramiko

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(BASE_DIR, "configs", "network.env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"\'')
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

RPI_HOST = os.environ.get("RPI4_HOST", os.environ.get("EDGE_BROKER_HOST", os.environ.get("RPI4_IP", "rpi4-edge.tail1ab139.ts.net")))
RPI_USER = os.environ.get("RPI4_USER", "pi")
RPI_PASS = os.environ.get("RPI4_PASS")
RPI_DEST = "/home/pi/predictive-maintenance"


def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(RPI_HOST, username=RPI_USER, password=RPI_PASS, timeout=10)
    print(f"[SSH] Connected to {RPI_HOST}")
    return ssh


def run_cmd(ssh, cmd, label="", timeout=300):
    if label:
        print(f"\n[{label}]")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    try:
        out = stdout.read().decode('utf-8', errors='replace').strip()
        err = stderr.read().decode('utf-8', errors='replace').strip()
    except Exception:
        out, err = "", ""
    if out:
        # Only print last 15 lines to avoid noise
        lines = out.split("\n")
        for line in lines[-15:]:
            safe_line = line.encode('ascii', errors='replace').decode('ascii')
            print(f"    {safe_line}")
    if err and exit_code != 0:
        for line in err.split("\n")[-5:]:
            safe_line = line.encode('ascii', errors='replace').decode('ascii')
            print(f"    [!] {safe_line}")
    return exit_code, out, err


def sudo_cmd(ssh, cmd, label="", timeout=300):
    full_cmd = f"echo '{RPI_PASS}' | sudo -S bash -c '{cmd}'"
    return run_cmd(ssh, full_cmd, label, timeout)


def upload_file(sftp, local_path, remote_path):
    if not os.path.exists(local_path):
        print(f"  [SKIP] Missing: {os.path.basename(local_path)}")
        return False
    remote_dir = os.path.dirname(remote_path)
    _ensure_remote_dir(sftp, remote_dir)
    print(f"  -> {os.path.basename(local_path)} -> {remote_path}")
    sftp.put(local_path, remote_path)
    return True


def _ensure_remote_dir(sftp, path):
    dirs_to_create = []
    while path and path != "/":
        try:
            sftp.stat(path)
            break
        except FileNotFoundError:
            dirs_to_create.insert(0, path)
            path = os.path.dirname(path)
    for d in dirs_to_create:
        try:
            sftp.mkdir(d)
        except Exception:
            pass


def main():
    print("=" * 60)
    print("  RPi4 FINAL SETUP (Post system update + reboot)")
    print("=" * 60)

    ssh = ssh_connect()

    # ─── Step 0: Verify system state ──────────────────────────────
    run_cmd(ssh, "uname -a", "SYSTEM INFO")
    run_cmd(ssh, "python3 --version", "PYTHON")
    run_cmd(ssh, "df -h / | tail -1", "DISK")
    run_cmd(ssh, "free -h | head -2", "RAM")

    # ─── Step 1: Install Mosquitto (apt previously failed) ────────
    print("\n" + "=" * 60)
    print("  STEP 1: INSTALL MOSQUITTO + SYSTEM DEPS")
    print("=" * 60)

    # Use correct packages for Debian 13 Trixie (libatlas-base-dev doesn't exist)
    sudo_cmd(ssh,
        "DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "mosquitto mosquitto-clients "
        "i2c-tools "
        "libopenblas-dev "
        "sqlite3",
        "APT INSTALL", timeout=300)

    # Configure Mosquitto for external connections
    sudo_cmd(ssh,
        "mkdir -p /etc/mosquitto/conf.d && "
        "echo -e \"listener 1883\\nallow_anonymous true\" > /etc/mosquitto/conf.d/external.conf",
        "MOSQUITTO CONFIG")

    sudo_cmd(ssh,
        "systemctl enable mosquitto && systemctl restart mosquitto",
        "MOSQUITTO ENABLE+START")

    # Verify
    run_cmd(ssh, "systemctl is-active mosquitto && mosquitto_pub -h 127.0.0.1 -t test -m hello && echo 'MQTT OK'",
            "MOSQUITTO VERIFY")

    # ─── Step 2: Install remaining Python packages ────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: INSTALL REMAINING PYTHON PACKAGES")
    print("=" * 60)

    # scikit-learn (previously crashed)
    run_cmd(ssh,
        f"{RPI_DEST}/venv/bin/pip install scikit-learn joblib 2>&1 | tail -3",
        "PIP: SKLEARN", timeout=600)

    # tflite-runtime — try multiple approaches for Python 3.13
    exit_code, out, _ = run_cmd(ssh,
        f"{RPI_DEST}/venv/bin/pip install tflite-runtime 2>&1 | tail -5",
        "PIP: TFLITE-RUNTIME", timeout=180)

    if exit_code != 0 or "ERROR" in out:
        # Try ai-edge-litert (Google's newer package)
        run_cmd(ssh,
            f"{RPI_DEST}/venv/bin/pip install ai-edge-litert 2>&1 | tail -5",
            "PIP: AI-EDGE-LITERT", timeout=180)

    # ─── Step 3: Re-deploy updated files ──────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: RE-DEPLOY UPDATED FILES")
    print("=" * 60)

    sftp = ssh.open_sftp()

    # Only re-upload files that were modified since first deployment
    updated_files = [
        ("edge/main.py", f"{RPI_DEST}/edge/main.py"),
        ("edge/mqtt_handler.py", f"{RPI_DEST}/edge/mqtt_handler.py"),
        ("server/training/feature_schema_v2.py", f"{RPI_DEST}/server/training/feature_schema_v2.py"),
        ("server/telemetry_agent/auto_discovery.py", f"{RPI_DEST}/server/telemetry_agent/auto_discovery.py"),
    ]

    # Also deploy telemetry agent files for local testing on RPi4
    extra_files = [
        ("server/telemetry_agent/collector_v2.py", f"{RPI_DEST}/server/telemetry_agent/collector_v2.py"),
        ("server/telemetry_agent/mqtt_publisher_v2.py", f"{RPI_DEST}/server/telemetry_agent/mqtt_publisher_v2.py"),
        ("server/telemetry_agent/server_identity.py", f"{RPI_DEST}/server/telemetry_agent/server_identity.py"),
        ("server/telemetry_agent/plugins/__init__.py", f"{RPI_DEST}/server/telemetry_agent/plugins/__init__.py"),
        ("server/telemetry_agent/plugins/cpu_plugin.py", f"{RPI_DEST}/server/telemetry_agent/plugins/cpu_plugin.py"),
        ("server/telemetry_agent/plugins/gpu_nvidia_plugin.py", f"{RPI_DEST}/server/telemetry_agent/plugins/gpu_nvidia_plugin.py"),
        ("server/telemetry_agent/plugins/storage_plugin.py", f"{RPI_DEST}/server/telemetry_agent/plugins/storage_plugin.py"),
        ("server/telemetry_agent/plugins/memory_plugin.py", f"{RPI_DEST}/server/telemetry_agent/plugins/memory_plugin.py"),
        ("server/telemetry_agent/plugins/psu_fan_plugin.py", f"{RPI_DEST}/server/telemetry_agent/plugins/psu_fan_plugin.py"),
    ]

    # Create needed directories
    for d in [f"{RPI_DEST}/server/telemetry_agent",
              f"{RPI_DEST}/server/telemetry_agent/plugins"]:
        run_cmd(ssh, f"mkdir -p {d}")

    # Create __init__.py files
    for init_dir in [f"{RPI_DEST}/server/telemetry_agent"]:
        try:
            sftp.stat(f"{init_dir}/__init__.py")
        except FileNotFoundError:
            with sftp.open(f"{init_dir}/__init__.py", 'w') as f:
                f.write("")

    uploaded = 0
    for local_rel, remote_path in updated_files + extra_files:
        local_full = os.path.join(BASE_DIR, local_rel)
        if upload_file(sftp, local_full, remote_path):
            uploaded += 1

    sftp.close()
    print(f"\n  [OK] Deployed {uploaded} updated files")

    # ─── Step 4: Configure systemd service ────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: CONFIGURE SYSTEMD SERVICE")
    print("=" * 60)

    service_content = (
        "[Unit]\\n"
        "Description=Edge AI Predictive Maintenance Service\\n"
        "After=network.target mosquitto.service\\n"
        "Wants=mosquitto.service\\n"
        "\\n"
        "[Service]\\n"
        "Type=simple\\n"
        "User=pi\\n"
        f"WorkingDirectory={RPI_DEST}\\n"
        f"ExecStart={RPI_DEST}/venv/bin/python -m edge.main --broker 127.0.0.1 --port 1883 --dashboard-port 8080\\n"
        "Restart=always\\n"
        "RestartSec=5\\n"
        f"Environment=PYTHONPATH={RPI_DEST}\\n"
        "Environment=MQTT_BROKER=127.0.0.1\\n"
        "Environment=MQTT_PORT=1883\\n"
        f"Environment=CLOUD_OFFLOAD_URL={os.environ.get('CLOUD_INFERENCE_URL', 'http://loq.tail1ab139.ts.net:8000')}/api/offload\\n"
        f"Environment=CLOUD_CHAT_URL={os.environ.get('CLOUD_CHAT_URL', 'http://loq.tail1ab139.ts.net:8001/api/chat')}\\n"
        "Environment=API_KEY=pm-edge-ac0008caeb077588ba481e57e1bde8af\\n"
        "\\n"
        "[Install]\\n"
        "WantedBy=multi-user.target"
    )

    sudo_cmd(ssh, f'echo -e "{service_content}" > /etc/systemd/system/edge-ai.service', "WRITE SERVICE")
    sudo_cmd(ssh, "systemctl daemon-reload", "DAEMON RELOAD")
    sudo_cmd(ssh, "systemctl enable edge-ai", "ENABLE SERVICE")
    sudo_cmd(ssh, "systemctl restart edge-ai", "START SERVICE")

    time.sleep(5)
    run_cmd(ssh, "systemctl status edge-ai --no-pager 2>&1 | head -15", "SERVICE STATUS")

    # ─── Step 5: Verification ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: VERIFICATION")
    print("=" * 60)

    run_cmd(ssh, "systemctl is-active mosquitto", "MOSQUITTO")
    run_cmd(ssh, "systemctl is-active edge-ai", "EDGE-AI SERVICE")
    run_cmd(ssh, "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null; echo", "DASHBOARD HTTP")
    run_cmd(ssh, f"ls -la {RPI_DEST}/edge/models/", "MODEL FILES")
    run_cmd(ssh, f"{RPI_DEST}/venv/bin/python -c 'import numpy; print(f\"NumPy: {{numpy.__version__}}\")' 2>&1", "NUMPY")
    run_cmd(ssh, f"{RPI_DEST}/venv/bin/python -c 'import sklearn; print(f\"Sklearn: {{sklearn.__version__}}\")' 2>&1", "SKLEARN")
    run_cmd(ssh, f"{RPI_DEST}/venv/bin/python -c 'import pandas; print(f\"Pandas: {{pandas.__version__}}\")' 2>&1", "PANDAS")

    # Check tflite availability
    run_cmd(ssh, (
        f"{RPI_DEST}/venv/bin/python -c '"
        "try:\n"
        "    import tflite_runtime.interpreter as tflite; print(\"TFLite Runtime: OK\")\n"
        "except:\n"
        "    try:\n"
        "        from ai_edge_litert.interpreter import Interpreter; print(\"AI Edge LiteRT: OK\")\n"
        "    except:\n"
        "        print(\"TFLite: NOT AVAILABLE\")\n"
        "' 2>&1"
    ), "TFLITE")

    # Service logs
    sudo_cmd(ssh, "journalctl -u edge-ai --no-pager -n 30 2>&1 | tail -30", "SERVICE LOGS")

    ssh.close()

    print("\n" + "=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print(f"  Dashboard:  http://{RPI_HOST}:8080")
    print(f"  MQTT:       {RPI_HOST}:1883")
    print(f"  SSH:        ssh pi@{RPI_HOST}")
    print(f"  Service:    sudo systemctl status edge-ai")
    print("=" * 60)


if __name__ == "__main__":
    main()
