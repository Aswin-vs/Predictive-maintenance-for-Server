"""
RPi4 Fresh Setup Script -- Phase G

Sets up a freshly formatted Raspberry Pi 4 from scratch:
1. Tests SSH connectivity
2. Installs all system dependencies (Python, Mosquitto, etc.)
3. Creates project directory structure
4. Deploys all edge files via SFTP
5. Installs Python dependencies
6. Configures systemd service
7. Starts the edge-ai service

Run from LOQ:
    python setup_rpi4_fresh.py
"""

import os
import sys
import time
import paramiko

# ─── Configuration ─────────────────────────────────────────────────────────
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
SSH_TIMEOUT = 10

def ssh_connect():
    """Establish SSH connection to RPi4."""
    print(f"\n[SSH] Connecting to {RPI_HOST} as '{RPI_USER}'...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(RPI_HOST, username=RPI_USER, password=RPI_PASS, timeout=SSH_TIMEOUT)
        print(f"[SSH] Connected successfully!")
        return ssh
    except Exception as e:
        print(f"[ERROR] Cannot connect: {e}")
        print(f"  - Is RPi4 powered on and connected to the hotspot?")
        print(f"  - Can you ping {RPI_HOST}?")
        sys.exit(1)

def run_cmd(ssh, cmd, label="", timeout=120):
    """Run a command on RPi4 via SSH and print output."""
    if label:
        print(f"\n[{label}] {cmd}")
    else:
        print(f"  $ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        for line in out.split("\n"):
            print(f"    {line}")
    if err and exit_code != 0:
        for line in err.split("\n"):
            print(f"    [stderr] {line}")
    return exit_code, out, err

def upload_file(sftp, local_path, remote_path):
    """Upload a single file via SFTP, creating directories as needed."""
    if not os.path.exists(local_path):
        print(f"  [SKIP] Missing: {os.path.basename(local_path)}")
        return False
    remote_dir = os.path.dirname(remote_path)
    _ensure_remote_dir(sftp, remote_dir)
    print(f"  -> {os.path.basename(local_path)} -> {remote_path}")
    sftp.put(local_path, remote_path)
    return True

def _ensure_remote_dir(sftp, path):
    """Recursively create remote directories."""
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
    print("  RPi4 FRESH SETUP -- Predictive Maintenance Edge Node")
    print("=" * 60)

    ssh = ssh_connect()

    # ─── Step 1: System Info ───────────────────────────────────────
    run_cmd(ssh, "uname -a", "SYSTEM INFO")
    run_cmd(ssh, "cat /etc/os-release | head -3")
    run_cmd(ssh, "free -h | head -2")
    run_cmd(ssh, "df -h / | tail -1")
    run_cmd(ssh, "python3 --version 2>&1 || echo 'Python3 not found'")

    # ─── Step 2: System Updates & Dependencies ─────────────────────
    print("\n" + "=" * 60)
    print("  INSTALLING SYSTEM DEPENDENCIES")
    print("=" * 60)

    run_cmd(ssh, "sudo apt-get update -y", "APT UPDATE", timeout=180)
    run_cmd(ssh, (
        "sudo apt-get install -y "
        "python3 python3-pip python3-venv "
        "mosquitto mosquitto-clients "
        "i2c-tools python3-smbus "
        "libatlas-base-dev libopenblas-dev "
        "sqlite3 "
        "git curl wget"
    ), "APT INSTALL", timeout=300)

    # Enable I2C and SPI for sensors
    run_cmd(ssh, "sudo raspi-config nonint do_i2c 0", "ENABLE I2C")
    run_cmd(ssh, "sudo raspi-config nonint do_spi 0", "ENABLE SPI")

    # Configure Mosquitto for external connections
    run_cmd(ssh, (
        "echo 'listener 1883' | sudo tee /etc/mosquitto/conf.d/external.conf && "
        "echo 'allow_anonymous true' | sudo tee -a /etc/mosquitto/conf.d/external.conf"
    ), "MOSQUITTO CONFIG")
    run_cmd(ssh, "sudo systemctl enable mosquitto && sudo systemctl restart mosquitto", "MOSQUITTO START")

    # ─── Step 3: Create Project Directories ────────────────────────
    print("\n" + "=" * 60)
    print("  CREATING PROJECT STRUCTURE")
    print("=" * 60)

    dirs = [
        f"{RPI_DEST}/edge",
        f"{RPI_DEST}/edge/models",
        f"{RPI_DEST}/dashboard/assets",
        f"{RPI_DEST}/server/training",
        f"{RPI_DEST}/server/utils",
        f"{RPI_DEST}/configs",
        f"{RPI_DEST}/logs",
    ]
    for d in dirs:
        run_cmd(ssh, f"mkdir -p {d}")
    print("  [OK] Project directories created")

    # ─── Step 4: Deploy All Files via SFTP ─────────────────────────
    print("\n" + "=" * 60)
    print("  DEPLOYING FILES VIA SFTP")
    print("=" * 60)

    sftp = ssh.open_sftp()

    # File mapping: (local_path, remote_path)
    files = [
        # Edge Python core
        ("edge/main.py", f"{RPI_DEST}/edge/main.py"),
        ("edge/api_server.py", f"{RPI_DEST}/edge/api_server.py"),
        ("edge/edge_inference.py", f"{RPI_DEST}/edge/edge_inference.py"),
        ("edge/local_store.py", f"{RPI_DEST}/edge/local_store.py"),
        ("edge/mqtt_handler.py", f"{RPI_DEST}/edge/mqtt_handler.py"),
        ("edge/sensor_reader.py", f"{RPI_DEST}/edge/sensor_reader.py"),
        ("edge/edge_logger.py", f"{RPI_DEST}/edge/edge_logger.py"),

        # Edge Models (INT8 TFLite + IsoForest + Scaler)
        ("edge/models/classifier_int8.tflite", f"{RPI_DEST}/edge/models/classifier_int8.tflite"),
        ("edge/models/classifier_scaler.pkl", f"{RPI_DEST}/edge/models/classifier_scaler.pkl"),
        ("edge/models/iso_forest.pkl", f"{RPI_DEST}/edge/models/iso_forest.pkl"),

        # Dashboard
        ("dashboard/index.html", f"{RPI_DEST}/dashboard/index.html"),
        ("dashboard/assets/style.css", f"{RPI_DEST}/dashboard/assets/style.css"),
        ("dashboard/assets/app.js", f"{RPI_DEST}/dashboard/assets/app.js"),

        # Server schema contracts (v1 + v2)
        ("server/training/feature_schema.py", f"{RPI_DEST}/server/training/feature_schema.py"),
        ("server/training/feature_schema_v2.py", f"{RPI_DEST}/server/training/feature_schema_v2.py"),

        # Server utils (logger, auth -- needed by some imports)
        ("server/utils/logger.py", f"{RPI_DEST}/server/utils/logger.py"),
        ("server/utils/auth.py", f"{RPI_DEST}/server/utils/auth.py"),

        # Configs
        ("configs/edge.env", f"{RPI_DEST}/configs/edge.env"),
        ("configs/edge-ai.service", f"{RPI_DEST}/configs/edge-ai.service"),
    ]

    # Create __init__.py files for Python packages
    init_dirs = [
        f"{RPI_DEST}/edge",
        f"{RPI_DEST}/server",
        f"{RPI_DEST}/server/training",
        f"{RPI_DEST}/server/utils",
    ]

    uploaded = 0
    for local_rel, remote_path in files:
        local_full = os.path.join(BASE_DIR, local_rel)
        if upload_file(sftp, local_full, remote_path):
            uploaded += 1

    # Create __init__.py files
    for init_dir in init_dirs:
        init_path = f"{init_dir}/__init__.py"
        try:
            sftp.stat(init_path)
        except FileNotFoundError:
            with sftp.open(init_path, 'w') as f:
                f.write("")
            print(f"  -> Created {init_path}")

    sftp.close()
    print(f"\n  [OK] Deployed {uploaded} files")

    # ─── Step 5: Create Python Virtual Environment & Install Deps ──
    print("\n" + "=" * 60)
    print("  INSTALLING PYTHON DEPENDENCIES")
    print("=" * 60)

    run_cmd(ssh, f"python3 -m venv {RPI_DEST}/venv", "CREATE VENV", timeout=60)

    pip_packages = [
        "paho-mqtt",
        "fastapi",
        "uvicorn[standard]",
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "tflite-runtime",
        "requests",
        "aiofiles",
        "sse-starlette",
    ]
    pip_cmd = f"{RPI_DEST}/venv/bin/pip install --upgrade pip && {RPI_DEST}/venv/bin/pip install {' '.join(pip_packages)}"
    run_cmd(ssh, pip_cmd, "PIP INSTALL", timeout=600)

    # ─── Step 6: Configure Systemd Service ─────────────────────────
    print("\n" + "=" * 60)
    print("  CONFIGURING SYSTEMD SERVICE")
    print("=" * 60)

    # Create the systemd service file
    service_content = f"""[Unit]
Description=Edge AI Predictive Maintenance Service
After=network.target mosquitto.service
Wants=mosquitto.service

[Service]
Type=simple
User=pi
WorkingDirectory={RPI_DEST}
ExecStart={RPI_DEST}/venv/bin/python -m edge.main --broker 127.0.0.1 --port 1883 --dashboard-port 8080
Restart=always
RestartSec=5
Environment=PYTHONPATH={RPI_DEST}
Environment=MQTT_BROKER=127.0.0.1
Environment=MQTT_PORT=1883
Environment=CLOUD_OFFLOAD_URL={os.environ.get('CLOUD_INFERENCE_URL', 'http://loq.tail1ab139.ts.net:8000')}/api/offload
Environment=CLOUD_CHAT_URL={os.environ.get('CLOUD_CHAT_URL', 'http://loq.tail1ab139.ts.net:8001/api/chat')}
Environment=API_KEY=pm-edge-ac0008caeb077588ba481e57e1bde8af

[Install]
WantedBy=multi-user.target
"""
    # Write service file via SSH
    escaped = service_content.replace("'", "'\\''")
    run_cmd(ssh, f"echo '{escaped}' | sudo tee /etc/systemd/system/edge-ai.service", "SERVICE FILE")
    run_cmd(ssh, "sudo systemctl daemon-reload", "DAEMON RELOAD")
    run_cmd(ssh, "sudo systemctl enable edge-ai", "ENABLE SERVICE")
    run_cmd(ssh, "sudo systemctl restart edge-ai", "START SERVICE")

    # Wait and check
    time.sleep(3)
    run_cmd(ssh, "sudo systemctl status edge-ai --no-pager -l | head -15", "SERVICE STATUS")

    # ─── Step 7: Verify ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VERIFICATION")
    print("=" * 60)

    run_cmd(ssh, "mosquitto_pub -h 127.0.0.1 -t test -m 'hello' && echo 'MQTT: OK' || echo 'MQTT: FAILED'", "MQTT TEST")
    run_cmd(ssh, f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:8080/ && echo ' Dashboard: OK' || echo 'Dashboard: FAILED'", "DASHBOARD TEST")
    run_cmd(ssh, f"ls -la {RPI_DEST}/edge/models/", "MODEL FILES")
    run_cmd(ssh, f"{RPI_DEST}/venv/bin/python -c \"import tflite_runtime; print(f'TFLite Runtime: {{tflite_runtime.__version__}}')\"", "TFLITE CHECK")

    ssh.close()

    print("\n" + "=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print(f"  Dashboard: http://{RPI_HOST}:8080")
    print(f"  MQTT:      {RPI_HOST}:1883")
    print(f"  SSH:       ssh pi@{RPI_HOST}")
    print(f"  Service:   sudo systemctl status edge-ai")
    print("=" * 60)


if __name__ == "__main__":
    main()
