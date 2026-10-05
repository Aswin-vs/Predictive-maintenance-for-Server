#!/usr/bin/env bash
# ==============================================================================
# Edge-AI Predictive Maintenance — Zero-Touch Patient Installer (Linux)
# Phase 3: Tailscale Mesh Networking & Automated Client Daemon
# ==============================================================================
# Usage:
#   sudo ./scripts/install_patient_linux.sh
#   Or with pre-set authkey:
#   sudo TS_AUTHKEY="tskey-auth-..." ./scripts/install_patient_linux.sh
# ==============================================================================

set -euo pipefail

# 1. Privilege Check
if [[ "${EUID}" -ne 0 ]]; then
    echo "[-] ERROR: This installer must be executed as root (or via sudo)." >&2
    exit 1
fi

echo "=============================================================================="
echo "  Edge-AI Predictive Maintenance — Patient Telemetry Installer (Linux)"
echo "=============================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="/opt/patient-telemetry"

# 2. Network / Dynamic Broker Resolution
EDGE_BROKER_HOST="${EDGE_BROKER_HOST:-rpi4-edge.tail1ab139.ts.net}"
EDGE_BROKER_PORT="${EDGE_BROKER_PORT:-1883}"

# Load configs/network.env if available in repo
if [[ -f "${REPO_ROOT}/configs/network.env" ]]; then
    while IFS='=' read -r key val || [[ -n "$key" ]]; do
        key=$(echo "$key" | tr -d ' \r\n')
        val=$(echo "$val" | tr -d ' \r\n"'\''')
        if [[ -n "$key" && ! "$key" =~ ^# ]]; then
            if [[ "$key" == "EDGE_BROKER_HOST" && -z "${EDGE_BROKER_HOST:-}" ]]; then
                EDGE_BROKER_HOST="$val"
            fi
            if [[ "$key" == "EDGE_BROKER_PORT" && -z "${EDGE_BROKER_PORT:-}" ]]; then
                EDGE_BROKER_PORT="$val"
            fi
        fi
    done < "${REPO_ROOT}/configs/network.env"
fi

echo "[+] Target Edge Broker: ${EDGE_BROKER_HOST}:${EDGE_BROKER_PORT}"

# 3. Tailscale Installation & Authentication
if ! command -v tailscale &>/dev/null; then
    echo "[+] Installing Tailscale via official script..."
    curl -fsSL https://tailscale.com/install.sh | sh
    systemctl enable --now tailscaled
else
    echo "[+] Tailscale binary found."
fi

# Ensure tailscaled is running
systemctl start tailscaled || true

# Check if already authenticated on tailnet
TS_STATUS=$(tailscale status --json 2>/dev/null | grep -Po '"BackendState":\s*"\K[^"]+' || echo "NeedsLogin")

if [[ "${TS_STATUS}" == "Running" ]]; then
    echo "[+] Tailscale is already connected to tailnet."
else
    TS_AUTHKEY="${TS_AUTHKEY:-}"
    if [[ -z "${TS_AUTHKEY}" ]]; then
        echo ""
        echo "--------------------------------------------------------------------------"
        echo " Tailscale Authentication Required"
        echo " If you have an Auth Key, enter it below. Otherwise press ENTER to generate"
        echo " an interactive web login link."
        echo "--------------------------------------------------------------------------"
        read -r -p "Enter TS_AUTHKEY (optional): " TS_AUTHKEY
        echo ""
    fi

    PATIENT_HOSTNAME="patient-$(hostname -s 2>/dev/null || hostname)"

    if [[ -n "${TS_AUTHKEY}" ]]; then
        echo "[+] Authenticating with Tailscale using provided Auth Key..."
        tailscale up --authkey="${TS_AUTHKEY}" --advertise-tags=tag:patient --hostname="${PATIENT_HOSTNAME}" --accept-routes
    else
        echo "[+] Starting interactive Tailscale authentication..."
        tailscale up --advertise-tags=tag:patient --hostname="${PATIENT_HOSTNAME}" --accept-routes
    fi
fi

# 4. System Package Dependencies
echo "[+] Ensuring system Python and virtual environment packages..."
if command -v apt-get &>/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv lm-sensors
elif command -v dnf &>/dev/null; then
    dnf install -y python3 python3-pip lm_sensors
elif command -v yum &>/dev/null; then
    yum install -y python3 python3-pip lm_sensors
fi

# 5. Application Directory Setup
echo "[+] Preparing application directory at ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}/plugins"
mkdir -p "${INSTALL_DIR}/server/training"

# Copy telemetry agent files
if [[ -d "${REPO_ROOT}/server/telemetry_agent" ]]; then
    echo "[+] Copying telemetry agent files from repository..."
    cp -r "${REPO_ROOT}/server/telemetry_agent/"* "${INSTALL_DIR}/" 2>/dev/null || true
    if [[ -d "${REPO_ROOT}/server/training" ]]; then
        cp "${REPO_ROOT}/server/training/feature_schema.py" "${INSTALL_DIR}/server/training/" 2>/dev/null || true
        cp "${REPO_ROOT}/server/training/feature_schema_v2.py" "${INSTALL_DIR}/server/training/" 2>/dev/null || true
    fi
else
    echo "[-] Warning: Repository source not found directly; ensuring placeholder structure..."
fi

# Create Python Virtual Environment
if [[ ! -d "${INSTALL_DIR}/venv" ]]; then
    echo "[+] Creating Python virtual environment..."
    python3 -m venv "${INSTALL_DIR}/venv"
fi

echo "[+] Installing Python dependencies in venv..."
"${INSTALL_DIR}/venv/bin/pip" install --upgrade pip --quiet
"${INSTALL_DIR}/venv/bin/pip" install --quiet psutil paho-mqtt

# 6. Systemd Service Configuration
SERVICE_FILE="/etc/systemd/system/patient-telemetry.service"
echo "[+] Writing systemd unit to ${SERVICE_FILE}..."

cat <<EOF > "${SERVICE_FILE}"
[Unit]
Description=Edge-AI Patient Server Telemetry Agent v2
After=network.target network-online.target tailscaled.service
Wants=network-online.target tailscaled.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment="PYTHONPATH=${INSTALL_DIR}"
Environment="EDGE_BROKER_HOST=${EDGE_BROKER_HOST}"
Environment="EDGE_BROKER_PORT=${EDGE_BROKER_PORT}"
ExecStart=${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/mqtt_publisher_v2.py --broker ${EDGE_BROKER_HOST} --port ${EDGE_BROKER_PORT} --mode production --interval 5
Restart=always
RestartSec=5
KillMode=process
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload and Start Service
echo "[+] Enabling and starting patient-telemetry.service..."
systemctl daemon-reload
systemctl enable patient-telemetry.service
systemctl restart patient-telemetry.service

# 7. Verification Gate
sleep 3
if systemctl is-active --quiet patient-telemetry.service; then
    echo ""
    echo "=============================================================================="
    echo "  [SUCCESS] Patient Telemetry Agent is Active and Running!"
    echo "  Systemd Service:  systemctl status patient-telemetry"
    echo "  Live Logs:        journalctl -u patient-telemetry -f"
    echo "  Target Broker:    ${EDGE_BROKER_HOST}:${EDGE_BROKER_PORT}"
    echo "=============================================================================="
else
    echo "[-] Warning: Service started but is not reporting active yet. Check logs:"
    journalctl -u patient-telemetry -n 20 --no-pager
fi
