#!/bin/bash
# ============================================================
#  PATIENT SERVER — ONE-CLICK STARTER (Linux/macOS)
#  Edit the BROKER_IP below to match your RPi4's IP address.
# ============================================================

# ====================== CONFIGURATION =========================
BROKER_IP="${EDGE_BROKER_HOST:-rpi4-edge.tail1ab139.ts.net}"
BROKER_PORT="${EDGE_BROKER_PORT:-1883}"
SERVER_NAME=""
INTERVAL=5
# ===========================================================

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo "================================================================"
echo "  PATIENT TELEMETRY AGENT v2.0"
echo "  Streaming hardware telemetry to RPi4 at $BROKER_IP:$BROKER_PORT"
echo "  Interval: every ${INTERVAL}s | Press Ctrl+C to stop"
echo "================================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found! Install: sudo apt install python3 python3-pip"
    exit 1
fi

# Check dependencies
python3 -c "import psutil, paho.mqtt.client" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARN] Missing dependencies. Installing..."
    pip3 install -r requirements.txt
    echo ""
fi

# Launch
if [ -z "$SERVER_NAME" ]; then
    python3 mqtt_publisher_v2.py --broker "$BROKER_IP" --port "$BROKER_PORT" --interval "$INTERVAL"
else
    python3 mqtt_publisher_v2.py --broker "$BROKER_IP" --port "$BROKER_PORT" --server-name "$SERVER_NAME" --interval "$INTERVAL"
fi
