@echo off
REM ============================================================
REM  PATIENT SERVER — ONE-CLICK STARTER
REM  Edit the BROKER_IP below to match your RPi4's IP address.
REM ============================================================
title Patient Telemetry Agent
color 0E
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ====================== CONFIGURATION =========================
if "%EDGE_BROKER_HOST%"=="" set "EDGE_BROKER_HOST=rpi4-edge.tail1ab139.ts.net"
set "BROKER_IP=%EDGE_BROKER_HOST%"
set "BROKER_PORT=1883"
set "SERVER_NAME="
set "INTERVAL=5"
REM ===========================================================
REM  BROKER_IP   = IP of the Raspberry Pi 4 (MQTT broker)
REM  SERVER_NAME = Leave empty to auto-detect hostname
REM                 Or set to something like: My-Workstation
REM  INTERVAL    = Seconds between telemetry snapshots
REM ===========================================================

echo.
echo ================================================================
echo   PATIENT TELEMETRY AGENT v2.0
echo   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo   This machine's hardware will be monitored by the AI system.
echo   Telemetry is streamed to RPi4 at %BROKER_IP%:%BROKER_PORT%
echo   Publish interval: every %INTERVAL% seconds
echo ================================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Install Python 3.8+ and add to PATH.
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check dependencies
python -c "import psutil, paho.mqtt.client" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Missing dependencies. Installing now...
    pip install -r requirements.txt
    echo.
)

REM Build the command
if "%SERVER_NAME%"=="" (
    echo [INFO] Server name: auto-detect from hostname
    python mqtt_publisher_v2.py --broker %BROKER_IP% --port %BROKER_PORT% --interval %INTERVAL%
) else (
    echo [INFO] Server name: %SERVER_NAME%
    python mqtt_publisher_v2.py --broker %BROKER_IP% --port %BROKER_PORT% --server-name "%SERVER_NAME%" --interval %INTERVAL%
)

echo.
echo [STOPPED] Telemetry agent has stopped.
pause
