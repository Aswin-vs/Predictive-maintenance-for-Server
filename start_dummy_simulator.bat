@echo off
title Dummy Telemetry Simulator
color 0C
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ---- Load Network Config ----
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
    set "%%A=%%B"
)

echo ============================================================
echo   Starting Dummy Telemetry Simulator
echo   Streaming anomalies to RPi4 Doctor (%RPI4_IP%:%MQTT_PORT%)
echo ============================================================
echo.
echo Press Ctrl+C in this window at any time to stop.
echo.

venv_loq_py311\Scripts\python.exe run_dummy_simulator.py --broker %RPI4_IP% --port %MQTT_PORT% --server-name "Simulator-01"

pause
