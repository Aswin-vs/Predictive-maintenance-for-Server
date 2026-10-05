@echo off
title Patient Telemetry v2 (IdeaPad from LOQ via SSH)
color 0E
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ---- Load Network Config ----
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
    set "%%A=%%B"
)

echo ============================================================
echo   Starting IdeaPad S145 Patient Telemetry v2.0 from LOQ via SSH
echo   Target Patient: %IDEAPAD_USER%@%IDEAPAD_IP%
echo   Target Broker:  RPi4 Doctor (%RPI4_IP%:%MQTT_PORT%)
echo   Schema:         v2.0 (Auto-Discovery + Plugins + Hierarchical)
echo ============================================================
echo.
echo Press Ctrl+C in this window at any time to stop.
echo.

venv_loq_py311\Scripts\python.exe scripts\run_remote_patient.py --broker %RPI4_IP% --port %MQTT_PORT% --v2

pause
