@echo off
title Patient Telemetry v2 (LOQ as Patient)
color 0E
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ---- Load Network Config ----
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
    set "%%A=%%B"
)

echo ============================================================
echo   LOQ Patient Telemetry v2.0 (Local Hardware Mode)
echo   Auto-Discovery + Plugin-Based Collection + Hierarchical Schema
echo   Streaming LOQ CPU/GPU/RAM metrics to RPi4 (%RPI4_IP%:%MQTT_PORT%)
echo ============================================================
echo.
echo Press Ctrl+C in this window at any time to stop.
echo.

venv_loq_py311\Scripts\python.exe server\telemetry_agent\mqtt_publisher_v2.py --broker %RPI4_IP% --port %MQTT_PORT% --server-name "LOQ-Patient" --interval 5

pause
