@echo off
title Patient Telemetry v2 (IdeaPad Local Execution)
color 0E
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM Load network config from configs/network.env
if exist "%~dp0configs\network.env" (
    for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
        set "%%A=%%B"
    )
)
if "%EDGE_BROKER_HOST%"=="" set "EDGE_BROKER_HOST=rpi4-edge.tail1ab139.ts.net"
if "%EDGE_BROKER_PORT%"=="" set "EDGE_BROKER_PORT=1883"

echo ============================================================
echo   IdeaPad S145 Patient Telemetry v2.0 (Local Machine)
echo   Auto-Discovery + Plugin-Based Collection + Hierarchical Schema
echo   Streaming hardware metrics to RPi4 (%EDGE_BROKER_HOST%:%EDGE_BROKER_PORT%)
echo ============================================================
echo.
echo Press Ctrl+C in this window at any time to stop.
echo.

python server\telemetry_agent\mqtt_publisher_v2.py --broker %EDGE_BROKER_HOST% --port %EDGE_BROKER_PORT% --server-name "IdeaPad-Patient" --interval 5

pause
