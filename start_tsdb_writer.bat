@echo off
title MQTT to TimescaleDB Ingestion Service
color 0D
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ---- Load Network Config ----
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
    set "%%A=%%B"
)

echo ============================================================
echo   MQTT to TimescaleDB Ingestion Service
echo   Subscribes to RPi4 MQTT (%EDGE_BROKER_HOST%:%MQTT_PORT%)
echo   Writes to TimescaleDB (localhost:5432)
echo ============================================================
echo.
echo Press Ctrl+C in this window at any time to stop.
echo.

venv_loq_py311\Scripts\python.exe server\deep_analysis\tsdb_writer.py --broker %EDGE_BROKER_HOST% --db-host localhost

pause
