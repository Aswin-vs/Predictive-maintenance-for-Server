@echo off
REM ============================================================
REM  BUILD PATIENT PACKAGE
REM  Creates a self-contained folder that can be copied to any
REM  Windows/Linux machine to turn it into a monitored patient.
REM ============================================================
title Building Patient Package...
color 0A
cd /d "%~dp0"

echo.
echo ================================================================
echo   BUILDING PATIENT TELEMETRY PACKAGE
echo   This creates: patient_package\
echo   Copy this folder to ANY machine to start monitoring it.
echo ================================================================
echo.

set "PKG=patient_package"

REM Clean old package (preserve launcher files by regenerating them)
if exist "%PKG%" (
    echo Cleaning old package files...
    REM Delete only the Python source files, keep launchers if re-creating
    del /q "%PKG%\*.py" 2>nul
    rd /s /q "%PKG%\plugins" 2>nul
    rd /s /q "%PKG%\server" 2>nul
)
if not exist "%PKG%" mkdir "%PKG%"
mkdir "%PKG%\plugins"

REM ---- Copy telemetry agent files ----
echo [1/5] Copying telemetry agent...
copy "server\telemetry_agent\mqtt_publisher_v2.py"   "%PKG%\" >nul
copy "server\telemetry_agent\collector_v2.py"         "%PKG%\" >nul
copy "server\telemetry_agent\auto_discovery.py"       "%PKG%\" >nul
copy "server\telemetry_agent\server_identity.py"      "%PKG%\" >nul
echo       OK

REM ---- Copy plugins ----
echo [2/5] Copying plugins...
copy "server\telemetry_agent\plugins\__init__.py"         "%PKG%\plugins\" >nul
copy "server\telemetry_agent\plugins\cpu_plugin.py"       "%PKG%\plugins\" >nul
copy "server\telemetry_agent\plugins\gpu_plugin.py"       "%PKG%\plugins\" >nul
copy "server\telemetry_agent\plugins\gpu_nvidia_plugin.py" "%PKG%\plugins\" >nul
copy "server\telemetry_agent\plugins\motherboard_plugin.py" "%PKG%\plugins\" >nul
copy "server\telemetry_agent\plugins\memory_plugin.py"    "%PKG%\plugins\" >nul
copy "server\telemetry_agent\plugins\psu_fan_plugin.py"   "%PKG%\plugins\" >nul
copy "server\telemetry_agent\plugins\storage_plugin.py"   "%PKG%\plugins\" >nul
echo       OK

REM ---- Copy feature schemas (needed by v2_to_v1 conversion) ----
echo [3/5] Copying feature schemas...
mkdir "%PKG%\server\training" 2>nul
mkdir "%PKG%\server\telemetry_agent" 2>nul
copy "server\training\feature_schema.py"    "%PKG%\server\training\" >nul
copy "server\training\feature_schema_v2.py" "%PKG%\server\training\" >nul
REM Create __init__.py files for package imports
echo. > "%PKG%\server\__init__.py"
echo. > "%PKG%\server\training\__init__.py"
echo. > "%PKG%\server\telemetry_agent\__init__.py"
echo       OK

REM ---- Create requirements and launchers if missing ----
echo [4/5] Checking launcher files...
if not exist "%PKG%\requirements.txt" (
    echo # Patient Telemetry Agent — Minimal Dependencies> "%PKG%\requirements.txt"
    echo psutil^>=5.9.0>> "%PKG%\requirements.txt"
    echo paho-mqtt^>=2.0.0>> "%PKG%\requirements.txt"
    echo       Created requirements.txt
) else (
    echo       requirements.txt OK
)
if not exist "%PKG%\START_PATIENT.bat" (
    echo       WARNING: START_PATIENT.bat missing! Re-create it manually.
) else (
    echo       START_PATIENT.bat OK
)
if not exist "%PKG%\INSTALL_DEPS.bat" (
    echo       WARNING: INSTALL_DEPS.bat missing! Re-create it manually.
) else (
    echo       INSTALL_DEPS.bat OK
)
if not exist "%PKG%\README.md" (
    echo       WARNING: README.md missing!
) else (
    echo       README.md OK
)
echo       Done

echo [5/5] Package built!
echo.
echo ================================================================
echo   PATIENT PACKAGE READY: %PKG%\
echo   
echo   To use on a new machine:
echo     1. Copy the '%PKG%' folder to the new machine
echo     2. Install Python 3.8+ if not already installed
echo     3. Run: install_deps.bat  (or: pip install -r requirements.txt)
echo     4. Run: start.bat         (or edit start.bat to set RPi4 broker host)
echo ================================================================
echo.
pause
