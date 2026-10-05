@echo off
title RPi4 Doctor Edge Node Launcher
color 0A
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo ============================================================
echo   Starting Raspberry Pi 4 Edge Node (Doctor) from LOQ
echo ============================================================
echo.

venv_loq_py311\Scripts\python.exe scripts\start_rpi4_service.py

echo.
pause
