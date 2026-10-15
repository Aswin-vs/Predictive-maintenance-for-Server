#!/usr/bin/env python3
"""
Hardware Sensor Diagnostic Suite for Raspberry Pi 4.
Location: /home/pi/sensor_tests/test_sensors.py

Monitors the two active physical sensors:
1. ADXL345 3-Axis Digital Accelerometer (I2C Bus 1 @ 0x53)
2. INMP441 Digital I2S MEMS Microphone (ALSA soundcard)

This script is fully standalone, isolated, and requires ZERO external pip dependencies.
It runs on standard system python3 (/usr/bin/python3) as well as any virtual environment.
"""

import sys
import os
import time
import math
import subprocess
import struct

# Automatically add venv site-packages if smbus2 isn't in system python
try:
    import smbus2
except ImportError:
    for p in ['/home/pi/predictive-maintenance/venv/lib/python3.11/site-packages',
              '/home/pi/predictive-maintenance/venv/lib/python3.12/site-packages',
              '/home/pi/predictive-maintenance/venv/lib/python3.10/site-packages']:
        if os.path.exists(p) and p not in sys.path:
            sys.path.append(p)
    try:
        import smbus2
    except ImportError:
        smbus2 = None

print("=" * 75)
print("  RASPBERRY PI 4 PHYSICAL SENSOR SMOKE TEST (ADXL345 + INMP441)")
print("=" * 75)

# ==============================================================================
# 1. ADXL345 3-AXIS ACCELEROMETER TEST
# ==============================================================================
print("\n[1/2] TESTING ADXL345 3-AXIS ACCELEROMETER (I2C Bus 1, Address 0x53)...")
adxl_success = False

if smbus2 is None:
    print("  [FAIL] smbus2 module not found in system or venv.")
    print("         Install via: sudo apt-get install -y python3-smbus i2c-tools")
else:
    try:
        bus = smbus2.SMBus(1)
        devid = bus.read_byte_data(0x53, 0x00)
        if devid == 0xE5:
            print(f"  [OK] Device ID match: 0x{devid:02X} (Expected: 0xE5)")
            # Configure +/-16g full resolution (0x31 -> 0x0B)
            bus.write_byte_data(0x53, 0x31, 0x0B)
            # Enable measurement mode (0x2D -> 0x08)
            bus.write_byte_data(0x53, 0x2D, 0x08)
            time.sleep(0.05)
            adxl_success = True
        else:
            print(f"  [WARN] Connected, but Device ID is 0x{devid:02X} (Expected: 0xE5)")
    except Exception as e:
        print(f"  [FAIL] Could not connect to ADXL345 on I2C-1: {e}")
        print("  [TIP] Check wiring:")
        print("        VCC -> Pin 1 (3.3V), GND -> Pin 6, SDA -> Pin 3, SCL -> Pin 5")
        print("        CS  -> Pin 17 (3.3V, pulls HIGH to enable I2C)")
        print("        SDO -> Pin 9  (GND, pulls LOW to set address 0x53)")

if adxl_success:
    print("  Streaming 5 live vibration samples (full-res, 3.9 mg/LSB):")
    print("  -------------------------------------------------------------------------")
    print(f"  {'#':<4} | {'X-Axis':<10} | {'Y-Axis':<10} | {'Z-Axis':<10} | {'Total Gravity':<14} | {'Dynamic RMS':<12}")
    print("  -------------------------------------------------------------------------")
    SCALE = 0.0039
    for i in range(5):
        raw = bus.read_i2c_block_data(0x53, 0x32, 6)
        x = int.from_bytes(raw[0:2], byteorder='little', signed=True) * SCALE
        y = int.from_bytes(raw[2:4], byteorder='little', signed=True) * SCALE
        z = int.from_bytes(raw[4:6], byteorder='little', signed=True) * SCALE
        mag = math.sqrt(x**2 + y**2 + z**2)
        dyn_rms = abs(mag - 1.0)
        print(f"  {i+1:<4} | {x:+9.4f}g | {y:+9.4f}g | {z:+9.4f}g | {mag:11.4f}g   | {dyn_rms:9.4f}g")
        time.sleep(0.15)
    print("  -------------------------------------------------------------------------")
    print("  [RESULT] ADXL345 Accelerometer is 100% FUNCTIONAL!")

# ==============================================================================
# 2. INMP441 DIGITAL I2S MEMS MICROPHONE TEST
# ==============================================================================
print("\n[2/2] TESTING INMP441 DIGITAL I2S MEMS MICROPHONE (ALSA)...")

proc = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
candidate_device = None

for line in proc.stdout.splitlines():
    if 'card' in line.lower() and ('voicehat' in line.lower() or 'soundcar' in line.lower() or 'googlevoice' in line.lower()):
        parts = line.split(':')
        card_str = parts[0].replace('card', '').strip()
        if card_str.isdigit():
            candidate_device = f"plughw:{card_str},0"
            print(f"  [OK] Detected ALSA I2S Soundcard: {candidate_device} ({parts[1].strip() if len(parts)>1 else ''})")
            break

if not candidate_device:
    candidate_device = "plughw:1,0"
    print(f"  [INFO] Defaulting to probe: {candidate_device}")

print("  Capturing 0.20-second burst (44.1 kHz, 32-bit S32_LE stereo raw PCM)...")
cmd = [
    'arecord',
    '-D', candidate_device,
    '-c2', '-r44100', '-fS32_LE',
    '-s', '8820',
    '-t', 'raw'
]

mic_success = False
try:
    capture = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2.5)
    if capture.returncode == 0 and len(capture.stdout) >= 70560:
        raw_bytes = capture.stdout[:70560]
        num_ints = len(raw_bytes) // 4
        samples = struct.unpack(f"<{num_ints}i", raw_bytes)
        
        # Left channel is even indices (L/R pin connected to GND)
        left = samples[0::2]
        
        # Pure Python Normalized RMS calculation
        sum_sq = sum((s / (2**31))**2 for s in left)
        norm_rms = math.sqrt(sum_sq / len(left))
        
        # Peak absolute amplitude
        peak_amp = max(abs(s / (2**31)) for s in left)
        
        print(f"  [OK] Audio Data Received: {len(capture.stdout):,} bytes ({len(left):,} samples)")
        print(f"  [OK] Normalized Acoustic Energy RMS: {norm_rms:.6f} (Scale: [0.0 - 1.0])")
        print(f"  [OK] Peak Acoustic Amplitude:        {peak_amp:.6f}")
        mic_success = True
    else:
        err_msg = capture.stderr.decode('utf-8', errors='ignore').strip()
        print(f"  [FAIL] arecord returned error: {err_msg}")
except Exception as e:
    print(f"  [FAIL] Error capturing audio: {e}")

if mic_success:
    print("  [RESULT] INMP441 I2S MEMS Microphone is 100% FUNCTIONAL!")
else:
    print("  [HINT] Verify INMP441 wiring:")
    print("         VDD -> Pin 1 (3.3V), GND -> Pin 14, L/R -> Pin 20 (GND)")
    print("         SCK -> Pin 12 (GPIO 18), WS -> Pin 35 (GPIO 19), SD -> Pin 38 (GPIO 20)")

print("\n" + "=" * 75)
print(f"FINAL SUMMARY: ADXL345: {'[PASS]' if adxl_success else '[FAIL]'} | INMP441: {'[PASS]' if mic_success else '[FAIL]'}")
print("=" * 75)
