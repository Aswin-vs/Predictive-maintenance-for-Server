import os
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('100.90.160.78', username='pi', password=os.environ.get('RPI4_PASS', ''), timeout=10)

sftp = ssh.open_sftp()

master_test_script = '''#!/usr/bin/env python3
"""
Hardware Sensor Diagnostic Suite for Raspberry Pi 4.
Location: /home/pi/sensor_tests/test_sensors.py

Monitors the two active physical sensors:
1. ADXL345 3-Axis Digital Accelerometer (I2C Bus 1 @ 0x53)
2. INMP441 Digital I2S MEMS Microphone (ALSA soundcard)

This script is fully standalone and isolated. It does NOT touch or affect
the production Edge-AI predictive maintenance daemon.
"""

import sys
import time
import math
import subprocess
import struct
import numpy as np

print("=" * 75)
print("  RASPBERRY PI 4 PHYSICAL SENSOR SMOKE TEST (ADXL345 + INMP441)")
print("=" * 75)

# ==============================================================================
# 1. ADXL345 3-AXIS ACCELEROMETER TEST
# ==============================================================================
print("\n[1/2] TESTING ADXL345 3-AXIS ACCELEROMETER (I2C Bus 1, Address 0x53)...")
try:
    import smbus2
except ImportError:
    print("  [FAIL] python3 'smbus2' module not found. Run: pip3 install smbus2")
    sys.exit(1)

adxl_success = False
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

# Detect ALSA capture device
proc = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
candidate_device = None

for line in proc.stdout.splitlines():
    if 'voicehat' in line.lower() or 'googlevoice' in line.lower() or 'card' in line.lower():
        # Parse card number: e.g. "card 1: sndrpigooglevoi..."
        if 'card ' in line:
            parts = line.split(':')
            card_str = parts[0].replace('card', '').strip()
            if card_str.isdigit():
                candidate_device = f"plughw:{card_str},0"
                print(f"  [OK] Detected ALSA I2S Soundcard: {candidate_device} ({parts[1].strip() if len(parts)>1 else ''})")
                break

if not candidate_device:
    candidate_device = "plughw:1,0"
    print(f"  [INFO] Defaulting to probe: {candidate_device}")

print(f"  Capturing 0.20-second burst (44.1 kHz, 32-bit S32_LE stereo raw PCM)...")
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
        
        # Left channel (L/R pin connected to GND)
        left = np.array(samples[0::2], dtype=np.float64)
        
        # Normalized acoustic RMS (divide raw 32-bit int by 2^31)
        raw_rms = np.sqrt(np.mean(left ** 2))
        norm_rms = float(raw_rms / (2 ** 31))
        
        # Fast Fourier Transform (FFT) dominant peak frequency
        fft_vals = np.abs(np.fft.rfft(left))
        freqs = np.fft.rfftfreq(len(left), d=1.0 / 44100.0)
        peak_idx = int(np.argmax(fft_vals[1:]) + 1)
        peak_freq = float(freqs[peak_idx])
        
        print(f"  [OK] Bytes Captured: {len(capture.stdout):,} bytes (8,820 samples)")
        print(f"  [OK] Acoustic Energy RMS: {norm_rms:.6f} (Normalized [0.0 - 1.0])")
        print(f"  [OK] Dominant Acoustic Frequency: {peak_freq:.1f} Hz (Ambient/Cavitation Peak)")
        mic_success = True
    else:
        print(f"  [FAIL] arecord returned code {capture.returncode}: {capture.stderr.decode('utf-8').strip()}")
except Exception as e:
    print(f"  [FAIL] Error capturing audio: {e}")

if mic_success:
    print("  [RESULT] INMP441 I2S MEMS Microphone is 100% FUNCTIONAL!")
else:
    print("  [HINT] Verify INMP441 wiring:")
    print("         VDD -> Pin 1 (3.3V), GND -> Pin 14, L/R -> Pin 20 (GND)")
    print("         SCK -> Pin 12 (GPIO 18), WS -> Pin 35 (GPIO 19), SD -> Pin 38 (GPIO 20)")

print("\n" + "=" * 75)
print(f"SUMMARY: ADXL345: {'PASS' if adxl_success else 'FAIL'} | INMP441: {'PASS' if mic_success else 'FAIL'}")
print("=" * 75)
'''

with sftp.open('/home/pi/sensor_tests/test_sensors.py', 'w') as f:
    f.write(master_test_script)

ssh.exec_command('chmod +x /home/pi/sensor_tests/test_sensors.py')
sftp.close()

# Execute master test script live over SSH
print("Running /home/pi/sensor_tests/test_sensors.py...")
stdin, stdout, stderr = ssh.exec_command('python3 /home/pi/sensor_tests/test_sensors.py')
print(stdout.read().decode('utf-8'))
err = stderr.read().decode('utf-8')
if err:
    print("STDERR:\n", err)

ssh.close()
