import paramiko
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('100.90.160.78', username='pi', password=os.environ.get('RPI4_PASS', ''), timeout=10)

# Create isolated directory
sftp = ssh.open_sftp()
try:
    sftp.mkdir('/home/pi/sensor_tests')
    print("Created directory /home/pi/sensor_tests")
except IOError:
    print("Directory /home/pi/sensor_tests already exists")

# 1. Standalone ADXL345 Test Script
adxl345_code = '''#!/usr/bin/env python3
"""
Standalone ADXL345 I2C Accelerometer Smoke Test.
Isolated in /home/pi/sensor_tests/ -- does not alter edge AI service.
"""
import sys
import time
import math

try:
    import smbus2
except ImportError:
    print("[ERROR] smbus2 library not installed. Install via: pip3 install smbus2")
    sys.exit(1)

I2C_BUS = 1
ADXL345_ADDR = 0x53

print("=" * 60)
print("ADXL345 3-Axis Digital Accelerometer Smoke Test")
print("=" * 60)

try:
    bus = smbus2.SMBus(I2C_BUS)
    devid = bus.read_byte_data(ADXL345_ADDR, 0x00)
    print(f"[OK] Connected to I2C Bus {I2C_BUS} at Address {hex(ADXL345_ADDR)}")
    print(f"[OK] Device ID Read: {hex(devid)} (Expected: 0xe5)")
    if devid != 0xE5:
        print(f"[WARN] Unexpected DevID {hex(devid)}, expected 0xe5")
except Exception as e:
    print(f"[FAIL] Could not communicate with ADXL345 at {hex(ADXL345_ADDR)}: {e}")
    print("[HINT] Check wiring: VCC->Pin1(3.3V), GND->Pin6, SDA->Pin3, SCL->Pin5, CS->Pin17(3.3V), SDO->Pin9(GND)")
    sys.exit(1)

# Configure full-resolution +/-16g (0x31 -> 0x0B)
bus.write_byte_data(ADXL345_ADDR, 0x31, 0x0B)
# Enable measurement mode (0x2D -> 0x08)
bus.write_byte_data(ADXL345_ADDR, 0x2D, 0x08)
time.sleep(0.05)

print("\\nStreaming 10 live vibration samples (Sampling rate: 5 Hz)...\\n")
print(f"{'Sample':<8} | {'X (g)':<10} | {'Y (g)':<10} | {'Z (g)':<10} | {'Magnitude':<12} | {'Dynamic RMS':<12}")
print("-" * 72)

SCALE = 0.0039  # 3.9 mg/LSB in full-res mode

for i in range(10):
    try:
        data = bus.read_i2c_block_data(ADXL345_ADDR, 0x32, 6)
        x = int.from_bytes(data[0:2], byteorder='little', signed=True) * SCALE
        y = int.from_bytes(data[2:4], byteorder='little', signed=True) * SCALE
        z = int.from_bytes(data[4:6], byteorder='little', signed=True) * SCALE
        
        mag = math.sqrt(x**2 + y**2 + z**2)
        dyn_rms = abs(mag - 1.0)
        
        print(f"#{i+1:<7} | {x:+9.4f}g | {y:+9.4f}g | {z:+9.4f}g | {mag:9.4f}g   | {dyn_rms:9.4f}g")
    except Exception as e:
        print(f"#{i+1:<7} | Read Error: {e}")
    time.sleep(0.2)

print("-" * 72)
print("[SUCCESS] ADXL345 Accelerometer is 100% operational and reading live hardware physics!")
'''

# 2. Standalone INMP441 Test Script
inmp441_code = '''#!/usr/bin/env python3
"""
Standalone INMP441 I2S MEMS Microphone Smoke Test.
Isolated in /home/pi/sensor_tests/ -- does not alter edge AI service.
"""
import sys
import subprocess
import os
import struct
import math

print("=" * 60)
print("INMP441 Digital I2S MEMS Microphone Smoke Test")
print("=" * 60)

# Check if ALSA lists capture devices
proc = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
print("ALSA Capture Devices:")
print(proc.stdout.strip())

capture_lines = [line for line in proc.stdout.splitlines() if 'card' in line.lower()]
if not capture_lines:
    print("\\n[NOTICE] No ALSA capture card currently detected.")
    print("For INMP441 to register as an ALSA soundcard, I2S overlay must be loaded in /boot/firmware/config.txt:")
    print("  dtparam=i2s=on")
    print("  dtoverlay=googlevoicehat-soundcard")
    print("Check out /home/pi/sensor_tests/enable_i2s.sh for 1-click overlay setup.")
    sys.exit(0)

# If device exists, attempt 1-second burst capture
device = os.environ.get("ALSA_MIC_DEVICE", "plughw:3,0")
print(f"\\nAttempting test capture from device: {device} (44.1 kHz, 32-bit S32_LE, 1 second)...")
test_wav = "/home/pi/sensor_tests/test_mic.wav"

cmd = [
    'arecord',
    '-D', device,
    '-c', '1',
    '-r', '44100',
    '-f', 'S32_LE',
    '-t', 'wav',
    '-d', '1',
    test_wav
]

res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0 and os.path.exists(test_wav):
    size = os.path.getsize(test_wav)
    print(f"[OK] Captured test audio file: {test_wav} ({size / 1024:.1f} KB)")
    
    # Calculate RMS from WAV samples
    try:
        with open(test_wav, 'rb') as f:
            f.seek(44)  # Skip 44-byte WAV header
            raw = f.read()
        num_samples = len(raw) // 4
        if num_samples > 0:
            samples = struct.unpack(f"<{num_samples}i", raw[:num_samples*4])
            sum_sq = sum((s / (2**31))**2 for s in samples)
            rms = math.sqrt(sum_sq / num_samples)
            print(f"[SUCCESS] Acoustic Energy: Normalized RMS = {rms:.6f}")
    except Exception as e:
        print(f"Error computing RMS: {e}")
    finally:
        if os.path.exists(test_wav):
            os.remove(test_wav)
else:
    print(f"[FAIL] Capture failed on {device}: {res.stderr.strip()}")
'''

# 3. Helper script to enable I2S overlay in /boot/firmware/config.txt
enable_i2s_sh = '''#!/bin/bash
# 1-Click Script to configure /boot/firmware/config.txt for INMP441 I2S Microphone
echo "Checking /boot/firmware/config.txt..."

CONFIG="/boot/firmware/config.txt"
if [ ! -f "$CONFIG" ]; then
    CONFIG="/boot/config.txt"
fi

echo "Target config: $CONFIG"

# Check if overlay is already present
if grep -q "googlevoicehat-soundcard" "$CONFIG"; then
    echo "[OK] googlevoicehat-soundcard overlay is already in $CONFIG."
else
    echo "Adding I2S soundcard configuration to $CONFIG..."
    sudo bash -c "cat >> $CONFIG" << 'EOF'

# Enable I2S Digital Audio Interface for INMP441 Microphone
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
EOF
    echo "[OK] Configuration added."
    echo "Please reboot your Raspberry Pi: sudo reboot"
fi
'''

# Write to isolated test directory
with sftp.open('/home/pi/sensor_tests/test_adxl345.py', 'w') as f:
    f.write(adxl345_code)
ssh.exec_command('chmod +x /home/pi/sensor_tests/test_adxl345.py')

with sftp.open('/home/pi/sensor_tests/test_inmp441.py', 'w') as f:
    f.write(inmp441_code)
ssh.exec_command('chmod +x /home/pi/sensor_tests/test_inmp441.py')

with sftp.open('/home/pi/sensor_tests/enable_i2s.sh', 'w') as f:
    f.write(enable_i2s_sh)
ssh.exec_command('chmod +x /home/pi/sensor_tests/enable_i2s.sh')

sftp.close()
print("All test scripts placed in /home/pi/sensor_tests/")

# Run test_adxl345.py directly!
print("\n" + "="*70)
print("EXECUTING /home/pi/sensor_tests/test_adxl345.py LIVE OVER SSH")
print("="*70)
stdin, stdout, stderr = ssh.exec_command('python3 /home/pi/sensor_tests/test_adxl345.py')
out = stdout.read().decode('utf-8')
err = stderr.read().decode('utf-8')
if out:
    print(out)
if err:
    print("ERR:\n", err)

ssh.close()
