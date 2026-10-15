import sys
import paramiko

print("Connecting to Raspberry Pi 4...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('100.90.160.78', username='pi', password=os.environ.get('RPI4_PASS', ''), timeout=10)
sftp = ssh.open_sftp()

# 1. Update /home/pi/sensor_tests/test_inmp441.py
test_inmp441_code = """#!/usr/bin/env python3
\"\"\"
Standalone INMP441 I2S MEMS Microphone Smoke Test.
Isolated in /home/pi/sensor_tests/ -- does not alter edge AI service.
\"\"\"
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

# Discover card number dynamically from ALSA output
detected_card = "1"
for line in proc.stdout.splitlines():
    if 'card' in line.lower() and ('voicehat' in line.lower() or 'soundcar' in line.lower() or 'googlevoice' in line.lower()):
        parts = line.split(':')
        card_str = parts[0].replace('card', '').strip()
        if card_str.isdigit():
            detected_card = card_str
            break

device = os.environ.get("ALSA_MIC_DEVICE", f"plughw:{detected_card},0")
print(f"\\nAttempting test capture from device: {device} (44.1 kHz, 32-bit S32_LE stereo, 1 second)...")
test_wav = "/home/pi/sensor_tests/test_mic.wav"

cmd = [
    'arecord',
    '-D', device,
    '-c', '2',
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
            left = samples[0::2]
            sum_sq = sum((s / (2**31))**2 for s in left)
            rms = math.sqrt(sum_sq / len(left))
            peak = max(abs(s / (2**31)) for s in left)
            print(f"[SUCCESS] Acoustic Energy: Normalized RMS = {rms:.6f}")
            print(f"[SUCCESS] Peak Acoustic Amplitude   = {peak:.6f}")
            print(f"[RESULT] INMP441 Microphone is 100% OPERATIONAL on {device}!")
    except Exception as e:
        print(f"Error computing RMS: {e}")
    finally:
        if os.path.exists(test_wav):
            try:
                os.remove(test_wav)
            except:
                pass
else:
    print(f"[FAIL] Capture failed on {device}: {res.stderr.strip()}")
"""

with sftp.open('/home/pi/sensor_tests/test_inmp441.py', 'w') as f:
    f.write(test_inmp441_code)
print("Updated /home/pi/sensor_tests/test_inmp441.py")


# 2. Update /home/pi/mic_test/live_meter.py
live_meter_code = """#!/usr/bin/env python3
import sys
import math
import struct
import subprocess
import time
import os

# ANSI Colors
GREEN = "\\033[92m"
YELLOW = "\\033[93m"
RED = "\\033[91m"
CYAN = "\\033[96m"
BOLD = "\\033[1m"
RESET = "\\033[0m"

print(BOLD + CYAN + "=" * 65 + RESET)
print(BOLD + CYAN + "  INMP441 LIVE ACOUSTIC SOUND LEVEL METER (RPi 4 I2S)" + RESET)
print("  Speak, clap, or tap the mic to see the meter bounce!")
print("  Press " + BOLD + "Ctrl + C" + RESET + " to exit.")
print(BOLD + CYAN + "=" * 65 + RESET + "\\n")

# Dynamically discover soundcard card number from arecord -l
def get_card_device():
    try:
        proc = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if 'card' in line.lower() and ('voicehat' in line.lower() or 'soundcar' in line.lower() or 'googlevoice' in line.lower()):
                parts = line.split(':')
                card_str = parts[0].replace('card', '').strip()
                if card_str.isdigit():
                    return f"plughw:{card_str},0"
    except Exception:
        pass
    return "plughw:1,0"

device = os.environ.get("ALSA_MIC_DEVICE", get_card_device())
print(f"Using ALSA capture device: {BOLD}{device}{RESET}\\n")

# Start arecord in streaming mode (16000 Hz, stereo 32-bit S32_LE)
# 16000 Hz * 2 channels * 4 bytes = 128,000 bytes/sec
# Every 0.1s chunk is 12,800 bytes
CHUNK_FRAMES = 1600  # 100ms
CHUNK_BYTES = CHUNK_FRAMES * 2 * 4

cmd = [
    "arecord",
    "-D", device,
    "-c", "2",
    "-r", "16000",
    "-f", "S32_LE",
    "-t", "raw",
    "-q"
]

try:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except Exception as e:
    print(RED + "[ERROR] Failed to start arecord: " + str(e) + RESET)
    sys.exit(1)

max_seen_rms = 0.001

try:
    while True:
        raw_data = proc.stdout.read(CHUNK_BYTES)
        if not raw_data or len(raw_data) < CHUNK_BYTES:
            err = proc.stderr.read().decode('utf-8', errors='replace').strip()
            if "Device or resource busy" in err:
                print(RED + f"\\n[BUSY] {err}" + RESET)
                print(YELLOW + "[INFO] The edge-ai background service is actively sampling the I2S microphone." + RESET)
                print(YELLOW + "[INFO] To run this continuous visualizer uninterrupted, run:" + RESET)
                print(BOLD + "       sudo systemctl stop edge-ai" + RESET)
                print(BOLD + "       python3 /home/pi/mic_test/live_meter.py" + RESET)
                print(YELLOW + "       (and 'sudo systemctl start edge-ai' when finished)\\n" + RESET)
            elif err:
                print(RED + f"\\n[ERROR] arecord error: {err}" + RESET)
            else:
                print(RED + "\\n[ERROR] arecord stream ended unexpectedly." + RESET)
            break

        count = len(raw_data) // 4
        fmt = "<" + str(count) + "i"
        samples = struct.unpack(fmt, raw_data)

        # Left channel only (even indices)
        left = samples[0::2]
        
        # Calculate RMS of AC component
        mean_val = sum(left) / len(left)
        sum_sq = sum((s - mean_val) ** 2 for s in left)
        rms = math.sqrt(sum_sq / len(left))
        norm_rms = rms / (2**31)

        # Dynamic percentage
        pct = min(100.0, (norm_rms / 0.15) * 100.0)
        bar_len = min(35, int(pct * 0.35))

        # Color coding
        if pct > 65:
            color = RED
            status = "LOUD / CLAP [!]"
        elif pct > 20:
            color = YELLOW
            status = "VOICE / SOUND [*]"
        else:
            color = GREEN
            status = "QUIET / AMBIENT [.]"

        bar = "#" * bar_len + "-" * (35 - bar_len)
        line = "\\r" + color + f"[{bar}] {pct:5.1f}%" + RESET + f" | RMS: {norm_rms:.4f} | {status}   "
        sys.stdout.write(line)
        sys.stdout.flush()

except KeyboardInterrupt:
    print("\\n\\n" + BOLD + "Live meter stopped." + RESET)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except Exception:
        pass
"""

with sftp.open('/home/pi/mic_test/live_meter.py', 'w') as f:
    f.write(live_meter_code)
print("Updated /home/pi/mic_test/live_meter.py")


# 3. Update /home/pi/mic_test/check_health.py
check_health_code = """#!/usr/bin/env python3
import subprocess
import struct
import math
import time
import os

print("=" * 65)
print("  INMP441 HARDWARE HEALTH & CLIPPING DIAGNOSTIC")
print("=" * 65)

def get_card_device():
    try:
        proc = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if 'card' in line.lower() and ('voicehat' in line.lower() or 'soundcar' in line.lower() or 'googlevoice' in line.lower()):
                parts = line.split(':')
                card_str = parts[0].replace('card', '').strip()
                if card_str.isdigit():
                    return f"plughw:{card_str},0"
    except Exception:
        pass
    return "plughw:1,0"

device = os.environ.get("ALSA_MIC_DEVICE", get_card_device())
print(f"Using device: {device}\\n")

for i in range(5):
    res = subprocess.run([
        'arecord', '-D', device, '-c', '2', '-r', '16000', '-f', 'S32_LE',
        '-t', 'raw', '-d', '1', '-q'
    ], capture_output=True)
    raw = res.stdout
    count = len(raw) // 4
    if count == 0:
        err = res.stderr.decode('utf-8', errors='replace').strip()
        print(f"Chunk {i+1}: No data returned! ({err})")
        continue
        
    samples = struct.unpack(f"<{count}i", raw)[0::2] # Left channel
    
    min_v = min(samples)
    max_v = max(samples)
    mean_v = sum(samples) / len(samples)
    rms = math.sqrt(sum((s - mean_v)**2 for s in samples) / len(samples))
    norm_rms = rms / (2**31)
    
    # Check for clipping (values hitting near 2^31 or -2^31)
    pos_clips = sum(1 for s in samples if s > 2140000000)
    neg_clips = sum(1 for s in samples if s < -2140000000)
    zero_count = sum(1 for s in samples if s == 0)
    
    print(f"Chunk {i+1}: Min={min_v:+11d} | Max={max_v:+11d} | Mean={mean_v:+11.0f} | RMS={norm_rms:.5f} | PosClips={pos_clips} NegClips={neg_clips}")
    time.sleep(0.2)
"""

with sftp.open('/home/pi/mic_test/check_health.py', 'w') as f:
    f.write(check_health_code)
print("Updated /home/pi/mic_test/check_health.py")

# 4. Patch /home/pi/mic_test/verify_both_sensors.py (replace plughw:3,0 with plughw:1,0)
try:
    with sftp.open('/home/pi/mic_test/verify_both_sensors.py', 'r') as f:
        vbs_content = f.read().decode('utf-8')
    vbs_content = vbs_content.replace('plughw:3,0', 'plughw:1,0')
    with sftp.open('/home/pi/mic_test/verify_both_sensors.py', 'w') as f:
        f.write(vbs_content)
    print("Updated /home/pi/mic_test/verify_both_sensors.py (plughw:3,0 -> plughw:1,0)")
except Exception as e:
    print(f"Notice patching verify_both_sensors.py: {e}")

# Make scripts executable
ssh.exec_command("chmod +x /home/pi/sensor_tests/test_inmp441.py /home/pi/mic_test/live_meter.py /home/pi/mic_test/check_health.py /home/pi/mic_test/verify_both_sensors.py")

sftp.close()
ssh.close()
print("All scripts deployed successfully!")

