import os
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('100.90.160.78', username='pi', password=os.environ.get('RPI4_PASS', ''), timeout=10)

test_code = """
import smbus2, time, math

bus = smbus2.SMBus(1)
devid = bus.read_byte_data(0x53, 0x00)
print(f'ADXL345 DevID: {hex(devid)} (Expected: 0xe5)')

# Configure ADXL345: full resolution (+/-16g)
bus.write_byte_data(0x53, 0x31, 0x0B)
# Put in measurement mode
bus.write_byte_data(0x53, 0x2D, 0x08)
time.sleep(0.05)

for i in range(5):
    data = bus.read_i2c_block_data(0x53, 0x32, 6)
    x = int.from_bytes(data[0:2], byteorder='little', signed=True) * 0.0039
    y = int.from_bytes(data[2:4], byteorder='little', signed=True) * 0.0039
    z = int.from_bytes(data[4:6], byteorder='little', signed=True) * 0.0039
    mag = math.sqrt(x**2 + y**2 + z**2)
    dyn_rms = abs(mag - 1.0)
    print(f'Sample {i+1}: X={x:+.4f}g, Y={y:+.4f}g, Z={z:+.4f}g | Mag={mag:.4f}g | Dyn RMS={dyn_rms:.4f}g')
    time.sleep(0.2)
"""

# Run remote test
stdin, stdout, stderr = ssh.exec_command('python3 -c "' + test_code.replace('\n', '; ') + '"')
out = stdout.read().decode('utf-8')
err = stderr.read().decode('utf-8')
print("STDOUT:\n", out)
if err:
    print("STDERR:\n", err)

ssh.close()
