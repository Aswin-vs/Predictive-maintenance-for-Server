import os
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('100.90.160.78', username='pi', password=os.environ.get('RPI4_PASS', ''), timeout=10)

cmds = [
    'ls -l /dev/i2c*',
    'lsmod',
    '/usr/sbin/i2cdetect -y 1 2>&1 || which i2cdetect',
    'grep -E "i2c|i2s|sound|voicehat" /boot/firmware/config.txt 2>/dev/null',
    'cat /proc/asound/cards'
]

for cmd in cmds:
    print(f"=== {cmd} ===")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8')
    err = stderr.read().decode('utf-8')
    if out:
        print(out.strip())
    if err:
        print("ERR:", err.strip())

ssh.close()
