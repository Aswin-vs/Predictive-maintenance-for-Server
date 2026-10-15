import os
import sys
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('100.90.160.78', username='pi', password=os.environ.get('RPI4_PASS', ''), timeout=10)

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8')
    err = stderr.read().decode('utf-8')
    print(f"=== {cmd} ===")
    if out: print(out.strip())
    if err: print("ERR:", err.strip())

run("ps -eo pid,user,args | grep -E 'arecord|sensor|edge|python' | grep -v grep")
run("fuser -v /dev/snd/* 2>&1")
run("sudo fuser -v /dev/snd/* 2>&1")

ssh.close()
