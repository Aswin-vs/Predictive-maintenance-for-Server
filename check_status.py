import os
import sys
import paramiko

def load_network_env():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    net_env = os.path.join(base_dir, "configs", "network.env")
    if os.path.exists(net_env):
        with open(net_env, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k:
                        os.environ[k] = v

def check_remote(host, user, password, cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, username=user, password=password, timeout=10)
        print(f"--- {host} ---")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        err = stderr.read().decode()
        if err:
            print("STDERR:")
            print(err)
    except Exception as e:
        print(f"Failed to connect to {host}: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    load_network_env()
    rpi_host = os.environ.get("RPI4_HOST", "rpi4-edge.tail1ab139.ts.net")
    rpi_user = os.environ.get("RPI4_USER", "pi")
    rpi_pass = os.environ.get("RPI4_PASS")
    
    ideapad_host = os.environ.get("IDEAPAD_HOST", "ideapads145.tail1ab139.ts.net")
    ideapad_user = os.environ.get("IDEAPAD_USER", "Aswin")
    ideapad_pass = os.environ.get("IDEAPAD_PASS")

    check_remote(rpi_host, rpi_user, rpi_pass, "sudo systemctl status edge-ai; sudo journalctl -u edge-ai -n 50 --no-pager")
    check_remote(ideapad_host, ideapad_user, ideapad_pass, "powershell -Command \"Get-Process python -ErrorAction SilentlyContinue | Select-Object -Property Id, ProcessName, CommandLine\"")
