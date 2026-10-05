"""
Start / Restart Raspberry Pi 4 Edge Node Service over SSH from LOQ.
Zero questions asked.
"""
import sys
import os
import paramiko

# Load network config from configs/network.env
def _load_network_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'network.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()

_load_network_env()

RPI4_HOST = os.environ.get("RPI4_HOST", os.environ.get("EDGE_BROKER_HOST", os.environ.get("RPI4_IP", "rpi4-edge.tail1ab139.ts.net")))
RPI4_USER = os.environ.get("RPI4_USER", "pi")
RPI4_PASS = os.environ.get("RPI4_PASS")

def start_rpi4():
    print("=" * 60)
    print("  STARTING RASPBERRY PI 4 (DOCTOR / EDGE NODE)")
    print(f"  Target: {RPI4_USER}@{RPI4_HOST}")
    print("=" * 60)
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print("Connecting to RPi4 via SSH...")
        ssh.connect(RPI4_HOST, username=RPI4_USER, password=RPI4_PASS, timeout=8.0)
        print("[SUCCESS] SSH Connected to RPi4!")
        
        print("Restarting 'edge-ai' systemd service...")
        sudo_pass = RPI4_PASS or ""
        cmd = f"echo '{sudo_pass}' | sudo -S systemctl restart edge-ai"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        stdout.channel.recv_exit_status()
        
        print("Checking service status...")
        stdin, stdout, stderr = ssh.exec_command("systemctl is-active edge-ai")
        status = stdout.read().decode().strip()
        
        if status == "active":
            print("\n" + "=" * 60)
            print("  [ONLINE] RPi4 Edge Node is ACTIVE and RUNNING!")
            print(f"  -> Operator Dashboard: http://{RPI4_HOST}:8080")
            print(f"  -> Mosquitto MQTT:     {RPI4_HOST}:1883")
            print("=" * 60)
        else:
            print(f"\n[WARN] Service status: {status}")
            stdin, stdout, stderr = ssh.exec_command("journalctl -u edge-ai -n 15 --no-pager")
            print(stdout.read().decode())
            
        ssh.close()
    except Exception as e:
        print(f"\n[ERROR] Could not connect to RPi4 at {RPI4_HOST}: {e}")
        print(f"Ensure RPi4 is powered on and reachable at {RPI4_HOST}.")

if __name__ == "__main__":
    start_rpi4()
