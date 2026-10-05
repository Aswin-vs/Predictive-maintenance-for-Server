"""
Automated Multi-Machine Deployment Script via SSH & SFTP.
Deploys updated Edge & Dashboard files to Raspberry Pi 4 (Doctor)
and Telemetry Agent files to IdeaPad S145 (Patient Server).
"""

import os
import sys
import paramiko

def deploy_to_linux(host, user, password, local_remote_pairs, post_cmd=None, timeout=5.0):
    print(f"\n[DEPLOY] Connecting to Linux host {host} as '{user}'...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, username=user, password=password, timeout=timeout)
        sftp = ssh.open_sftp()
        print(f"[DEPLOY] Connected successfully to {host}!")
        
        for local_path, remote_path in local_remote_pairs:
            if not os.path.exists(local_path):
                print(f"[WARN] Local file missing: {local_path}")
                continue
            remote_dir = os.path.dirname(remote_path).replace("\\", "/")
            try:
                ssh.exec_command(f"mkdir -p '{remote_dir}'")
            except Exception as e:
                pass
            print(f" -> Uploading: {os.path.basename(local_path)} -> {remote_path}")
            sftp.put(local_path, remote_path)
            
        sftp.close()
        
        if post_cmd:
            print(f"[EXEC] Running remote command: {post_cmd}", flush=True)
            try:
                stdin, stdout, stderr = ssh.exec_command(post_cmd, timeout=10.0)
                stdin.close()
                out = stdout.read().decode().strip()
                err = stderr.read().decode().strip()
                if out:
                    print(f"Output: {out}", flush=True)
                if err:
                    print(f"Stderr: {err}", flush=True)
            except Exception as cmd_err:
                print(f"[WARN] Command error or timeout: {cmd_err}", flush=True)
        ssh.close()
        print(f"[SUCCESS] Deployment to {host} completed successfully!", flush=True)
        return True
    except Exception as e:
        print(f"[FAILED] Could not connect to {host} ({e}). (Check if device is powered on and connected to hotspot)", flush=True)
        return False


def deploy_to_windows(host, user, password, local_remote_pairs, post_cmd=None, timeout=5.0):
    print(f"\n[DEPLOY] Connecting to Windows host {host} as '{user}'...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, username=user, password=password, timeout=timeout, look_for_keys=False, allow_agent=False)
        sftp = ssh.open_sftp()
        print(f"[DEPLOY] Connected successfully to {host}!")
        
        for local_path, remote_path in local_remote_pairs:
            if not os.path.exists(local_path):
                print(f"[WARN] Local file missing: {local_path}")
                continue
            remote_dir_win = os.path.dirname(remote_path).replace("/", "\\")
            remote_path_sftp = remote_path.replace("\\", "/")
            
            # Use powershell to create directory recursively
            try:
                stdin, stdout, stderr = ssh.exec_command(f"powershell -Command \"New-Item -ItemType Directory -Force -Path '{remote_dir_win}'\"")
                stdout.channel.recv_exit_status() # Wait for directory creation
            except Exception as e:
                pass
            print(f" -> Uploading: {os.path.basename(local_path)} -> {remote_path_sftp}")
            sftp.put(local_path, remote_path_sftp)
            
        sftp.close()
        
        if post_cmd:
            print(f"[EXEC] Running remote command: {post_cmd}")
            stdin, stdout, stderr = ssh.exec_command(post_cmd)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            if out:
                print(f"Output: {out}")
            if err:
                print(f"Stderr: {err}")
        ssh.close()
        print(f"[SUCCESS] Deployment to {host} completed successfully!")
        return True
    except Exception as e:
        print(f"[FAILED] Could not connect to {host} ({e}). (Check if OpenSSH server is running on IdeaPad)")
        return False


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load configs/network.env if present
    net_env = os.path.join(base_dir, "configs", "network.env")
    if os.path.exists(net_env):
        with open(net_env, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v

    rpi_host = os.environ.get("RPI4_HOST", "rpi4-edge.tail1ab139.ts.net")
    rpi_user = os.environ.get("RPI4_USER", "pi")
    rpi_pass = os.environ.get("RPI4_PASS")
    rpi_fallback = os.environ.get("RPI4_FALLBACK_HOST")

    ideapad_host = os.environ.get("IDEAPAD_HOST", "ideapads145.tail1ab139.ts.net")
    ideapad_user = os.environ.get("IDEAPAD_USER", "Aswin")
    ideapad_pass = os.environ.get("IDEAPAD_PASS")
    ideapad_fallback = os.environ.get("IDEAPAD_FALLBACK_HOST")

    loq_host = os.environ.get("LOQ_HOST", "loq.tail1ab139.ts.net")
    
    # ─── 1. Raspberry Pi 4 (Edge Node / Doctor) ───
    print("========================================")
    print(" 1. RASPBERRY PI 4 (DOCTOR / EDGE NODE) ")
    print("========================================")
    rpi_dest = "/home/pi/predictive-maintenance"
    rpi_files = [
        # Edge Python core
        (os.path.join(base_dir, "edge", "main.py"), f"{rpi_dest}/edge/main.py"),
        (os.path.join(base_dir, "edge", "api_server.py"), f"{rpi_dest}/edge/api_server.py"),
        (os.path.join(base_dir, "edge", "edge_inference.py"), f"{rpi_dest}/edge/edge_inference.py"),
        (os.path.join(base_dir, "edge", "local_store.py"), f"{rpi_dest}/edge/local_store.py"),
        (os.path.join(base_dir, "edge", "mqtt_handler.py"), f"{rpi_dest}/edge/mqtt_handler.py"),
        (os.path.join(base_dir, "edge", "sensor_reader.py"), f"{rpi_dest}/edge/sensor_reader.py"),
        (os.path.join(base_dir, "edge", "edge_logger.py"), f"{rpi_dest}/edge/edge_logger.py"),
        (os.path.join(base_dir, "edge", "adaptive_baseline.py"), f"{rpi_dest}/edge/adaptive_baseline.py"),
        (os.path.join(base_dir, "edge", "model_router.py"), f"{rpi_dest}/edge/model_router.py"),
        (os.path.join(base_dir, "edge", "ensemble_decision.py"), f"{rpi_dest}/edge/ensemble_decision.py"),
        
        # Edge Models
        (os.path.join(base_dir, "edge", "models", "classifier_int8.tflite"), f"{rpi_dest}/edge/models/classifier_int8.tflite"),
        (os.path.join(base_dir, "edge", "models", "classifier_scaler.pkl"), f"{rpi_dest}/edge/models/classifier_scaler.pkl"),
        (os.path.join(base_dir, "edge", "models", "iso_forest.pkl"), f"{rpi_dest}/edge/models/iso_forest.pkl"),
        (os.path.join(base_dir, "edge", "models", "fan_bearing_cnn_int8.tflite"), f"{rpi_dest}/edge/models/fan_bearing_cnn_int8.tflite"),
        (os.path.join(base_dir, "edge", "models", "fan_bearing_scaler.joblib"), f"{rpi_dest}/edge/models/fan_bearing_scaler.joblib"),
        (os.path.join(base_dir, "edge", "models", "tcn_storage_rul_int8.tflite"), f"{rpi_dest}/edge/models/tcn_storage_rul_int8.tflite"),
        (os.path.join(base_dir, "edge", "models", "tcn_storage_scaler.joblib"), f"{rpi_dest}/edge/models/tcn_storage_scaler.joblib"),
        (os.path.join(base_dir, "edge", "models", "stgcn_int8.tflite"), f"{rpi_dest}/edge/models/stgcn_int8.tflite"),
        
        # Revamped Dashboard UI
        (os.path.join(base_dir, "dashboard", "index.html"), f"{rpi_dest}/dashboard/index.html"),
        (os.path.join(base_dir, "dashboard", "assets", "style.css"), f"{rpi_dest}/dashboard/assets/style.css"),
        (os.path.join(base_dir, "dashboard", "assets", "app.js"), f"{rpi_dest}/dashboard/assets/app.js"),
        
        # Server schema contracts (v1 + v2)
        (os.path.join(base_dir, "server", "training", "feature_schema.py"), f"{rpi_dest}/server/training/feature_schema.py"),
        (os.path.join(base_dir, "server", "training", "feature_schema_v2.py"), f"{rpi_dest}/server/training/feature_schema_v2.py"),
        
        # Telemetry Agent & Plugins on RPi4 (for consistency)
        (os.path.join(base_dir, "server", "telemetry_agent", "collector_v2.py"), f"{rpi_dest}/server/telemetry_agent/collector_v2.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "mqtt_publisher_v2.py"), f"{rpi_dest}/server/telemetry_agent/mqtt_publisher_v2.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "__init__.py"), f"{rpi_dest}/server/telemetry_agent/plugins/__init__.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "cpu_plugin.py"), f"{rpi_dest}/server/telemetry_agent/plugins/cpu_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "gpu_plugin.py"), f"{rpi_dest}/server/telemetry_agent/plugins/gpu_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "gpu_nvidia_plugin.py"), f"{rpi_dest}/server/telemetry_agent/plugins/gpu_nvidia_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "motherboard_plugin.py"), f"{rpi_dest}/server/telemetry_agent/plugins/motherboard_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "storage_plugin.py"), f"{rpi_dest}/server/telemetry_agent/plugins/storage_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "memory_plugin.py"), f"{rpi_dest}/server/telemetry_agent/plugins/memory_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "psu_fan_plugin.py"), f"{rpi_dest}/server/telemetry_agent/plugins/psu_fan_plugin.py"),

        # Configs
        (os.path.join(base_dir, "configs", "edge.env"), f"{rpi_dest}/configs/edge.env"),
        (os.path.join(base_dir, "configs", "network.env"), f"{rpi_dest}/configs/network.env"),
        (os.path.join(base_dir, "configs", "edge-ai.service"), f"{rpi_dest}/configs/edge-ai.service"),
    ]
    rpi_target = rpi_host
    sudo_pass = rpi_pass or ""
    rpi_cmd = f"echo '{sudo_pass}' | sudo -S systemctl daemon-reload && echo '{sudo_pass}' | sudo -S systemctl restart edge-ai || echo 'Service edge-ai restarted'"
    rpi_success = deploy_to_linux(rpi_target, rpi_user, rpi_pass, rpi_files, rpi_cmd)
    if not rpi_success and rpi_fallback:
        print(f"[RETRY] Attempting fallback host: {rpi_fallback}")
        rpi_success = deploy_to_linux(rpi_fallback, rpi_user, rpi_pass, rpi_files, rpi_cmd)
        if rpi_success:
            rpi_target = rpi_fallback
    
    # ─── 2. IdeaPad S145 (Patient Server) ───
    print("\n========================================")
    print(" 2. IDEAPAD S145 (PATIENT SERVER)       ")
    print("========================================")
    ideapad_dest = "C:\\Users\\Aswin\\Documents\\Predictive_Maintenance_Patient"
    ideapad_files = [
        # Phase G v2 telemetry agent files
        (os.path.join(base_dir, "server", "telemetry_agent", "collector_v2.py"), f"{ideapad_dest}\\collector_v2.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "mqtt_publisher_v2.py"), f"{ideapad_dest}\\mqtt_publisher_v2.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "auto_discovery.py"), f"{ideapad_dest}\\auto_discovery.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "server_identity.py"), f"{ideapad_dest}\\server_identity.py"),
        
        # v2 Plugins
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "__init__.py"), f"{ideapad_dest}\\plugins\\__init__.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "cpu_plugin.py"), f"{ideapad_dest}\\plugins\\cpu_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "gpu_plugin.py"), f"{ideapad_dest}\\plugins\\gpu_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "gpu_nvidia_plugin.py"), f"{ideapad_dest}\\plugins\\gpu_nvidia_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "motherboard_plugin.py"), f"{ideapad_dest}\\plugins\\motherboard_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "storage_plugin.py"), f"{ideapad_dest}\\plugins\\storage_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "memory_plugin.py"), f"{ideapad_dest}\\plugins\\memory_plugin.py"),
        (os.path.join(base_dir, "server", "telemetry_agent", "plugins", "psu_fan_plugin.py"), f"{ideapad_dest}\\plugins\\psu_fan_plugin.py"),
        
        # Schema contracts (v1 + v2)
        (os.path.join(base_dir, "server", "training", "feature_schema.py"), f"{ideapad_dest}\\server\\training\\feature_schema.py"),
        (os.path.join(base_dir, "server", "training", "feature_schema.py"), f"{ideapad_dest}\\feature_schema.py"),
        (os.path.join(base_dir, "server", "training", "feature_schema_v2.py"), f"{ideapad_dest}\\server\\training\\feature_schema_v2.py"),
        (os.path.join(base_dir, "server", "training", "feature_schema_v2.py"), f"{ideapad_dest}\\feature_schema_v2.py"),
        
        # Configs
        (os.path.join(base_dir, "configs", "server.env"), f"{ideapad_dest}\\server.env"),
        (os.path.join(base_dir, "configs", "network.env"), f"{ideapad_dest}\\network.env"),
        (os.path.join(base_dir, "start_patient_ideapad_local.bat"), f"{ideapad_dest}\\start_patient_ideapad_local.bat"),
    ]
    ideapad_target = ideapad_host
    ideapad_cmd = 'cmd /c "schtasks /End /TN PredictiveMaintenancePatient & ping 127.0.0.1 -n 3 >nul & schtasks /Run /TN PredictiveMaintenancePatient"'
    ideapad_success = deploy_to_windows(ideapad_target, ideapad_user, ideapad_pass, ideapad_files, post_cmd=ideapad_cmd)
    if not ideapad_success and ideapad_fallback:
        print(f"[RETRY] Attempting fallback host: {ideapad_fallback}")
        ideapad_success = deploy_to_windows(ideapad_fallback, ideapad_user, ideapad_pass, ideapad_files, post_cmd=ideapad_cmd)
        if ideapad_success:
            ideapad_target = ideapad_fallback
    
    print("\n========================================")
    print(" 3. LENOVO LOQ (CLOUD / HOSPITAL)       ")
    print("========================================")
    print(f"Local workspace is active on LOQ (Host {loq_host}). All files up to date!")
    
    print("\n--- DEPLOYMENT SUMMARY ---")
    print(f"RPi4 ({rpi_target}): {'ONLINE & DEPLOYED' if rpi_success else 'OFFLINE (Connect to Tailnet/LAN to sync)'}")
    print(f"IdeaPad ({ideapad_target}): {'ONLINE & DEPLOYED' if ideapad_success else 'OFFLINE (Connect to Tailnet/LAN to sync)'}")
    print(f"LOQ Cloud ({loq_host}): UP TO DATE")
