"""
Model Deployer — Pushes trained models from Cloud (LOQ) to Edge (RPi4).

Handles:
- SFTP file transfer of TFLite models + scalers to RPi4
- Deploy manifest generation for edge hot-swap
- Checksum verification
- Deployment status tracking via model_registry

Dependencies:
    pip install paramiko  (already used by start_all.py)

This module runs on the LOQ RTX 4060 (Cloud Backend).

Author: @claude-architect (Phase D — Production-Ready Upgrade)
Date: 2026-08-09
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from server.deep_analysis.model_registry import ModelRegistry

# Auto-load configs/network.env if present
_env_path = os.path.join(BASE_DIR, "configs", "network.env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"\'')
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# RPi4 connection defaults (dynamic resolution via Tailscale MagicDNS or network.env)
DEFAULT_RPI_HOST = os.environ.get("RPI4_HOST", os.environ.get("EDGE_BROKER_HOST", os.environ.get("RPI4_IP", "rpi4-edge.tail1ab139.ts.net")))
DEFAULT_RPI_USER = os.environ.get("RPI4_USER", "pi")
DEFAULT_RPI_PASS = os.environ.get("RPI4_PASS")
DEFAULT_RPI_MODELS_DIR = "/home/pi/predictive-maintenance/edge/models"


def compute_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def deploy_model(server_id: str, 
                 rpi_host: str = DEFAULT_RPI_HOST,
                 rpi_user: str = DEFAULT_RPI_USER,
                 rpi_pass: str = DEFAULT_RPI_PASS,
                 rpi_models_dir: str = DEFAULT_RPI_MODELS_DIR,
                 dry_run: bool = False):
    """
    Deploy the latest model for a server to the RPi4 edge node.
    
    Args:
        server_id: UUID of the server whose model to deploy
        rpi_host: RPi4 IP address
        rpi_user: SSH username
        rpi_pass: SSH password
        rpi_models_dir: Remote directory for models
        dry_run: If True, only show what would be transferred
    """
    registry = ModelRegistry()
    
    # Get the active model
    model = registry.get_active_model(server_id, "adapter")
    if not model:
        # Try base classifier
        model = registry.get_active_model(server_id, "classifier")
    
    if not model:
        print(f"[DEPLOY] No active model found for server {server_id}")
        return False
    
    tflite_path = model.get('tflite_path')
    scaler_path = model.get('scaler_path')
    
    if not tflite_path or not os.path.exists(tflite_path):
        print(f"[DEPLOY] TFLite model not found: {tflite_path}")
        return False
    
    # Prepare files to transfer
    files_to_transfer = []
    
    files_to_transfer.append({
        "local": tflite_path,
        "remote": f"{rpi_models_dir}/{os.path.basename(tflite_path)}",
        "checksum": compute_checksum(tflite_path),
        "size": os.path.getsize(tflite_path),
    })
    
    if scaler_path and os.path.exists(scaler_path):
        files_to_transfer.append({
            "local": scaler_path,
            "remote": f"{rpi_models_dir}/{os.path.basename(scaler_path)}",
            "checksum": compute_checksum(scaler_path),
            "size": os.path.getsize(scaler_path),
        })
    
    # Create deploy manifest
    manifest = {
        "server_id": server_id,
        "model_version": model['version'],
        "model_type": model['model_type'],
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "files": files_to_transfer,
        "val_accuracy": model.get('val_accuracy'),
        "val_mae": model.get('val_mae'),
    }
    
    manifest_path = os.path.join(os.path.dirname(tflite_path), 'deploy_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    files_to_transfer.append({
        "local": manifest_path,
        "remote": f"{rpi_models_dir}/deploy_manifest.json",
        "checksum": compute_checksum(manifest_path),
        "size": os.path.getsize(manifest_path),
    })
    
    print(f"\n[DEPLOY] Model Deployment Plan:")
    print(f"  Server:  {server_id[:12]}...")
    print(f"  Version: v{model['version']} ({model['model_type']})")
    print(f"  Target:  {rpi_user}@{rpi_host}:{rpi_models_dir}")
    print(f"  Files:")
    for f in files_to_transfer:
        print(f"    {os.path.basename(f['local'])} ({f['size']/1024:.1f} KB)")
    
    if dry_run:
        print("\n  [DRY RUN] No files transferred.")
        return True
    
    # Execute SFTP transfer
    try:
        import paramiko
    except ImportError:
        print("[DEPLOY] ERROR: paramiko not installed. pip install paramiko")
        return False
    
    print(f"\n  Connecting to {rpi_host}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(rpi_host, username=rpi_user, password=rpi_pass, timeout=10)
        sftp = ssh.open_sftp()
        
        # Ensure remote directory exists
        try:
            sftp.stat(rpi_models_dir)
        except FileNotFoundError:
            print(f"  Creating remote directory: {rpi_models_dir}")
            ssh.exec_command(f"mkdir -p {rpi_models_dir}")
        
        # Transfer files
        for file_info in files_to_transfer:
            local = file_info['local']
            remote = file_info['remote']
            print(f"  Uploading: {os.path.basename(local)} -> {remote}")
            sftp.put(local, remote)
        
        sftp.close()
        
        print(f"\n  [OK] Deployment complete! {len(files_to_transfer)} files transferred.")
        
        # Mark as deployed in registry
        registry.mark_deployed(model['id'])
        
        return True
        
    except Exception as e:
        print(f"\n  [FAILED] Deployment failed: {e}")
        return False
    finally:
        ssh.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Model to Edge (RPi4)")
    parser.add_argument("--server-id", type=str, required=True, help="Server UUID")
    parser.add_argument("--host", type=str, default=DEFAULT_RPI_HOST, help="RPi4 IP")
    parser.add_argument("--user", type=str, default=DEFAULT_RPI_USER, help="SSH user")
    parser.add_argument("--password", type=str, default=DEFAULT_RPI_PASS, help="SSH password")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without transferring")
    args = parser.parse_args()
    
    deploy_model(args.server_id, args.host, args.user, args.password, dry_run=args.dry_run)
