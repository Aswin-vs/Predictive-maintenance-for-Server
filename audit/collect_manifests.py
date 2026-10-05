"""
Multi-Machine Manifest Collector & Diff Analyzer
================================================
Transfers `generate_manifest.py` to Raspberry Pi 4 via SSH/SFTP, executes
it to generate `audit/manifest_rpi4.json`, downloads the manifest back to LOQ,
and performs a comprehensive version skew analysis across nodes.
"""

from __future__ import annotations
import os
import sys
import json
import socket
import paramiko
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(BASE_DIR, "audit")


def load_env_file(env_path: str):
    """Loads key=value pairs into os.environ if not already present."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


def fetch_rpi4_manifest(host: str, user: str, password: str, remote_dir: str = "/home/pi/predictive-maintenance") -> str:
    """Connects to RPi4 via SSH, runs generate_manifest.py, and downloads manifest_rpi4.json."""
    print(f"\n[MANIFEST] Connecting to RPi4 at {host} as '{user}'...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(host, username=user, password=password, timeout=10.0)
        sftp = ssh.open_sftp()
        print(f"[MANIFEST] Connected successfully to {host}!")

        # Ensure remote audit directory exists
        remote_audit_dir = f"{remote_dir}/audit"
        try:
            ssh.exec_command(f"mkdir -p '{remote_audit_dir}'")
        except Exception:
            pass

        # 1. Upload generate_manifest.py
        local_script = os.path.join(AUDIT_DIR, "generate_manifest.py")
        remote_script = f"{remote_audit_dir}/generate_manifest.py"
        print(f" -> Uploading manifest generator: {local_script} -> {remote_script}")
        sftp.put(local_script, remote_script)

        # 2. Execute generate_manifest.py on RPi4
        remote_out_path = f"{remote_audit_dir}/manifest_rpi4.json"
        cmd = f"python3 {remote_script} --output {remote_out_path}"
        print(f"[EXEC] Running on RPi4: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30.0)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(f"RPi4 stdout: {out}")
        if err:
            print(f"RPi4 stderr: {err}")

        # 3. Download manifest_rpi4.json
        local_out_path = os.path.join(AUDIT_DIR, "manifest_rpi4.json")
        print(f" <- Downloading: {remote_out_path} -> {local_out_path}")
        sftp.get(remote_out_path, local_out_path)

        sftp.close()
        ssh.close()
        print(f"[SUCCESS] Remote manifest fetched and saved to: {local_out_path}")
        return local_out_path
    except Exception as e:
        print(f"[ERROR] Failed to fetch manifest from RPi4: {e}")
        return None


def diff_manifests(loq_manifest_path: str, rpi4_manifest_path: str) -> Dict[str, Any]:
    """Compares LOQ and RPi4 manifests to identify identical, skewed, and machine-unique files."""
    with open(loq_manifest_path, "r", encoding="utf-8") as f:
        loq = json.load(f)
    with open(rpi4_manifest_path, "r", encoding="utf-8") as f:
        rpi4 = json.load(f)

    loq_files = loq.get("files", {})
    rpi4_files = rpi4.get("files", {})

    all_keys = set(loq_files.keys()).union(set(rpi4_files.keys()))

    common_identical = []
    version_skew = []
    loq_only = []
    rpi4_only = []

    for path in sorted(all_keys):
        in_loq = path in loq_files
        in_rpi4 = path in rpi4_files

        if in_loq and in_rpi4:
            loq_info = loq_files[path]
            rpi4_info = rpi4_files[path]
            loq_sha = loq_info.get("sha256")
            rpi4_sha = rpi4_info.get("sha256")

            if loq_sha and rpi4_sha and loq_sha == rpi4_sha:
                common_identical.append({
                    "path": path,
                    "size_bytes": loq_info.get("size_bytes"),
                    "sha256": loq_sha
                })
            else:
                version_skew.append({
                    "path": path,
                    "loq_size": loq_info.get("size_bytes"),
                    "rpi4_size": rpi4_info.get("size_bytes"),
                    "loq_sha256": loq_sha,
                    "rpi4_sha256": rpi4_sha,
                    "loq_mtime": loq_info.get("mtime_utc"),
                    "rpi4_mtime": rpi4_info.get("mtime_utc"),
                })
        elif in_loq:
            loq_only.append({
                "path": path,
                "size_bytes": loq_files[path].get("size_bytes")
            })
        else:
            rpi4_only.append({
                "path": path,
                "size_bytes": rpi4_files[path].get("size_bytes")
            })

    diff_summary = {
        "loq_total_files": loq.get("total_files"),
        "rpi4_total_files": rpi4.get("total_files"),
        "common_identical_count": len(common_identical),
        "version_skew_count": len(version_skew),
        "loq_only_count": len(loq_only),
        "rpi4_only_count": len(rpi4_only),
        "version_skew_files": version_skew,
        "loq_only_files": loq_only,
        "rpi4_only_files": rpi4_only
    }

    diff_path = os.path.join(AUDIT_DIR, "manifest_diff.json")
    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(diff_summary, f, indent=2)

    print(f"\n[MANIFEST DIFF SUMMARY]")
    print(f"  LOQ Total Files:        {diff_summary['loq_total_files']}")
    print(f"  RPi4 Total Files:       {diff_summary['rpi4_total_files']}")
    print(f"  Common Identical Files: {diff_summary['common_identical_count']}")
    print(f"  Version Skew Files:     {diff_summary['version_skew_count']}")
    print(f"  LOQ-Only Files:         {diff_summary['loq_only_count']} (cloud trainers, datasets, server backend)")
    print(f"  RPi4-Only Files:        {diff_summary['rpi4_only_count']} (local logs, sqlite cache, edge runtime)")
    print(f"  Saved diff report to:   {diff_path}")

    if version_skew:
        print("\n[VERSION SKEW DETECTED]")
        for item in version_skew[:10]:
            print(f"  * {item['path']} (LOQ: {item['loq_size']}B, RPi4: {item['rpi4_size']}B)")
        if len(version_skew) > 10:
            print(f"  ... and {len(version_skew) - 10} more files.")
    else:
        print("\n[PERFECT ALIGNMENT] Zero version skew among shared deployed files!")

    return diff_summary


def main():
    # Load network configs
    load_env_file(os.path.join(BASE_DIR, "configs", "network.env"))

    rpi_host = os.environ.get("RPI4_HOST", "rpi4-edge.tail1ab139.ts.net")
    rpi_user = os.environ.get("RPI4_USER", "pi")
    rpi_pass = os.environ.get("RPI4_PASS")

    # 1. Ensure LOQ manifest exists
    loq_manifest = os.path.join(AUDIT_DIR, "manifest_loq.json")
    if not os.path.exists(loq_manifest):
        print("[MANIFEST] Generating local LOQ manifest...")
        from audit.generate_manifest import scan_manifest, ROOT_DIR
        manifest = scan_manifest(ROOT_DIR)
        with open(loq_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # 2. Fetch RPi4 manifest
    rpi4_manifest = fetch_rpi4_manifest(rpi_host, rpi_user, rpi_pass)
    if not rpi4_manifest or not os.path.exists(rpi4_manifest):
        print("[WARN] Could not retrieve RPi4 manifest remotely. Checking existing manifest_rpi4.json...")
        rpi4_manifest = os.path.join(AUDIT_DIR, "manifest_rpi4.json")

    if os.path.exists(rpi4_manifest):
        diff_manifests(loq_manifest, rpi4_manifest)
    else:
        print("[ERROR] manifest_rpi4.json not available for diff.")


if __name__ == "__main__":
    main()
