"""
Multi-Machine File Manifest Generator
====================================
Generates a cryptographically verifiable manifest of all project files
containing relative path, size in bytes, ISO-8601 modification time,
and SHA-256 hash.

Used for multi-machine version skew detection between LOQ, RPi4, and Patient servers.
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import socket
from datetime import datetime, timezone
from typing import Dict, Any

# Root directory of the repository
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXCLUDE_DIRS = {
    ".git",
    "venv_loq_py311",
    "venv_loq",
    "venv",
    ".pytest_cache",
    ".system_generated",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
}

MAX_HASH_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit for full sha256 calculation


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        size = os.path.getsize(filepath)
        if size > MAX_HASH_SIZE_BYTES:
            # For massive datasets (>50MB), hash first 1MB + size to prevent long hashing delays
            with open(filepath, "rb") as f:
                h.update(f.read(1024 * 1024))
            h.update(str(size).encode())
            return f"partial_sha256_{h.hexdigest()}"

        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"error_{e}"


def scan_manifest(root_dir: str) -> Dict[str, Any]:
    """Scans all files and generates manifest."""
    files_map = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude directories in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith("venv")]

        for fname in sorted(filenames):
            if fname.endswith(".pyc") or fname == ".DS_Store":
                continue

            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root_dir).replace("\\", "/")

            try:
                stat = os.stat(full_path)
                mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                file_hash = compute_sha256(full_path)

                files_map[rel_path] = {
                    "size_bytes": stat.st_size,
                    "mtime_utc": mtime_iso,
                    "sha256": file_hash,
                }
            except Exception as e:
                files_map[rel_path] = {
                    "error": str(e)
                }

    hostname = socket.gethostname().lower().replace(" ", "_")
    return {
        "hostname": hostname,
        "platform": sys.platform,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(files_map),
        "files": files_map
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate repository file manifest")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    manifest = scan_manifest(ROOT_DIR)

    output_path = args.output
    if not output_path:
        audit_dir = os.path.join(ROOT_DIR, "audit")
        os.makedirs(audit_dir, exist_ok=True)
        hostname = manifest["hostname"]
        # Normalize LOQ / RPi4 name
        if "loq" in hostname:
            clean_name = "loq"
        elif "rpi" in hostname:
            clean_name = "rpi4"
        elif "ideapad" in hostname:
            clean_name = "ideapad"
        else:
            clean_name = hostname
        output_path = os.path.join(audit_dir, f"manifest_{clean_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest successfully generated for '{manifest['hostname']}' ({manifest['total_files']} files): {output_path}")


if __name__ == "__main__":
    main()
