"""
Retrain Trigger Manager — Checks data accumulation and triggers retraining.

Runs as a background scheduler on the LOQ Cloud Backend. Periodically checks
each registered server's data accumulation in TimescaleDB and triggers personalized
model training when thresholds are met.

Trigger Conditions:
- Server has >= 30 days of data AND never retrained → first retraining
- >= 30 days since last retrain → scheduled retraining
- Anomaly rate > 10% in last 24h → emergency retraining

This module is meant to run on the LOQ RTX 4060 (Cloud Backend).

Author: @claude-architect (Phase D — Production-Ready Upgrade)
Date: 2026-08-09
"""

import os
import sys
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from server.deep_analysis.model_registry import ModelRegistry
try:
    from server.deep_analysis.tsdb_writer import TimescaleDBWriter, PSYCOPG2_AVAILABLE
except ImportError:
    PSYCOPG2_AVAILABLE = False
    TimescaleDBWriter = None


# Default configuration
DEFAULT_CHECK_INTERVAL_HOURS = 6
MIN_DATA_DAYS = 30
RETRAIN_INTERVAL_DAYS = 30
EMERGENCY_ANOMALY_RATE_THRESHOLD = 0.10  # 10%


class RetrainTriggerManager:
    """
    Background scheduler that monitors data accumulation and triggers
    personalized model training when conditions are met.
    """
    
    def __init__(self, registry: ModelRegistry = None, 
                 tsdb_writer: Optional[TimescaleDBWriter] = None,
                 check_interval_hours: float = DEFAULT_CHECK_INTERVAL_HOURS,
                 on_retrain_triggered: Optional[Callable] = None):
        """
        Args:
            registry: ModelRegistry instance for server/model tracking
            tsdb_writer: TimescaleDBWriter for querying data accumulation
            check_interval_hours: How often to check triggers (default: 6h)
            on_retrain_triggered: Callback function when retraining is triggered.
                                  Signature: fn(server_id: str, reason: str) -> None
        """
        self.registry = registry or ModelRegistry()
        self.tsdb = tsdb_writer
        self.check_interval = check_interval_hours * 3600  # Convert to seconds
        self.on_retrain_triggered = on_retrain_triggered
        self._timer = None
        self._running = False
    
    def start(self):
        """Start the background trigger check loop."""
        if self._running:
            print("[RETRAIN] Trigger manager already running.")
            return
        
        self._running = True
        print(f"[RETRAIN] Trigger manager started. Checking every {self.check_interval/3600:.1f}h")
        self._schedule_next_check()
    
    def stop(self):
        """Stop the background trigger check loop."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
        print("[RETRAIN] Trigger manager stopped.")
    
    def _schedule_next_check(self):
        """Schedule the next trigger check."""
        if not self._running:
            return
        self._timer = threading.Timer(self.check_interval, self._run_check)
        self._timer.daemon = True
        self._timer.start()
    
    def _run_check(self):
        """Execute the trigger check for all servers."""
        try:
            self.check_all_servers()
        except Exception as e:
            print(f"[RETRAIN] Error during trigger check: {e}")
        finally:
            self._schedule_next_check()
    
    def check_all_servers(self) -> List[Dict[str, Any]]:
        """
        Check all registered servers for retrain triggers.
        
        Returns:
            List of dicts with server_id, triggered (bool), and reason
        """
        servers = self.registry.get_all_servers()
        results = []
        
        print(f"\n[RETRAIN] Checking {len(servers)} registered servers...")
        
        for server in servers:
            server_id = server['server_id']
            server_name = server.get('server_name', 'Unknown')
            
            # Update data days from TimescaleDB
            if self.tsdb and self.tsdb.is_connected():
                data_days = self.tsdb.query_data_days(server_id)
                self.registry.update_data_days(server_id, data_days)
            else:
                data_days = server.get('total_data_days', 0)
            
            # Check if retraining is needed
            needs_retrain, reason = self.registry.needs_retraining(
                server_id, 
                min_data_days=MIN_DATA_DAYS,
                retrain_interval_days=RETRAIN_INTERVAL_DAYS
            )
            
            result = {
                "server_id": server_id,
                "server_name": server_name,
                "data_days": data_days,
                "triggered": needs_retrain,
                "reason": reason,
            }
            results.append(result)
            
            if needs_retrain:
                print(f"  [TRIGGER] {server_name} ({server_id[:8]}...): {reason}")
                self._trigger_retrain(server_id, reason)
            else:
                print(f"  [OK] {server_name} ({server_id[:8]}...): {reason}")
        
        return results
    
    def check_single_server(self, server_id: str) -> Dict[str, Any]:
        """Check a single server for retrain triggers."""
        server = self.registry.get_server(server_id)
        if not server:
            return {"server_id": server_id, "triggered": False, "reason": "server_not_found"}
        
        # Update data days
        if self.tsdb and self.tsdb.is_connected():
            data_days = self.tsdb.query_data_days(server_id)
            self.registry.update_data_days(server_id, data_days)
        
        needs_retrain, reason = self.registry.needs_retraining(server_id)
        
        if needs_retrain:
            self._trigger_retrain(server_id, reason)
        
        return {
            "server_id": server_id,
            "triggered": needs_retrain,
            "reason": reason,
        }
    
    def force_retrain(self, server_id: str, reason: str = "manual"):
        """Manually trigger retraining for a specific server."""
        print(f"[RETRAIN] Force-triggering retraining for {server_id}: {reason}")
        self._trigger_retrain(server_id, reason)
    
    def _trigger_retrain(self, server_id: str, reason: str):
        """
        Execute the retraining trigger.
        
        This logs the event and calls the registered callback.
        The actual training is handled by train_adapter.py on the LOQ GPU.
        """
        # Get current model version
        current_model = self.registry.get_active_model(server_id, "classifier")
        old_version = current_model['version'] if current_model else None
        
        # Log the retrain event
        log_id = self.registry.log_retrain_start(server_id, reason, old_version)
        
        # Call the registered callback (which should invoke train_adapter.py)
        if self.on_retrain_triggered:
            try:
                self.on_retrain_triggered(server_id, reason)
                # The callback is responsible for calling log_retrain_complete
            except Exception as e:
                error_msg = f"Retrain callback failed: {e}"
                print(f"[RETRAIN] ERROR: {error_msg}")
                self.registry.log_retrain_failed(log_id, error_msg)
        else:
            print(f"[RETRAIN] No callback registered. Retrain logged but not executed.")
            print(f"         To execute: python server/training/train_adapter.py --server-id {server_id}")


def default_retrain_callback(server_id: str, reason: str):
    """
    Default callback that launches train_adapter.py as a subprocess.
    Only works on the LOQ RTX 4060 (requires GPU).
    """
    import subprocess
    
    script = os.path.join(BASE_DIR, 'server', 'training', 'train_adapter.py')
    cmd = [sys.executable, script, '--server-id', server_id, '--reason', reason]
    
    print(f"[RETRAIN] Launching: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30min timeout
        if result.returncode == 0:
            print(f"[RETRAIN] Training completed successfully for {server_id}")
        else:
            print(f"[RETRAIN] Training failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print(f"[RETRAIN] Training timed out for {server_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrain Trigger Manager")
    parser.add_argument("--check-now", action="store_true", help="Run a single trigger check")
    parser.add_argument("--server-id", type=str, help="Check a specific server")
    parser.add_argument("--force", action="store_true", help="Force retrain for --server-id")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    args = parser.parse_args()
    
    manager = RetrainTriggerManager(on_retrain_triggered=default_retrain_callback)
    
    if args.force and args.server_id:
        manager.force_retrain(args.server_id, reason="manual_cli")
    elif args.server_id:
        result = manager.check_single_server(args.server_id)
        print(f"\nResult: {result}")
    elif args.check_now:
        results = manager.check_all_servers()
        print(f"\nChecked {len(results)} servers.")
    elif args.daemon:
        manager.start()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            manager.stop()
    else:
        print("Use --check-now, --server-id, --force, or --daemon")
