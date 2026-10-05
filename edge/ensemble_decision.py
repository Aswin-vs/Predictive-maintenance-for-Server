"""
G3.9: Ensemble Decision Engine
================================
Combines outputs from all specialist models into a single health verdict.
Implements weighted voting with confidence levels and cloud offloading
for uncertain predictions.

Decision Flow:
  1. Collect results from ModelRouter (1-4 model outputs)
  2. Weight each result by model importance + confidence
  3. Compute composite health score (0-100)
  4. Determine action: normal / warning / critical / offload-to-cloud
  5. Log decision to local store
"""

import time
import json
import os
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from edge.edge_logger import get_logger
from edge.model_router import ModelResult

logger = get_logger("ensemble_decision")

# Cloud offload endpoint
_cloud_fallback = os.environ.get("CLOUD_INFERENCE_URL", "http://localhost:8000").rstrip("/") + "/api/offload"
CLOUD_OFFLOAD_URL = os.environ.get("CLOUD_OFFLOAD_URL", _cloud_fallback)

# Confidence threshold below which we offload to cloud
OFFLOAD_CONFIDENCE_THRESHOLD = float(os.environ.get("OFFLOAD_THRESHOLD", "0.85"))


@dataclass
class EnsembleDecision:
    """Final decision from the ensemble of all edge models."""
    health_score: float = 100.0        # 0 (critical) to 100 (healthy)
    severity: str = "normal"           # normal / warning / critical
    action: str = "monitor"            # monitor / alert / offload / emergency
    offloaded: bool = False
    cloud_response: Optional[Dict] = None
    model_results: List[Dict] = field(default_factory=list)
    storage_rul_days: Optional[float] = None
    fan_status: Optional[str] = None
    anomaly_detected: bool = False
    inference_time_ms: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/dashboard."""
        return {
            'health_score': round(self.health_score, 1),
            'severity': self.severity,
            'action': self.action,
            'offloaded': self.offloaded,
            'cloud_response': self.cloud_response,
            'model_results': self.model_results,
            'storage_rul_days': round(self.storage_rul_days, 1) if self.storage_rul_days else None,
            'fan_status': self.fan_status,
            'anomaly_detected': self.anomaly_detected,
            'inference_time_ms': round(self.inference_time_ms, 2),
            'timestamp': self.timestamp,
        }


# Model importance weights for health score calculation
MODEL_WEIGHTS = {
    'classifier': 0.35,        # General health assessment (highest weight)
    'isolation_forest': 0.20,  # PSU/voltage anomaly
    'tcn_storage': 0.25,       # Storage is critical infrastructure
    'fan_bearing': 0.20,       # Cooling is important but less critical
}


class EnsembleDecisionEngine:
    """
    Combines multiple model outputs into a single health verdict.
    
    Usage:
        from edge.model_router import ModelRouter
        router = ModelRouter()
        ensemble = EnsembleDecisionEngine()
        
        results = router.route(telemetry)
        decision = ensemble.decide(results, telemetry)
    """
    
    def __init__(self, offload_threshold: float = OFFLOAD_CONFIDENCE_THRESHOLD):
        self.offload_threshold = offload_threshold
        self.decision_count = 0
        self.offload_count = 0
        logger.info(f"Ensemble engine initialized (offload threshold: {offload_threshold})")
    
    def decide(self, model_results: List[ModelResult], 
               telemetry: Optional[Dict] = None,
               server_id: str = "unknown") -> EnsembleDecision:
        """
        Combine model results into a final health decision.
        
        Args:
            model_results: List of ModelResult from ModelRouter.route()
            telemetry: Original telemetry for cloud offloading
            server_id: Server identifier for logging
        
        Returns:
            EnsembleDecision with health score, severity, and action
        """
        start = time.time()
        
        decision = EnsembleDecision(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        
        if not model_results:
            decision.severity = "unknown"
            decision.action = "offload"
            return decision
        
        # Process each model result
        health_penalties = []
        total_weight = 0.0
        avg_confidence = 0.0
        
        for result in model_results:
            result_dict = {
                'model': result.model_name,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'latency_ms': result.latency_ms,
                'error': result.error,
            }
            if result.metadata:
                result_dict['metadata'] = result.metadata
            decision.model_results.append(result_dict)
            
            if result.error:
                continue
            
            weight = MODEL_WEIGHTS.get(result.model_name, 0.1)
            total_weight += weight
            avg_confidence += result.confidence * weight
            
            # Calculate health penalty from each model
            penalty = self._compute_penalty(result)
            health_penalties.append((weight, penalty))
            
            # Extract specific model outputs
            if result.model_name == 'tcn_storage' and result.metadata:
                decision.storage_rul_days = result.metadata.get('rul_days')
            
            if result.model_name == 'fan_bearing':
                decision.fan_status = str(result.prediction)
            
            if result.model_name == 'isolation_forest':
                decision.anomaly_detected = (result.prediction == 'anomaly')
        
        # Compute weighted health score
        if health_penalties:
            total_penalty = sum(w * p for w, p in health_penalties) / sum(w for w, _ in health_penalties)
            decision.health_score = max(0, min(100, 100 - total_penalty))
        
        # Compute average confidence
        if total_weight > 0:
            avg_confidence /= total_weight
        
        # Determine severity
        decision.severity = self._compute_severity(decision)
        
        # Determine action
        decision.action = self._compute_action(decision, avg_confidence)
        
        # Offload to cloud if needed
        if decision.action == "offload" and telemetry:
            decision.offloaded = True
            decision.cloud_response = self._offload_to_cloud(
                telemetry, decision, server_id
            )
        
        decision.inference_time_ms = (time.time() - start) * 1000
        
        self.decision_count += 1
        if decision.offloaded:
            self.offload_count += 1
        
        logger.info(
            f"[{server_id}] Health: {decision.health_score:.0f}/100 | "
            f"Severity: {decision.severity} | Action: {decision.action} | "
            f"Models: {len(model_results)} | Time: {decision.inference_time_ms:.1f}ms"
        )
        
        return decision
    
    def _compute_penalty(self, result: ModelResult) -> float:
        """Compute health penalty (0-100) from a model result."""
        
        if result.model_name == 'classifier':
            # Higher prediction class = more severe failure
            if isinstance(result.prediction, (int, float)):
                # Classes: 0=healthy, 1=degraded, 2=critical
                return result.prediction * 40 * result.confidence
            return 0.0
        
        elif result.model_name == 'isolation_forest':
            # Anomaly = significant penalty
            return 60.0 * result.confidence if result.prediction == 'anomaly' else 0.0
        
        elif result.model_name == 'tcn_storage':
            # Low RUL = high penalty
            rul_days = result.metadata.get('rul_days', 90) if result.metadata else 90
            if rul_days < 7:
                return 80.0  # Critical
            elif rul_days < 30:
                return 40.0  # Warning
            elif rul_days < 60:
                return 15.0  # Mild concern
            return 0.0
        
        elif result.model_name == 'fan_bearing':
            # Degrading bearing = moderate penalty
            return 50.0 * result.confidence if result.prediction == 'degrading' else 0.0
        
        return 0.0
    
    def _compute_severity(self, decision: EnsembleDecision) -> str:
        """Determine severity level from health score."""
        if decision.health_score >= 80:
            return "normal"
        elif decision.health_score >= 50:
            return "warning"
        else:
            return "critical"
    
    def _compute_action(self, decision: EnsembleDecision, avg_confidence: float) -> str:
        """Determine recommended action."""
        # Low confidence = offload to cloud for deeper analysis
        if avg_confidence < self.offload_threshold:
            return "offload"
        
        # Emergency actions
        if decision.severity == "critical":
            return "emergency"
        
        # Storage-specific urgency
        if decision.storage_rul_days is not None and decision.storage_rul_days < 7:
            return "emergency"
        
        # Warning actions
        if decision.severity == "warning":
            return "alert"
        
        # Fan bearing degradation
        if decision.fan_status == 'degrading':
            return "alert"
        
        # PSU anomaly
        if decision.anomaly_detected:
            return "alert"
        
        return "monitor"
    
    def _offload_to_cloud(self, telemetry: Dict, decision: EnsembleDecision,
                          server_id: str) -> Optional[Dict]:
        """Send uncertain predictions to cloud for deeper analysis."""
        try:
            payload = {
                "server_id": server_id,
                "telemetry": telemetry,
                "edge_health_score": decision.health_score,
                "edge_severity": decision.severity,
                "edge_model_results": decision.model_results,
                "reason": "low_confidence_offload",
            }
            
            api_key = os.environ.get("API_KEY", "")
            if not api_key:
                _keys_path = os.path.join(BASE_DIR, 'configs', 'api_keys.json')
                if os.path.exists(_keys_path):
                    try:
                        with open(_keys_path) as _f:
                            api_key = json.load(_f).get('keys', [''])[0]
                    except Exception:
                        pass
            headers = {"X-API-Key": api_key} if api_key else {}

            response = requests.post(
                CLOUD_OFFLOAD_URL,
                json=payload,
                headers=headers,
                timeout=5.0
            )
            
            if response.status_code == 200:
                cloud_data = response.json()
                logger.info(f"[{server_id}] Cloud offload successful")
                return cloud_data
            else:
                logger.warning(f"[{server_id}] Cloud returned {response.status_code}")
                return {"status": "error", "code": response.status_code}
                
        except requests.exceptions.ConnectionError:
            logger.info(f"[{server_id}] Cloud unreachable — edge decision stands")
            return {"status": "cloud_unreachable"}
        except Exception as e:
            logger.warning(f"[{server_id}] Cloud offload failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics for dashboard."""
        return {
            "total_decisions": self.decision_count,
            "offloads": self.offload_count,
            "offload_rate": f"{self.offload_count / max(1, self.decision_count) * 100:.1f}%",
            "offload_threshold": self.offload_threshold,
        }
