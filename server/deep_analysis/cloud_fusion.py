"""
G4.3: Cloud Decision Fusion Engine
Combines outputs from cloud-side deep models (MAD-GAN, STGCN, LSTM-AE)
with edge predictions for final anomaly verdict.

This runs on the LOQ cloud server as part of inference_server.py.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class CloudDecisionFusion:
    """
    Fuses predictions from multiple cloud models + edge offloaded predictions.
    
    Cloud Models:
      - MAD-GAN: Multivariate anomaly detection (discrimination + reconstruction)
      - STGCN: Spatio-temporal cross-correlation anomaly detection
      - LSTM-AE: Autoencoder reconstruction error anomaly detection
    
    Edge Input:
      - Offloaded predictions (uncertain cases with confidence < threshold)
      
    Output:
      - Final anomaly verdict with confidence
      - Risk assessment (0-100 scale)
      - Contributing model breakdown
    """
    
    # Model weights (importance in final decision)
    # F-026 MITIGATION (2026-09-30): MAD-GAN F1=5.1%, STGCN F1=4.1% — both near-random.
    # Reduced to token weight (0.05) to prevent them from degrading fusion reliability.
    # Edge classifier (99.3% accuracy) now dominates decisions.
    # TODO: Restore original weights (0.30/0.25) after retraining MAD-GAN/STGCN on real data.
    MODEL_WEIGHTS = {
        "madgan": 0.05,      # Near-random (F1=5.1%) — token weight until retrained
        "stgcn": 0.05,       # Near-random (F1=4.1%) — token weight until retrained
        "lstm_ae": 0.15,     # Temporal reconstruction (functional but noisy on real data)
        "edge_classifier": 0.75,  # Edge 1D-CNN: 99.3% accuracy — trusted primary model
    }
    
    # Severity mapping
    SEVERITY_LEVELS = {
        (0, 20): "CRITICAL",
        (20, 40): "WARNING",
        (40, 60): "CAUTION",
        (60, 80): "MONITORING",
        (80, 100): "HEALTHY",
    }
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), '..', 'models')
        self.thresholds = {}
        self.models_loaded = set()
        self._load_thresholds()
    
    def _load_thresholds(self):
        """Load anomaly thresholds from model metadata."""
        meta_files = {
            "madgan": "madgan_metadata.json",
            "stgcn": "stgcn_metadata.json",
            "lstm_ae": "lstm_ae_metadata.json",  # Legacy
        }
        
        for name, fname in meta_files.items():
            path = os.path.join(self.model_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    meta = json.load(f)
                self.thresholds[name] = meta.get("anomaly_threshold", 0.5)
                self.models_loaded.add(name)
                logger.info(f"Loaded {name} threshold: {self.thresholds[name]:.4f}")
        
        logger.info(f"Cloud fusion: {len(self.models_loaded)} models loaded")
    
    def fuse_predictions(
        self,
        server_id: str,
        edge_prediction: Optional[Dict] = None,
        madgan_score: Optional[float] = None,
        stgcn_score: Optional[float] = None,
        lstm_ae_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fuse all model predictions into a single verdict.
        
        Args:
            server_id: Server identifier
            edge_prediction: Edge model output (class, confidence, ttf)
            madgan_score: MAD-GAN anomaly score (higher = more anomalous)
            stgcn_score: STGCN anomaly probability (0-1)
            lstm_ae_score: LSTM-AE reconstruction error
            
        Returns:
            Fused decision with health score, severity, and breakdown
        """
        model_votes = {}
        total_weight = 0
        weighted_score = 0
        
        # Edge classifier
        if edge_prediction:
            conf = edge_prediction.get("confidence", 0.5)
            is_anomaly = edge_prediction.get("class", 0) != 0
            edge_score = (1 - conf) if is_anomaly else conf
            model_votes["edge_classifier"] = {
                "score": float(edge_score),
                "is_anomaly": is_anomaly,
                "raw": edge_prediction,
            }
            weighted_score += edge_score * self.MODEL_WEIGHTS["edge_classifier"]
            total_weight += self.MODEL_WEIGHTS["edge_classifier"]
        
        # MAD-GAN
        if madgan_score is not None:
            threshold = self.thresholds.get("madgan", 0.5)
            is_anomaly = madgan_score > threshold
            # Normalize to 0-1 health score (0 = bad, 1 = good)
            health = max(0, min(1, 1 - (madgan_score / (2 * threshold + 1e-8))))
            model_votes["madgan"] = {
                "score": float(health),
                "is_anomaly": is_anomaly,
                "raw_score": float(madgan_score),
                "threshold": float(threshold),
            }
            weighted_score += health * self.MODEL_WEIGHTS["madgan"]
            total_weight += self.MODEL_WEIGHTS["madgan"]
        
        # STGCN
        if stgcn_score is not None:
            threshold = self.thresholds.get("stgcn", 0.5)
            is_anomaly = stgcn_score > threshold
            health = 1 - stgcn_score
            model_votes["stgcn"] = {
                "score": float(health),
                "is_anomaly": is_anomaly,
                "raw_score": float(stgcn_score),
                "threshold": float(threshold),
            }
            weighted_score += health * self.MODEL_WEIGHTS["stgcn"]
            total_weight += self.MODEL_WEIGHTS["stgcn"]
        
        # LSTM-AE
        if lstm_ae_score is not None:
            threshold = self.thresholds.get("lstm_ae", 0.5)
            is_anomaly = lstm_ae_score > threshold
            health = max(0, min(1, 1 - (lstm_ae_score / (2 * threshold + 1e-8))))
            model_votes["lstm_ae"] = {
                "score": float(health),
                "is_anomaly": is_anomaly,
                "raw_score": float(lstm_ae_score),
                "threshold": float(threshold),
            }
            weighted_score += health * self.MODEL_WEIGHTS["lstm_ae"]
            total_weight += self.MODEL_WEIGHTS["lstm_ae"]
        
        # Compute final health score (0-100)
        if total_weight > 0:
            health_score = (weighted_score / total_weight) * 100
        else:
            health_score = 50.0  # Unknown
        
        # Determine severity
        severity = "UNKNOWN"
        for (lo, hi), label in self.SEVERITY_LEVELS.items():
            if lo <= health_score < hi:
                severity = label
                break
        if health_score >= 80:
            severity = "HEALTHY"
        
        # Count anomaly votes
        n_anomaly = sum(1 for v in model_votes.values() if v.get("is_anomaly", False))
        n_models = len(model_votes)
        consensus = n_anomaly / n_models if n_models > 0 else 0
        
        # Final verdict
        is_anomalous = health_score < 60 or consensus > 0.5
        
        result = {
            "server_id": server_id,
            "health_score": round(health_score, 1),
            "severity": severity,
            "is_anomalous": is_anomalous,
            "consensus": f"{n_anomaly}/{n_models} models flagged anomaly",
            "model_breakdown": model_votes,
            "models_used": list(model_votes.keys()),
            "recommendation": self._get_recommendation(severity, health_score, model_votes),
        }
        
        logger.info(f"[{server_id}] Cloud fusion: {severity} ({health_score:.1f}) | {n_anomaly}/{n_models} anomaly")
        
        return result
    
    def _get_recommendation(self, severity: str, score: float, votes: Dict) -> str:
        """Generate human-readable recommendation."""
        if severity == "CRITICAL":
            flagged = [k for k, v in votes.items() if v.get("is_anomaly")]
            return f"IMMEDIATE ACTION: Multiple models ({', '.join(flagged)}) detected critical anomaly. Inspect hardware immediately."
        elif severity == "WARNING":
            return "Schedule maintenance within 24-48 hours. Monitor closely for degradation."
        elif severity == "CAUTION":
            return "Elevated risk detected. Increase monitoring frequency. Plan maintenance window."
        elif severity == "MONITORING":
            return "Minor deviation detected. Continue normal monitoring."
        else:
            return "All systems nominal. No action required."


# Cloud inference integration
def create_cloud_fusion(model_dir: str = None) -> CloudDecisionFusion:
    """Factory function for creating the cloud fusion engine."""
    return CloudDecisionFusion(model_dir)
