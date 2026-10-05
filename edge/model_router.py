"""
G3.8: Edge Model Router
========================
Routes incoming telemetry to the appropriate specialist model based on 
data type. Manages loading and running multiple TFLite INT8 models.

Models:
  1. classifier_int8.tflite  — Existing 1D-CNN (thermal/power classification)
  2. iso_forest.pkl          — Existing Isolation Forest (PSU/voltage anomalies)
  3. tcn_storage_rul_int8.tflite — NEW: TCN for storage RUL prediction
  4. fan_bearing_cnn_int8.tflite — NEW: Fan bearing degradation detector

Routing Logic:
  - ALL telemetry → Classifier + Isolation Forest (general health)
  - Storage SMART data present → TCN Storage RUL
  - Fan PWM/RPM data present → Fan Bearing CNN
"""

import os
import time
import json
import numpy as np
import joblib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from edge.edge_logger import get_logger

# Try to import TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                tflite = None

logger = get_logger("model_router")

MODELS_DIR = os.path.join(BASE_DIR, 'edge', 'models')


@dataclass
class ModelResult:
    """Result from a single model inference."""
    model_name: str
    prediction: Any = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TFLiteModel:
    """Wrapper for a TFLite model with INT8 quantization support."""
    
    def __init__(self, name: str, model_path: str, scaler_path: Optional[str] = None):
        self.name = name
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.scaler = None
        self.loaded = False
        
        self._load(scaler_path)
    
    def _load(self, scaler_path: Optional[str] = None):
        """Load the TFLite model and optional scaler."""
        if tflite is None:
            logger.warning(f"[{self.name}] TFLite runtime not available")
            return
        
        if not os.path.exists(self.model_path):
            logger.warning(f"[{self.name}] Model not found: {self.model_path}")
            return
        
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.loaded = True
            
            size_kb = os.path.getsize(self.model_path) / 1024
            logger.info(f"[{self.name}] Loaded ({size_kb:.1f} KB)")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load: {e}")
        
        # Load scaler
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"[{self.name}] Loaded scaler: {scaler_path}")
            except Exception as e:
                logger.warning(f"[{self.name}] Scaler load failed: {e}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on the model. Handles INT8 quantization automatically."""
        if not self.loaded:
            return None
        
        # Apply scaler if available
        if self.scaler is not None:
            original_shape = input_data.shape
            flat = input_data.reshape(-1, original_shape[-1])
            flat = self.scaler.transform(flat)
            input_data = flat.reshape(original_shape)
        
        # Get expected input shape
        expected_shape = self.input_details[0]['shape']
        input_data = input_data.reshape(expected_shape).astype(np.float32)
        
        # Handle INT8 quantization
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.int8:
            quant_params = self.input_details[0].get('quantization_parameters', {})
            if quant_params:
                scale = quant_params.get('scales', [1.0])[0]
                zero_point = quant_params.get('zero_points', [0])[0]
            else:
                scale = self.input_details[0].get('quantization', (1.0, 0))[0]
                zero_point = self.input_details[0].get('quantization', (1.0, 0))[1]
            
            input_data = (input_data / scale + zero_point).clip(-128, 127).astype(np.int8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output


class ModelRouter:
    """
    Routes telemetry data to appropriate specialist models.
    
    Usage:
        router = ModelRouter()
        results = router.route(telemetry_payload, server_topology)
    """
    
    def __init__(self):
        self.models: Dict[str, TFLiteModel] = {}
        self.iso_forest = None
        self.iso_scaler = None
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available models."""
        logger.info("Loading edge models...")
        
        # 1. Existing 1D-CNN Classifier (thermal/power)
        classifier_path = os.path.join(MODELS_DIR, 'classifier_int8.tflite')
        classifier_scaler = os.path.join(MODELS_DIR, 'classifier_scaler.pkl')
        self.models['classifier'] = TFLiteModel(
            'Classifier', classifier_path, 
            classifier_scaler if os.path.exists(classifier_scaler) else None
        )
        
        # 2. TCN Storage RUL (NEW)
        tcn_path = os.path.join(MODELS_DIR, 'tcn_storage_rul_int8.tflite')
        tcn_scaler = os.path.join(MODELS_DIR, 'tcn_storage_scaler.joblib')
        self.models['tcn_storage'] = TFLiteModel(
            'TCN Storage RUL', tcn_path,
            tcn_scaler if os.path.exists(tcn_scaler) else None
        )
        
        # 3. Fan Bearing CNN (NEW)
        fan_path = os.path.join(MODELS_DIR, 'fan_bearing_cnn_int8.tflite')
        fan_scaler = os.path.join(MODELS_DIR, 'fan_bearing_scaler.joblib')
        self.models['fan_bearing'] = TFLiteModel(
            'Fan Bearing CNN', fan_path,
            fan_scaler if os.path.exists(fan_scaler) else None
        )
        
        # 4. Existing Isolation Forest
        iso_path = os.path.join(MODELS_DIR, 'iso_forest.pkl')
        if os.path.exists(iso_path):
            try:
                self.iso_forest = joblib.load(iso_path)
                logger.info("[Isolation Forest] Loaded")
            except Exception as e:
                logger.warning(f"[Isolation Forest] Failed: {e}")
        
        loaded = sum(1 for m in self.models.values() if m.loaded)
        loaded += 1 if self.iso_forest else 0
        total = len(self.models) + 1
        logger.info(f"Models loaded: {loaded}/{total}")
    
    def route(self, telemetry: Dict[str, Any], 
              topology: Optional[Dict] = None) -> List[ModelResult]:
        """
        Route telemetry to appropriate models and return results.
        
        Args:
            telemetry: v2 hierarchical telemetry payload OR v1 flat features
            topology: Optional hardware topology for context
        
        Returns:
            List of ModelResult from all applicable models
        """
        results = []
        
        # Detect payload version
        is_v2 = 'cpus' in telemetry or 'storage' in telemetry or 'fans' in telemetry
        
        # Route 1: General Classifier (runs on ALL telemetry)
        classifier_result = self._run_classifier(telemetry, is_v2)
        if classifier_result:
            results.append(classifier_result)
        
        # Route 2: Isolation Forest (PSU/voltage anomalies — runs on ALL)
        iso_result = self._run_isolation_forest(telemetry, is_v2)
        if iso_result:
            results.append(iso_result)
        
        # Route 3: TCN Storage RUL (only if storage data present)
        if self._has_storage_data(telemetry, is_v2):
            tcn_result = self._run_tcn_storage(telemetry, is_v2)
            if tcn_result:
                results.append(tcn_result)
        
        # Route 4: Fan Bearing CNN (only if fan data present)
        if self._has_fan_data(telemetry, is_v2):
            fan_result = self._run_fan_bearing(telemetry, is_v2)
            if fan_result:
                results.append(fan_result)
        
        return results
    
    def _has_storage_data(self, telemetry: Dict, is_v2: bool) -> bool:
        """Check if storage SMART data is present."""
        if is_v2:
            return 'storage' in telemetry and len(telemetry.get('storage', [])) > 0
        # v1: check for storage-related features
        storage_keys = ['disk_smart_reallocated_sectors', 'disk_io_read_latency',
                       'disk_io_write_latency', 'disk_smart_temperature']
        return any(k in telemetry for k in storage_keys)
    
    def _has_fan_data(self, telemetry: Dict, is_v2: bool) -> bool:
        """Check if fan PWM/RPM data is present."""
        if is_v2:
            return 'fans' in telemetry and len(telemetry.get('fans', [])) > 0
        # v1: check for fan-related features
        return 'fan_rpm_deviation' in telemetry or 'fan_duty_cycle' in telemetry
    
    def _run_classifier(self, telemetry: Dict, is_v2: bool) -> Optional[ModelResult]:
        """Run the general 1D-CNN classifier."""
        model = self.models.get('classifier')
        if not model or not model.loaded:
            return None
        
        try:
            start = time.time()
            
            # Extract 28 v1 features
            if is_v2:
                from server.training.feature_schema_v2 import v2_to_v1_feature_vector
                features = v2_to_v1_feature_vector(telemetry)
            else:
                from server.training.feature_schema import FEATURE_NAMES_V1
                features = np.array([telemetry.get(f, 0.0) for f in FEATURE_NAMES_V1])
            
            features = features.reshape(1, -1, 1).astype(np.float32)
            output = model.predict(features)
            
            latency = (time.time() - start) * 1000
            
            # Output: [health_score, failure_probability, ...]
            if output is not None:
                confidence = float(np.max(output))
                prediction = int(np.argmax(output))
                return ModelResult(
                    model_name='classifier',
                    prediction=prediction,
                    confidence=confidence,
                    latency_ms=latency,
                    metadata={'raw_output': output.tolist()}
                )
        except (ValueError, IndexError, RuntimeError) as e:
            return ModelResult(
                model_name='classifier',
                error=str(e),
                latency_ms=0.0
            )
        
        return None
    
    def _run_isolation_forest(self, telemetry: Dict, is_v2: bool) -> Optional[ModelResult]:
        """Run Isolation Forest for PSU/voltage anomaly detection."""
        if self.iso_forest is None:
            return None
        
        try:
            start = time.time()
            
            if is_v2:
                psu = telemetry.get('psu', {})
                features = np.array([[
                    psu.get('voltage_3v3_deviation', 0.0),
                    psu.get('voltage_12v_deviation', 0.0),
                    psu.get('voltage_stability', 1.0),
                    telemetry.get('fans', [{}])[0].get('rpm_deviation', 0.0) if telemetry.get('fans') else 0.0
                ]])
            else:
                features = np.array([[
                    telemetry.get('psu_voltage_3v3_dev', 0.0),
                    telemetry.get('psu_voltage_12v_dev', 0.0),
                    telemetry.get('psu_voltage_stability', 1.0),
                    telemetry.get('fan_rpm_deviation', 0.0)
                ]])
            
            pred = self.iso_forest.predict(features)
            latency = (time.time() - start) * 1000
            
            return ModelResult(
                model_name='isolation_forest',
                prediction='anomaly' if pred[0] == -1 else 'normal',
                confidence=0.95 if pred[0] == -1 else 0.5,
                latency_ms=latency
            )
        except (ValueError, IndexError, RuntimeError) as e:
            return ModelResult(model_name='isolation_forest', error=str(e))
    
    def _run_tcn_storage(self, telemetry: Dict, is_v2: bool) -> Optional[ModelResult]:
        """Run TCN to predict storage RUL in days."""
        model = self.models.get('tcn_storage')
        if not model or not model.loaded:
            return None
        
        try:
            start = time.time()
            
            # Extract SMART features
            # TCN expects (1, window_size, 8) — but for real-time, we use a single snapshot
            # padded to match the training window
            if is_v2:
                storage_devices = telemetry.get('storage', [])
                if not storage_devices:
                    return None
                    
                device = storage_devices[0]  # Primary storage device
                smart_features = np.array([
                    device.get('smart_5_raw', device.get('reallocated_sectors', 0)),
                    device.get('smart_9_raw', device.get('power_on_hours', 0)),
                    device.get('smart_187_raw', device.get('uncorrectable_errors', 0)),
                    device.get('smart_188_raw', device.get('command_timeout', 0)),
                    device.get('smart_194_raw', device.get('temperature', 35)),
                    device.get('smart_197_raw', device.get('pending_sectors', 0)),
                    device.get('smart_198_raw', device.get('offline_uncorrectable', 0)),
                    device.get('smart_241_raw', device.get('total_lba_written', 0)),
                ], dtype=np.float32)
            else:
                smart_features = np.array([
                    telemetry.get('disk_smart_reallocated_sectors', 0),
                    telemetry.get('disk_power_on_hours', 0),
                    telemetry.get('disk_smart_uncorrectable', 0),
                    telemetry.get('disk_smart_command_timeout', 0),
                    telemetry.get('disk_smart_temperature', 35),
                    telemetry.get('disk_smart_pending_sectors', 0),
                    telemetry.get('disk_smart_offline_uncorrectable', 0),
                    telemetry.get('disk_smart_total_lba_written', 0),
                ], dtype=np.float32)
            
            # Pad to match window size (30 steps) by repeating the snapshot
            window_size = 30
            window = np.tile(smart_features, (window_size, 1))  # (30, 8)
            window = window.reshape(1, window_size, -1)
            
            output = model.predict(window)
            latency = (time.time() - start) * 1000
            
            if output is not None:
                rul_days = float(np.clip(output[0][0], 0, 90))
                confidence = min(1.0, rul_days / 90.0)  # Lower RUL = higher urgency
                
                return ModelResult(
                    model_name='tcn_storage',
                    prediction=rul_days,
                    confidence=1.0 - confidence,  # Higher confidence when RUL is low (urgent)
                    latency_ms=latency,
                    metadata={
                        'rul_days': rul_days,
                        'urgency': 'critical' if rul_days < 7 else 'warning' if rul_days < 30 else 'normal'
                    }
                )
        except (ValueError, IndexError, RuntimeError) as e:
            return ModelResult(model_name='tcn_storage', error=str(e))
        
        return None
    
    def _run_fan_bearing(self, telemetry: Dict, is_v2: bool) -> Optional[ModelResult]:
        """Run Fan Bearing CNN for bearing degradation detection."""
        model = self.models.get('fan_bearing')
        if not model or not model.loaded:
            return None
        
        try:
            start = time.time()
            
            # Extract fan features
            if is_v2:
                fans = telemetry.get('fans', [])
                if not fans:
                    return None
                
                fan = fans[0]  # Primary fan
                pwm_pct = fan.get('duty_cycle', fan.get('pwm_pct', 50.0))
                actual_rpm = fan.get('rpm', 1500)
                max_rpm = fan.get('max_rpm', 3000)
                expected_rpm = (pwm_pct / 100.0) * max_rpm
                ratio = actual_rpm / max(expected_rpm, 1.0)
            else:
                pwm_pct = telemetry.get('fan_duty_cycle', 50.0)
                actual_rpm = telemetry.get('fan_rpm', 1500)
                max_rpm = 3000
                expected_rpm = (pwm_pct / 100.0) * max_rpm
                ratio = actual_rpm / max(expected_rpm, 1.0)
            
            # Create a 60-tick window by repeating (real-time single snapshot)
            features = np.array([pwm_pct, actual_rpm, ratio], dtype=np.float32)
            window = np.tile(features, (60, 1))  # (60, 3)
            window = window.reshape(1, 60, 3)
            
            output = model.predict(window)
            latency = (time.time() - start) * 1000
            
            if output is not None:
                degradation_prob = float(output[0][0])
                
                return ModelResult(
                    model_name='fan_bearing',
                    prediction='degrading' if degradation_prob > 0.5 else 'healthy',
                    confidence=degradation_prob if degradation_prob > 0.5 else 1.0 - degradation_prob,
                    latency_ms=latency,
                    metadata={
                        'degradation_probability': degradation_prob,
                        'pwm_pct': pwm_pct,
                        'actual_rpm': actual_rpm,
                        'rpm_ratio': ratio
                    }
                )
        except (ValueError, IndexError, RuntimeError) as e:
            return ModelResult(model_name='fan_bearing', error=str(e))
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Return model loading status for dashboard."""
        status = {
            'models_loaded': sum(1 for m in self.models.values() if m.loaded) + (1 if self.iso_forest else 0),
            'models_total': len(self.models) + 1,
            'models': {}
        }
        for name, model in self.models.items():
            status['models'][name] = {
                'loaded': model.loaded,
                'path': model.model_path,
                'size_kb': os.path.getsize(model.model_path) / 1024 if model.loaded else 0
            }
        status['models']['isolation_forest'] = {
            'loaded': self.iso_forest is not None,
            'type': 'scikit-learn'
        }
        return status
