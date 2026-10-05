"""
RPi4 Benchmark Worker — Runs locally on Raspberry Pi 4 to measure exact
physical sensor read latency and LiteRT INT8 inference latency.
"""
import os
import sys
import time
import json
import numpy as np

# Ensure path
sys.path.insert(0, "/home/pi/predictive-maintenance")

results = {}

# 1. Physical Hardware Sensor Polling
try:
    from edge.sensor_reader import SensorReader
    sr = SensorReader()
    # Warmup
    for _ in range(3):
        sr.read_all()
    times = []
    sample_data = None
    for _ in range(15):
        t0 = time.perf_counter()
        sample_data = sr.read_all()
        times.append((time.perf_counter() - t0) * 1000)
    
    results["sensors"] = {
        "avg_ms": round(float(np.mean(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
        "sample": sample_data
    }
except Exception as e:
    results["sensors"] = {"error": str(e)}

# 2. LiteRT INT8 Model Inference Latency
try:
    from ai_edge_litert.interpreter import Interpreter
    models_dir = "/home/pi/predictive-maintenance/edge/models"
    model_files = {
        "classifier": "classifier_int8.tflite",
        "stgcn": "stgcn_int8.tflite",
        "fan_cnn": "fan_bearing_cnn_int8.tflite",
        "tcn_storage": "tcn_storage_rul_int8.tflite"
    }
    
    bench_models = {}
    for name, filename in model_files.items():
        m_path = os.path.join(models_dir, filename)
        if not os.path.exists(m_path):
            bench_models[name] = {"status": "file_not_found"}
            continue
        
        interp = Interpreter(model_path=m_path)
        interp.allocate_tensors()
        inp_info = interp.get_input_details()[0]
        dummy_input = np.zeros(inp_info["shape"], dtype=inp_info["dtype"])
        
        # Warmup
        for _ in range(5):
            interp.set_tensor(inp_info["index"], dummy_input)
            interp.invoke()
            
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            interp.set_tensor(inp_info["index"], dummy_input)
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)
            
        bench_models[name] = {
            "avg_ms": round(float(np.mean(times)), 3),
            "min_ms": round(float(np.min(times)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "size_kb": round(os.path.getsize(m_path) / 1024, 1)
        }
        
    results["edge_models"] = bench_models
except Exception as e:
    results["edge_models"] = {"error": str(e)}

print(json.dumps(results))
