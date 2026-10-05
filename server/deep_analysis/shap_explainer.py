"""
SHAP Explainability Module.

Generates SHAP (SHapley Additive exPlanations) values to explain why the 
deep learning models made specific predictions. This is used by the RAG 
Chatbot to explain anomalies to the user.
"""

import os
import numpy as np
import pandas as pd
import joblib

try:
    import shap
except ImportError:
    shap = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'server', 'models')

class SHAPExplainer:
    def __init__(self):
        self.xgb_model = None
        self.explainer = None
        self._load_model()
        
    def _load_model(self):
        xgb_path = os.path.join(MODEL_DIR, 'xgboost_hdd.pkl')
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            if shap:
                # Use TreeExplainer for XGBoost
                self.explainer = shap.TreeExplainer(self.xgb_model)
        else:
            print(f"Warning: XGBoost model not found at {xgb_path}")

    def explain_hdd_prediction(self, features: pd.DataFrame) -> dict:
        """
        Explain a single HDD TTF prediction.
        Returns a dictionary mapping feature names to their SHAP impact values.
        """
        if not self.explainer or not shap:
            return {"error": "SHAP or model not initialized."}
            
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # For a single sample, shap_values is a 1D array
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
                
            # Create a dictionary of feature -> impact
            feature_names = features.columns.tolist()
            impact = {name: float(val) for name, val in zip(feature_names, shap_values)}
            
            # Sort by absolute impact (highest first)
            impact_sorted = dict(sorted(impact.items(), key=lambda item: abs(item[1]), reverse=True))
            
            return {
                "base_value": float(self.explainer.expected_value),
                "feature_impact": impact_sorted
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    if not shap:
        print("Please install shap: pip install shap")
    else:
        print("SHAP explainer initialized.")
