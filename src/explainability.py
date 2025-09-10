# src/explainability.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

def save_top_features(feature_names, importances, out_json: str, top_k: int = 20):
    idx = np.argsort(importances)[::-1][:top_k]
    top = [{"feature": feature_names[i], "importance": float(importances[i])} for i in idx]
    Path(out_json).write_text(json.dumps(top, indent=2))

# SHAP/LIME hooks (optional heavy deps)
def try_shap_tree(model, X_sample: pd.DataFrame, out_json: str, max_samples: int = 1000):
    try:
        import shap  # lazy import
        explainer = shap.TreeExplainer(model)
        sample = X_sample.sample(min(len(X_sample), max_samples), random_state=42)
        vals = explainer.shap_values(sample)
        np.save(out_json.replace('.json', '.npy'), vals)  # raw values for later plotting
        return True
    except Exception as e:
        Path(out_json).write_text(json.dumps({"error": str(e)}))
        return False
