from __future__ import annotations

import joblib
import pandas as pd
import shap
import streamlit as st
from sklearn.pipeline import Pipeline

from dashboard_app.config import FEATURE_COLS


@st.cache_resource
def load_model_and_data():
    """
    Load model/scaler and dataset.
    """
    try:
        model_obj = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # If the saved object is a (imblearn/sklearn) Pipeline, TreeExplainer can't use it directly.
        # SHAP TreeExplainer supports tree models, not preprocessing/sampling pipelines.
        if hasattr(model_obj, "named_steps"):
            steps = getattr(model_obj, "named_steps", {})
            # Try common names first
            xgb_model = steps.get("classifier") or steps.get("xgb")
            if xgb_model is None and len(steps) > 0:
                # Fall back to last step
                xgb_model = list(steps.values())[-1]

            # Prefer scaler from the pipeline if present (keeps things consistent)
            scaler_in_pipe = steps.get("scaler")
            if scaler_in_pipe is not None:
                scaler = scaler_in_pipe
        else:
            xgb_model = model_obj

        df = pd.read_csv("air_quality_health_impact_data.csv")
        X = df[FEATURE_COLS]
        y = df["HealthImpactClass"].astype(int)

        pipeline = Pipeline([("scaler", scaler), ("classifier", xgb_model)])
        return xgb_model, scaler, pipeline, X, y, df
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        st.stop()


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

