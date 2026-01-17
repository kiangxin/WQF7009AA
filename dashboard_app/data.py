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
        xgb_model = joblib.load("xgb_model.pkl")
        scaler = joblib.load("scaler.pkl")

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

