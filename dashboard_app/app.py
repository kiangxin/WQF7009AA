from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

from dashboard_app.config import (
    CLASS_DESCRIPTIONS,
    CLASS_LABELS_SHORT,
    FEATURE_COLS,
    FEATURE_DESCRIPTIONS,
    ROLES,
)
from dashboard_app.data import get_shap_explainer, load_model_and_data
from dashboard_app.styles import apply_light_theme_css
from dashboard_app.views.public_health import public_health_officer_view
from dashboard_app.views.public_user import public_user_view
from dashboard_app.views.scientist import scientist_view
from dashboard_app.views.regulator import regulator_view


def main() -> None:
    st.set_page_config(
        page_title="Air Quality Health Impact XAI Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_light_theme_css()

    st.markdown(
        '<p class="main-header">🏥 Air Quality Health Impact XAI Dashboard</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model and data..."):
        xgb_model, scaler, pipeline, X, y, _df = load_model_and_data()
        explainer = get_shap_explainer(xgb_model)

        # SHAP sampling for performance, deterministic
        sample_size = min(1000, len(X))
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(X), sample_size, replace=False)
        X_shap = X.iloc[sample_indices].copy()
        X_scaled_shap = scaler.transform(X_shap)

        X_scaled = scaler.transform(X)

    # Sidebar - Role selection
    st.sidebar.title("👤 Select Your Role")
    selected_role = st.sidebar.radio(
        "Choose your role:",
        list(ROLES.keys()),
        format_func=lambda x: f"{ROLES[x]['icon']} {x}",
        key="role_select",
    )
    st.sidebar.markdown(f"**{ROLES[selected_role]['description']}**")

    # Sidebar - Dataset info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.markdown(f"Total Instances: {len(X)}")
    st.sidebar.markdown(f"Features: {len(FEATURE_COLS)}")
    st.sidebar.markdown("**Class Distribution:**")
    for cls in sorted(np.unique(y)):
        count = int((y == cls).sum())
        pct = (count / len(y)) * 100
        st.sidebar.markdown(f"- Class {cls}: {count} ({pct:.1f}%)")

    with st.sidebar.expander("📖 Feature Descriptions"):
        for feat in FEATURE_COLS:
            st.markdown(f"**{feat}**: {FEATURE_DESCRIPTIONS.get(feat, '')}")

    with st.sidebar.expander("🏥 Health Impact Classes"):
        for cls in sorted(CLASS_DESCRIPTIONS.keys()):
            st.markdown(f"**Class {cls}**: {CLASS_LABELS_SHORT[cls]}")

    st.markdown("---")

    if selected_role == "Scientist":
        scientist_view(explainer, X_shap, X_scaled_shap, FEATURE_COLS, y)
    elif selected_role == "Regulator":
        regulator_view(explainer, X_shap, X_scaled_shap, X, y, FEATURE_COLS, pipeline)
    elif selected_role == "Public Health Officer":
        public_health_officer_view(explainer, X, X_scaled, FEATURE_COLS, y)
    elif selected_role == "Public User":
        public_user_view(explainer, X, X_scaled, FEATURE_COLS, xgb_model)

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Air Quality Health Impact XAI Dashboard | Built with Streamlit & SHAP</p>",
        unsafe_allow_html=True,
    )

