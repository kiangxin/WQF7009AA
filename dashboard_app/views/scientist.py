from __future__ import annotations

import numpy as np
import streamlit as st

from dashboard_app.shap_utils import global_mean_abs_by_class, global_shap_total_bar_values
from dashboard_app.views.plots import (
    plot_global_shap_all_classes_stacked_bar,
    plot_global_shap_for_class,
    plot_global_shap_total_bar,
)


def scientist_view(explainer, X_shap, X_scaled_shap, feature_cols, y):
    st.markdown('<p class="role-header">🔬 Scientist View</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Analyze global feature importance across health impact classes.</div>',
        unsafe_allow_html=True,
    )

    class_labels = sorted(np.unique(y))
    options = ["All Classes", "Total"] + [str(c) for c in class_labels]

    selected = st.selectbox(
        "Select class:",
        options,
        format_func=lambda v: v if v in {"All Classes", "Total"} else f"Class {v}",
        key="scientist_class_select",
    )

    if selected == "All Classes":
        st.subheader("Global SHAP Summary (All Classes at once)")
        mean_abs = global_mean_abs_by_class(explainer, X_scaled_shap, feature_cols)
        plot_global_shap_all_classes_stacked_bar(feature_cols, mean_abs, max_display=12)
        return

    if selected == "Total":
        st.subheader("Global SHAP (Total across all classes)")
        total_mean_abs = global_shap_total_bar_values(explainer, X_scaled_shap, feature_cols)
        plot_global_shap_total_bar(feature_cols, total_mean_abs)

        top3 = np.argsort(total_mean_abs)[::-1][:3]
        st.markdown("**Top 3 overall contributing factors:**")
        for i, idx in enumerate(top3, 1):
            st.markdown(f"{i}. **{feature_cols[idx]}** (mean |SHAP|: {total_mean_abs[idx]:.4f})")
        return

    class_idx = int(selected)
    st.subheader(f"Global SHAP - Class {class_idx}")
    plot_global_shap_for_class(explainer, X_shap, X_scaled_shap, feature_cols, class_idx)

