from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

from dashboard_app.config import CLASS_DESCRIPTIONS, FEATURE_HIGH_EXPLANATION
from dashboard_app.shap_utils import local_shap_1d_and_base_value


def public_user_view(explainer, X, X_scaled, feature_cols, model):
    st.markdown('<p class="role-header">👤 Public User View</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Understand your personal health risk based on current air quality conditions.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<style>
.explanation-card {
    background-color: rgba(248, 249, 250, 0.85);
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
}
.explanation-card h4 {
    margin: 0 0 10px 0;
}
.explanation-card p {
    margin: 0 0 10px 0;
}
.explanation-card p:last-child {
    margin-bottom: 0;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    def render_explanation_card(markdown_text: str, title: str = "Simple Explanation") -> None:
        st.markdown(
            f"""
<div class="explanation-card">
<h4>{title}</h4>
{markdown_text}
</div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Check Your Health Risk")
    user_idx_raw = st.text_input(
        "Enter your location/instance index (integer):",
        value="0",
        key="public_user_idx_text",
        help=f"Valid range: 0 to {len(X) - 1}",
    )

    if st.button("Check My Risk", key="public_user_btn"):
        user_idx_raw = user_idx_raw.strip()
        if not user_idx_raw.isdigit():
            st.error("Invalid input. Please enter a whole number (e.g., 0, 12, 350).")
            return

        user_idx = int(user_idx_raw)
        if user_idx < 0 or user_idx >= len(X):
            st.error(f"Index out of range. Please enter a value between 0 and {len(X) - 1}.")
            return

        with st.spinner("Analyzing..."):
            instance = X_scaled[user_idx : user_idx + 1]
            instance_original = X.iloc[user_idx : user_idx + 1]

            pred_class = int(model.predict(instance)[0])

            # Alert box based on class (0=worst, 4=best)
            if pred_class >= 3:
                st.markdown(
                    f'<div class="info-box"><strong>✅</strong><br>{CLASS_DESCRIPTIONS[pred_class]}</div>',
                    unsafe_allow_html=True,
                )
            elif pred_class == 2:
                st.markdown(
                    f'<div class="alert-box"><strong>⚠️</strong><br>{CLASS_DESCRIPTIONS[pred_class]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="warning-box"><strong>🚨</strong><br>{CLASS_DESCRIPTIONS[pred_class]}</div>',
                    unsafe_allow_html=True,
                )

            shap_vals_1d, base_val = local_shap_1d_and_base_value(
                explainer=explainer,
                instance_scaled=instance,
                class_idx=pred_class,
                feature_cols=feature_cols,
            )

            local_exp = shap.Explanation(
                values=shap_vals_1d,
                base_values=base_val,
                data=instance_original.values[0],
                feature_names=feature_cols,
            )

            st.subheader("📊 Why This Risk Level?")
            feature_impacts = pd.DataFrame(
                {
                    "Feature": feature_cols,
                    "Value": instance_original.values[0],
                    "Impact": shap_vals_1d,
                }
            ).sort_values(by="Impact", key=abs, ascending=False)

            top = feature_impacts.iloc[0]
            top_feature = str(top["Feature"])
            top_value = float(top["Value"])
            top_impact = float(top["Impact"])

            # Use feature-specific explanation hints (when available)
            hint = FEATURE_HIGH_EXPLANATION.get(top_feature, {})
            reason_text = hint.get("reason", "this feature strongly influences the prediction")
            advice_text = hint.get("advice", "monitor air quality conditions and follow public health guidance")

            direction = "increases" if top_impact > 0 else "decreases"
            col_plot, col_text = st.columns([2, 1], gap="large")
            with col_plot:
                shap.plots.waterfall(local_exp, max_display=len(feature_cols), show=False)
                fig = plt.gcf()
                fig.set_size_inches(10, 6)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col_text:
                render_explanation_card(
                    f"""
<p><strong>Main Reason:</strong> Predicted <strong>Class {pred_class}</strong> mainly because <strong>{top_feature}</strong>
({top_value:.2f}) <strong>{direction}</strong> the health impact (SHAP: {top_impact:+.4f}).</p>
<p><strong>Why:</strong> {reason_text}.</p>
<p><strong>What you can do:</strong> {advice_text}.</p>
                    """
                )

