from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st


def public_health_officer_view(explainer, X, X_scaled, feature_cols, y):
    st.markdown('<p class="role-header">🏥 Public Health Officer View</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Aggregate local SHAP across multiple instances for population insight.</div>',
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

    def render_explanation_card(markdown_text: str, title: str = "Top Factors") -> None:
        st.markdown(
            f"""
<div class="explanation-card">
<h4>{title}</h4>
{markdown_text}
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Restricted to Class 0 only (high risk cases).**")
    selected_class = 0

    n_instances = st.slider("Number of instances to aggregate:", 5, 200, 20, key="pho_n")

    use_specific = st.checkbox(
        "Use specific instance IDs (optional)",
        value=False,
        key="pho_use_specific",
        help="If enabled, provide comma-separated instance indices to aggregate. Otherwise, a random sample is used.",
    )

    ids_raw = ""
    if use_specific:
        ids_raw = st.text_input(
            "Instance IDs (comma-separated, e.g. 0, 12, 350):",
            value="",
            key="pho_ids",
        )

    if st.button("Generate Aggregated Analysis", key="pho_btn"):
        with st.spinner("Computing..."):
            class_indices = np.where(np.asarray(y) == selected_class)[0]
            if len(class_indices) == 0:
                st.warning("No instances for this class.")
                return

            if use_specific:
                ids_clean = [s.strip() for s in ids_raw.split(",") if s.strip()]
                if not ids_clean:
                    st.error("Please provide at least one instance ID, or disable 'Use specific instance IDs'.")
                    return

                ids: list[int] = []
                for s in ids_clean:
                    if not s.isdigit():
                        st.error(f"Invalid instance id: '{s}'. Use integers only.")
                        return
                    idx = int(s)
                    if idx < 0 or idx >= len(X):
                        st.error(f"Instance id out of range: {idx} (valid 0..{len(X)-1})")
                        return
                    ids.append(idx)

                # Keep only class-0 ids
                ids = [i for i in ids if int(y.iloc[i]) == 0]
                if not ids:
                    st.error("None of the provided IDs are Class 0.")
                    return

                selected_indices = np.array(sorted(set(ids)), dtype=int)
                n = len(selected_indices)
            else:
                n = min(n_instances, len(class_indices))
                selected_indices = np.random.choice(class_indices, n, replace=False)

            X_sel = X_scaled[selected_indices]
            X_sel_original = X.iloc[selected_indices]

            raw = explainer.shap_values(X_sel)
            if isinstance(raw, list):
                shap_vals = np.asarray(raw[selected_class])
            else:
                arr = np.asarray(raw)
                if arr.ndim == 3 and arr.shape[2] == len(feature_cols):
                    shap_vals = arr[:, selected_class, :]
                elif arr.ndim == 3 and arr.shape[1] == len(feature_cols):
                    shap_vals = arr[:, :, selected_class]
                else:
                    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

            mean_shap = shap_vals.mean(axis=0)
            mean_feature_values = X_sel_original.mean(axis=0).values

            st.subheader(f"Aggregated Local SHAP - Class {selected_class} (n={n})")
            col_plot, col_text = st.columns([2, 1], gap="large")

            with col_plot:
                # Aggregated waterfall plot
                exp = shap.Explanation(
                    values=mean_shap,
                    base_values=0.0,
                    data=mean_feature_values,
                    feature_names=feature_cols,
                )

                plt.figure(figsize=(10, 6))
                shap.plots.waterfall(exp, max_display=len(feature_cols), show=False)
                fig = plt.gcf()
                fig.patch.set_facecolor("white")
                for ax in fig.axes:
                    ax.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_text:
                # Quick summary table (light theme, no dark chart)
                df_imp = (
                    pd.DataFrame({"Feature": feature_cols, "Mean SHAP": mean_shap})
                    .assign(AbsMeanSHAP=lambda d: d["Mean SHAP"].abs())
                    .sort_values("AbsMeanSHAP", ascending=False)
                    .drop(columns=["AbsMeanSHAP"])
                )
                render_explanation_card("<p><strong>Top factors (by |mean SHAP|)</strong></p>")
                st.dataframe(df_imp.head(10), use_container_width=True)

