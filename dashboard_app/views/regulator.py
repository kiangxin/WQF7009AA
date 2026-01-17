from __future__ import annotations

import numpy as np
import streamlit as st
from dice_ml import Data, Model, Dice
import pandas as pd

from dashboard_app.config import CLASS_DESCRIPTIONS
from dashboard_app.shap_utils import global_mean_abs_by_class
from dashboard_app.views.plots import (
    plot_global_shap_all_classes_stacked_bar,
    plot_global_shap_total_bar,
)


def regulator_view(explainer, X_shap, X_scaled_shap, X_full, y_full, feature_cols, pipeline):
    st.markdown('<p class="role-header">⚖️ Regulator View</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Overall Global SHAP (all classes at once) + counterfactuals for instance_idx=0.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Global SHAP (All Classes)")
    mean_abs = global_mean_abs_by_class(explainer, X_scaled_shap, feature_cols)
    plot_global_shap_all_classes_stacked_bar(feature_cols, mean_abs, max_display=12)

    st.subheader("Global SHAP (Total)")
    total_mean_abs = mean_abs.mean(axis=0)
    plot_global_shap_total_bar(feature_cols, total_mean_abs)

    st.subheader("🔄 Counterfactual Analysis")
    st.markdown("Use **instance_idx = 0** and choose a desired target class.")

    desired_class = st.selectbox(
        "Desired target class:",
        [0, 1, 2, 3, 4],
        format_func=lambda c: f"Class {c}",
        key="reg_cf_desired_class",
    )

    if st.button("Generate Counterfactual", key="reg_cf_btn"):
        with st.spinner("Generating counterfactuals... (may take 30-60s)"):
            query_instance = X_full.iloc[0:1]
            pred = int(pipeline.predict(query_instance)[0])

            st.info(f"instance_idx=0 predicted class = {pred} (target = {desired_class})")
            if desired_class == pred:
                st.warning("Choose a different desired class than the current prediction.")
                return

            # DiCE requires a pandas DataFrame here.
            # Reason: passing numpy arrays triggers `ValueError: should provide a pandas dataframe`.
            dice_df = pd.concat([X_full, y_full.rename("HealthImpactClass")], axis=1)
            dice_data = Data(
                dataframe=dice_df,
                continuous_features=feature_cols,
                outcome_name="HealthImpactClass",
            )

            dice_model = Model(model=pipeline, backend="sklearn", model_type="classifier")
            dice_exp = Dice(dice_data, dice_model, method="genetic")

            counterfactuals = dice_exp.generate_counterfactuals(
                query_instance,
                total_CFs=3,
                desired_class=int(desired_class),
                proximity_weight=0.5,
                diversity_weight=1.0,
            )

            cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                st.warning("No counterfactuals found for this target class. Try another class.")
                return

            st.success("✅ Counterfactuals generated!")
            st.markdown("**Original instance (idx=0):**")
            st.dataframe(query_instance.T, use_container_width=True)
            st.markdown("**Counterfactuals:**")
            st.dataframe(cf_df[feature_cols], use_container_width=True)

            st.markdown("**Key changes (first counterfactual):**")
            first = cf_df.iloc[0]
            for feat in feature_cols:
                ov = float(query_instance[feat].iloc[0])
                nv = float(first[feat])
                if abs(ov - nv) > 1e-6:
                    st.markdown(f"- **{feat}**: {ov:.2f} → {nv:.2f}")

