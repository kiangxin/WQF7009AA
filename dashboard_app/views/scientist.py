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

    def render_explanation_card(markdown_text: str, title: str = "Interpretation") -> None:
        st.markdown(
            f"""
<div class="explanation-card">
<h4>{title}</h4>
{markdown_text}
</div>
            """,
            unsafe_allow_html=True,
        )

    class_labels = sorted(np.unique(y))
    options = ["All Classes"] + [str(c) for c in class_labels]

    selected = st.selectbox(
        "Select class:",
        options,
        format_func=lambda v: v
        if v == "All Classes"
        else f"Class {v}",
        key="scientist_class_select",
    )

    if selected == "All Classes":
        st.subheader("Global SHAP Summary (All Classes at once)")
        col_plot, col_text = st.columns([2, 1], gap="large")
        with col_plot:
            mean_abs = global_mean_abs_by_class(explainer, X_scaled_shap, feature_cols)
            plot_global_shap_all_classes_stacked_bar(feature_cols, mean_abs, max_display=12)
        with col_text:
            render_explanation_card(
                """
<p><strong>Short summary</strong></p>
<p>AQI is the dominant global driver across all classes, contributing the largest share of SHAP magnitude and strongly influencing class separation.</p>
<p>Particulate pollutants (PM2.5 and PM10) are the next most influential features, with consistent impact across multiple classes, indicating their central role in distinguishing health impact levels.</p>
<p>Gaseous pollutants (O₃, NO₂, SO₂) show moderate but meaningful contributions, suggesting secondary yet class-relevant effects.</p>
<p>Meteorological factors (Wind Speed, Humidity, Temperature) have smaller but non-negligible influence, mainly refining predictions rather than driving them.</p>
<p>Health outcome variables (Respiratory, Hospital Admissions, Cardiovascular Cases) contribute the least globally, implying they act more as supportive/contextual signals than primary predictors.</p>
<p><strong>Overall takeaway</strong></p>
<p>The multiclass model is primarily driven by air quality indicators, especially AQI and particulate matter, with pollutants dominating global importance while weather and health variables play complementary roles.</p>
                """
            )
        return

    class_idx = int(selected)
    st.subheader(f"Global SHAP - Class {class_idx}")
    col_plot, col_text = st.columns([2, 1], gap="large")
    with col_plot:
        plot_global_shap_for_class(explainer, X_shap, X_scaled_shap, feature_cols, class_idx)
    with col_text:
        if class_idx == 0:
            render_explanation_card(
                """
<p>This plot explains how feature values influence predictions towards or away from Class 0.</p>
<p><strong>AQI is the strongest driver:</strong> High AQI values (red) strongly push predictions towards Class 0, while low AQI values (blue) push predictions away, indicating that Class 0 corresponds to very poor air quality and severe health risk.</p>
<p><strong>PM2.5 and PM10 show the same pattern:</strong> Higher particulate concentrations increase the likelihood of Class 0, whereas lower levels reduce it, reinforcing that extreme particulate pollution is a key indicator of very high health impact.</p>
<p><strong>Gaseous pollutants (O₃, NO₂, SO₂) have moderate influence:</strong> Higher values generally shift predictions towards Class 0, suggesting that multiple pollutant types jointly contribute to severe air quality conditions.</p>
<p><strong>Meteorological variables (temperature, humidity, wind speed) show smaller, mixed effects:</strong> These variables mainly fine-tune predictions rather than directly defining Class 0, indicating that weather conditions modulate but do not drive extreme health risk.</p>
<p><strong>Health outcome variables have limited global impact:</strong> Their relatively small SHAP values suggest that Class 0 is identified primarily through environmental exposure signals, rather than directly through observed health outcome counts.</p>
<p><strong>Key takeaway:</strong> Class 0 represents a very high health impact / very unhealthy air quality regime, where consistently high pollution levels especially AQI and particulate matter drive the model’s confidence in this class.</p>
                """
            )
        elif class_idx == 1:
            render_explanation_card(
                """
<p>This plot explains how feature values influence predictions towards or away from Class 1.</p>
<p><strong>AQI functions via "Exclusionary Logic":</strong> AQI acts as a filter. The plot shows that lower AQI values (blue points) actually generate positive SHAP contributions, while higher values (red points) penalize the prediction. This indicates the model defines Class 1 not by the presence of extreme peaks (which belong to Class 0), but by the <em>absence</em> of them—capturing "high but not hazardous" air.</p>
<p><strong>PM2.5 and PM10 exhibit a "Bounded" signature:</strong> Particulates show a "ceiling effect." Moderate-to-high concentrations contribute positively, but the most extreme concentrations (far-right red points) shift predictions away. This confirms Class 1 occupies a specific pollution interval: dangerous, but below the Class 0 threshold.</p>
<p><strong>Meteorological variables indicate atmospheric stagnation:</strong> High humidity (red points) and low wind speeds (blue points) push predictions toward Class 1. This implies the class is physically defined by stagnant conditions that trap pollutants, sustaining a high-impact environment.</p>
<p><strong>Health outcome &amp; Gaseous variables:</strong> These features play secondary roles. Gaseous pollutants reinforce the particulate signals, while health outcomes remain low-magnitude, confirming the model relies on environmental precursors.</p>
<p><strong>Key takeaway:</strong> Class 1 represents a "Bounded High-Impact" regime. Scientifically, the model treats this as a transitional buffer. It is defined by atmospheric stagnation and sub-extreme pollution, distinguishing itself from Class 0 primarily by rejecting the most toxic outliers.</p>
                """
            )
        elif class_idx == 2:
            render_explanation_card(
                """
<p>This plot explains how feature values influence predictions towards or away from Class 2.</p>
<p><strong>AQI is the strongest driver:</strong> Low to moderate AQI values push predictions towards Class 2, while high AQI values shift predictions away, indicating that Class 2 corresponds to acceptable to moderately clean air quality, rather than polluted conditions.</p>
<p><strong>PM2.5 and PM10 show a clear centering effect:</strong> Lower to moderate particulate concentrations increase the likelihood of Class 2, whereas higher levels reduce it, separating this class from higher health-impact regimes.</p>
<p><strong>Gaseous pollutants (O₃, NO₂, SO₂) show balanced, non-extreme influence:</strong> Lower concentrations tend to support Class 2 predictions, while higher values increasingly push predictions away, though their effects are weaker than particulate matter.</p>
<p><strong>Meteorological variables (humidity, temperature, wind speed) play a supporting role:</strong> These features contribute to fine-grained separation within the moderate regime rather than acting as primary drivers.</p>
<p><strong>Health outcome variables have minimal impact:</strong> Their limited contribution indicates that Class 2 is primarily defined by environmental conditions before severe health impacts emerge.</p>
<p><strong>Key takeaway:</strong> Class 2 represents a moderate health impact regime, characterized by mid-range pollution levels, acting as a central transition point between unhealthy conditions (Class 1) and lower-risk air quality states.</p>
                """
            )
        elif class_idx == 3:
            render_explanation_card(
                """
<p>This plot explains how feature values influence predictions towards or away from Class 3.</p>
<p><strong>PM2.5 is the strongest driver:</strong> Lower PM2.5 values push predictions towards towards Class 3, while higher particulate concentrations strongly shift predictions away, indicating that relatively clean particulate conditions are a defining characteristic of this class.</p>
<p><strong>O₃ is the second most influential feature:</strong> Mid-range O₃ concentrations are most compatible with Class 3, while both low and high O₃ values tend to push predictions away. This suggests that Class 3 corresponds to stable ozone exposure rather than extreme conditions.</p>
<p><strong>AQI shows a centered, non-extreme pattern:</strong> AQI values clustered around the mid range are associated with SHAP values close to zero, while both lower and higher AQI values tend to produce more negative SHAP values. This indicates that Class 3 is defined by stable, non-extreme air quality conditions rather than very clean or highly polluted air.</p>
<p><strong>Other gaseous pollutants (NO₂, SO₂) have secondary influence:</strong> Lower concentrations tend to be less penalized, while higher values increasingly push predictions away, though their impact is weaker than PM2.5 and O₃.</p>
<p><strong>Meteorological variables (humidity, temperature, wind speed) play a supporting role:</strong> These features contribute to fine-grained separation within healthy conditions rather than acting as primary drivers.</p>
<p><strong>Health outcome variables have minimal impact:</strong> Their SHAP values are tightly clustered around zero, indicating that Class 3 is identified primarily by environmental exposure patterns rather than health burden signals.</p>
<p><strong>Key takeaway:</strong> Class 3 represents a healthy air quality regime, characterized by balanced (non-extreme) pollutant levels, particularly PM2.5 and O₃, with minimal associated health impact.</p>
                """
            )
        elif class_idx == 4:
            render_explanation_card(
                """
<p>This plot explains how feature values influence predictions towards or away from Class 4.</p>
<p><strong>Critical Logic Inversion detected in PM₁₀ and AQI:</strong> Despite Class 4 representing the "Very Low Health Impact" (healthiest) regime, the model incorrectly identifies higher PM₁₀ and AQI values (red points) as positive drivers for this class. Physically, we would expect low pollution (blue points) to drive this prediction. This "Logic Inversion" suggests the model is confusing Class 4 with the high-pollution Class 0, likely due to data imbalance.</p>
<p><strong>Wind speed shows an inverse relationship:</strong> Higher wind speeds (red points) produce negative SHAP values, shifting predictions away from Class 4, whereas lower wind speeds (blue points) push predictions toward this class. This suggests that the model links low dispersion or stagnant conditions with Class 4 predictions, despite the class representing low health impact.</p>
<p><strong>Secondary meteorological and trace gas variables (SO₂, NO₂, temperature):</strong> These features show low-magnitude SHAP values clustered near zero, indicating a minor, fine-tuning role rather than primary influence.</p>
<p><strong>Health outcome variables show weak influence:</strong> Respiratory and cardiovascular case counts remain mostly centered near zero, consistent with the definition of Class 4 as a very low health impact category, though slight positive tails suggest residual overlap with other classes.</p>
<p><strong>Key takeaway:</strong> Class 4 exhibits a significant <strong>model alignment failure</strong>. The SHAP analysis reveals that the model has failed to learn the correct "clean air" signature, instead learning a high-pollution pattern almost identical to Class 0. This confirms that the model struggles to distinguish the "Very Low" impact minority class from the "Very High" impact extremes.</p>
                """
            )

