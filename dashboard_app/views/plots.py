from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from dashboard_app.config import CLASS_DESCRIPTIONS


def plot_global_shap_for_class(
    explainer: "shap.TreeExplainer",
    X_data,
    X_scaled: np.ndarray,
    feature_cols: list[str],
    class_idx: int,
) -> None:
    """
    Global SHAP beeswarm + bar for a single class.
    """
    raw_shap_values = explainer.shap_values(X_scaled)

    if isinstance(raw_shap_values, list):
        shap_vals_2d = raw_shap_values[class_idx]
        base_val = explainer.expected_value[class_idx]
    else:
        raw = np.asarray(raw_shap_values)
        if raw.ndim != 3:
            raise ValueError(f"Unexpected SHAP shape: {raw.shape}")
        if raw.shape[2] == len(feature_cols):
            shap_vals_2d = raw[:, class_idx, :]
        elif raw.shape[1] == len(feature_cols):
            shap_vals_2d = raw[:, :, class_idx]
        else:
            raise ValueError(f"Unexpected SHAP 3D shape: {raw.shape}")
        base_val = np.asarray(explainer.expected_value)[class_idx]

    exp = shap.Explanation(
        values=shap_vals_2d,
        base_values=np.repeat(base_val, shap_vals_2d.shape[0]),
        data=X_data.values,
        feature_names=feature_cols,
    )

    # Beeswarm
    shap.plots.beeswarm(exp, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.title(f"SHAP Beeswarm - Class {class_idx}")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Bar
    shap.plots.bar(exp, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.title(f"SHAP Feature Importance - Class {class_idx}")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_global_shap_total_bar(feature_cols: list[str], total_mean_abs: np.ndarray) -> None:
    """
    Bar plot for Total mean(|SHAP|) across classes.
    """
    order = np.argsort(total_mean_abs)[::-1]
    top_idx = order[:12]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_cols[i] for i in reversed(top_idx)],
        [total_mean_abs[i] for i in reversed(top_idx)],
        color="#64b5f6",
        edgecolor="#42a5f5",
    )
    ax.set_xlabel("Mean |SHAP| (averaged across classes + samples)")
    ax.set_title("Global SHAP - Total (All Classes)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_global_shap_all_classes_stacked_bar(
    feature_cols: list[str],
    mean_abs_by_class: np.ndarray,
    max_display: int = 12,
) -> None:
    """
    Plot a SHAP summary bar for *all classes at once* (stacked bars).

    This matches the common "summary_plot(plot_type='bar')" multi-class visualization,
    where each feature bar is split into colored segments by class.
    """
    if mean_abs_by_class.ndim != 2:
        raise ValueError(f"Expected (n_classes, n_features), got {mean_abs_by_class.shape}")

    n_classes, n_features = mean_abs_by_class.shape
    if n_features != len(feature_cols):
        raise ValueError("mean_abs_by_class columns must match feature_cols length")

    total = mean_abs_by_class.sum(axis=0)
    order = np.argsort(total)[::-1]
    top_idx = order[: max_display]

    # For horizontal stacked bars, plot from least->most so y-axis shows top feature at top
    top_idx = list(reversed(top_idx))

    fig, ax = plt.subplots(figsize=(11, 7))

    cmap = plt.get_cmap("tab10")
    left = np.zeros(len(top_idx))

    for c in range(n_classes):
        vals = [mean_abs_by_class[c, i] for i in top_idx]
        ax.barh(
            [feature_cols[i] for i in top_idx],
            vals,
            left=left,
            color=cmap(c % 10),
            edgecolor="white",
            linewidth=0.5,
            label=f"Class {c}",
        )
        left = left + np.array(vals)

    ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
    ax.set_title("SHAP Summary (All Classes) - Stacked Bar")
    ax.legend(loc="lower right", frameon=True)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_class_help() -> None:
    with st.expander("🏥 Health Impact Classes (reference)"):
        for cls in sorted(CLASS_DESCRIPTIONS.keys()):
            st.markdown(f"**Class {cls}**: {CLASS_DESCRIPTIONS[cls]}")

