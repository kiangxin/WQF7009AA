from __future__ import annotations

from typing import Tuple

import numpy as np
import shap


def local_shap_1d_and_base_value(
    explainer: "shap.TreeExplainer",
    instance_scaled: np.ndarray,
    class_idx: int,
    feature_cols: list[str],
) -> Tuple[np.ndarray, float]:
    """
    Normalize SHAP outputs to (n_features,) for a single instance and a class.
    """
    raw = explainer.shap_values(instance_scaled)
    n_features = len(feature_cols)

    if isinstance(raw, list):
        shap_2d = np.asarray(raw[class_idx])
        shap_1d = shap_2d[0]
        base = np.asarray(explainer.expected_value)[class_idx]
        return np.asarray(shap_1d).reshape(-1), float(base)

    arr = np.asarray(raw)
    if arr.ndim == 2:
        shap_1d = arr[0]
        base = explainer.expected_value
        return np.asarray(shap_1d).reshape(-1), float(np.asarray(base).reshape(-1)[0])

    if arr.ndim == 3:
        if arr.shape[2] == n_features:
            shap_1d = arr[0, class_idx, :]
        elif arr.shape[1] == n_features:
            shap_1d = arr[0, :, class_idx]
        else:
            raise ValueError(f"Unexpected SHAP shape for local explanation: {arr.shape}")

        base_vals = np.asarray(explainer.expected_value).reshape(-1)
        base = base_vals[class_idx] if len(base_vals) > 1 else base_vals[0]
        return np.asarray(shap_1d).reshape(-1), float(base)

    raise ValueError(f"Unexpected SHAP ndim for local explanation: {arr.ndim} with shape {arr.shape}")


def _global_shap_to_class_list(
    raw_shap_values,
    feature_cols: list[str],
) -> list[np.ndarray]:
    """
    Convert SHAP global output into list[class] -> (n_samples, n_features).
    """
    n_features = len(feature_cols)

    if isinstance(raw_shap_values, list):
        return [np.asarray(v) for v in raw_shap_values]

    arr = np.asarray(raw_shap_values)
    if arr.ndim == 3:
        # (n_samples, n_classes, n_features) or (n_samples, n_features, n_classes)
        if arr.shape[2] == n_features:
            return [arr[:, i, :] for i in range(arr.shape[1])]
        if arr.shape[1] == n_features:
            return [arr[:, :, i] for i in range(arr.shape[2])]
        raise ValueError(f"Unexpected SHAP 3D shape: {arr.shape}")

    raise ValueError(f"Unexpected SHAP shape for global explanation: {arr.shape}")


def global_shap_total_exp(
    explainer: "shap.TreeExplainer",
    X_scaled: np.ndarray,
    X_original: np.ndarray,
    feature_cols: list[str],
) -> shap.Explanation:
    """
    Build a 'Total' SHAP Explanation across classes.

    Reason: beeswarm requires (n_samples, n_features). For multi-class, we aggregate
    by taking the mean SHAP value across classes (signed) for beeswarm-like view.
    """
    raw = explainer.shap_values(X_scaled)
    per_class = _global_shap_to_class_list(raw, feature_cols)
    shap_vals_total = np.mean(np.stack(per_class, axis=0), axis=0)  # (n_samples, n_features)

    base_vals = np.asarray(explainer.expected_value).reshape(-1)
    base_val = float(np.mean(base_vals)) if base_vals.size else 0.0

    return shap.Explanation(
        values=shap_vals_total,
        base_values=np.repeat(base_val, shap_vals_total.shape[0]),
        data=X_original,
        feature_names=feature_cols,
    )


def global_shap_total_bar_values(
    explainer: "shap.TreeExplainer",
    X_scaled: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Return mean(|SHAP|) aggregated across classes and samples: (n_features,).
    """
    raw = explainer.shap_values(X_scaled)
    per_class = _global_shap_to_class_list(raw, feature_cols)
    stacked = np.stack(per_class, axis=0)  # (n_classes, n_samples, n_features)
    return np.mean(np.abs(stacked), axis=(0, 1))


def global_mean_abs_by_class(
    explainer: "shap.TreeExplainer",
    X_scaled: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Return mean(|SHAP|) per class: shape (n_classes, n_features).

    Useful for 'all classes at once' stacked bar plots.
    """
    raw = explainer.shap_values(X_scaled)
    per_class = _global_shap_to_class_list(raw, feature_cols)  # list[(n_samples, n_features)]
    return np.stack([np.mean(np.abs(v), axis=0) for v in per_class], axis=0)

