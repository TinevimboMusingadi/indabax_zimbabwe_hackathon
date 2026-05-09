"""SHAP-based model interpretability."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def shap_summary(
    model,  # type: ignore[no-untyped-def]
    X_val: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
    save_path: str | None = None,
) -> np.ndarray | None:
    """Compute and optionally plot SHAP values for a tree-based model.

    Returns:
        SHAP values array, or None if SHAP is unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping explanation.")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
    except Exception:
        logger.warning("TreeExplainer failed — trying KernelExplainer.")
        try:
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_val, min(100, len(X_val))),
            )
            shap_values = explainer.shap_values(X_val)
        except Exception as e:
            logger.error("SHAP explanation failed: %s", e)
            return None

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(
            shap_values,
            X_val,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("SHAP plot saved to %s", save_path)
        plt.close()
    except Exception as e:
        logger.warning("Could not generate SHAP plot: %s", e)

    return shap_values
