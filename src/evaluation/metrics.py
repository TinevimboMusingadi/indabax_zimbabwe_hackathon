"""Evaluation metrics and threshold tuning."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute full metrics suite for a model's predictions.

    Args:
        name: Model identifier.
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Decision threshold for binary metrics.

    Returns:
        Dict of metric name -> value.
    """
    y_pred = (y_prob >= threshold).astype(int)
    results = {
        "model": name,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "threshold": threshold,
    }
    logger.info(
        "[%s] AUC=%.4f  PR-AUC=%.4f  F1=%.4f",
        name, results["roc_auc"], results["pr_auc"], results["f1"],
    )
    return results


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Find the threshold that maximises F1.

    Returns:
        (best_threshold, best_f1).
    """
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, y_prob
    )
    f1_scores = (
        2 * precisions * recalls / (precisions + recalls + 1e-8)
    )
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    logger.info(
        "Best threshold=%.3f  F1=%.4f", best_thresh, best_f1
    )
    return best_thresh, best_f1
