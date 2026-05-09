"""Rank-averaged ensemble — robust and hard to overfit."""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


def rank_average(pred_list: list[np.ndarray]) -> np.ndarray:
    """Blend predictions by averaging their ranks.

    Each model's predictions are ranked, normalised to [0, 1],
    then averaged across models.

    Args:
        pred_list: List of prediction arrays (same length).

    Returns:
        Blended prediction array in [0, 1].
    """
    if not pred_list:
        raise ValueError("pred_list is empty.")

    ranked = [rankdata(p) / len(p) for p in pred_list]
    blended = np.mean(ranked, axis=0)
    logger.info(
        "Rank-averaged %d models -> range [%.4f, %.4f]",
        len(pred_list), blended.min(), blended.max(),
    )
    return blended
