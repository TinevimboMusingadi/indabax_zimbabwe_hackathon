"""Stacking ensemble — meta-learner on OOF predictions."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def stack_predictions(
    oof_list: list[np.ndarray],
    y_train: np.ndarray,
    test_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Train a meta-LR on OOF preds, predict on test preds.

    Args:
        oof_list: Per-model OOF prediction arrays.
        y_train: True training labels.
        test_list: Per-model test prediction arrays.

    Returns:
        (meta_oof_preds, meta_test_preds).
    """
    meta_train = np.column_stack(oof_list)
    meta_test = np.column_stack(test_list)

    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(meta_train, y_train)

    meta_oof = meta_model.predict_proba(meta_train)[:, 1]
    meta_test_preds = meta_model.predict_proba(meta_test)[:, 1]

    logger.info(
        "Stacking: %d base models -> meta LR",
        len(oof_list),
    )
    return meta_oof, meta_test_preds
