"""Probability calibration for OOF predictions."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def calibrate_platt(
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    test_preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Platt scaling — fit logistic regression on OOF preds."""
    lr = LogisticRegression()
    lr.fit(oof_preds.reshape(-1, 1), y_true)
    cal_oof = lr.predict_proba(oof_preds.reshape(-1, 1))[:, 1]
    cal_test = lr.predict_proba(test_preds.reshape(-1, 1))[:, 1]
    logger.info("Applied Platt calibration.")
    return cal_oof, cal_test


def calibrate_isotonic(
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    test_preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic regression calibration on OOF preds."""
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_preds, y_true)
    cal_oof = ir.predict(oof_preds)
    cal_test = ir.predict(test_preds)
    logger.info("Applied isotonic calibration.")
    return cal_oof, cal_test
