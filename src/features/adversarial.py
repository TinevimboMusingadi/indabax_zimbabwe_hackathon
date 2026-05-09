"""Adversarial validation — detect train/test distribution shift."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.data.loader import ID_COL, TARGET_COL

logger = logging.getLogger(__name__)


def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_cv: int = 5,
) -> float:
    """Train a classifier to distinguish train from test rows.

    A high AUC (>0.7) indicates significant distribution shift.

    Returns:
        Mean cross-validated AUC for the train-vs-test classifier.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.warning("LightGBM not installed — skipping adversarial.")
        return 0.5

    drop = {ID_COL, TARGET_COL}
    cols = [
        c for c in train.columns
        if c not in drop and c in test.columns
    ]
    num_cols = [
        c for c in cols
        if train[c].dtype.kind in ("i", "f")
    ]

    tr = train[num_cols].copy()
    te = test[num_cols].copy()
    tr["_is_test"] = 0
    te["_is_test"] = 1

    combined = pd.concat([tr, te], ignore_index=True)
    X = combined.drop(columns=["_is_test"]).values
    y = combined["_is_test"].values

    X = np.nan_to_num(X, nan=0.0)

    model = LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        verbose=-1,
    )
    scores = cross_val_score(model, X, y, cv=n_cv, scoring="roc_auc")
    mean_auc = float(np.mean(scores))

    if mean_auc > 0.7:
        logger.warning(
            "ADVERSARIAL AUC = %.4f (>0.7) — significant train/test "
            "distribution shift detected!",
            mean_auc,
        )
    else:
        logger.info("Adversarial AUC = %.4f — acceptable.", mean_auc)

    return mean_auc
