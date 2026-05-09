"""Cross-validation trainer with OOF and test predictions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.loader import ID_COL, TARGET_COL
from src.data.splits import get_fold_indices
from src.models.base import BaseModel, build_model
from src.utils.io import save_json

logger = logging.getLogger(__name__)


@dataclass
class CvResult:
    """Results from cross-validated training."""

    model_name: str
    variant: str
    oof_preds: np.ndarray
    test_preds: np.ndarray
    fold_aucs: list[float] = field(default_factory=list)
    mean_auc: float = 0.0
    std_auc: float = 0.0
    train_time_sec: float = 0.0


def cv_train(
    model_name: str,
    model_params: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    variant: str = "v2_ordinal",
    use_gpu: bool = False,
) -> CvResult:
    """Run stratified CV, collect OOF preds and averaged test preds.

    Args:
        model_name: Name key for build_model().
        model_params: Hyperparameters for the model.
        train_df: Training data including Target and ID columns.
        test_df: Test data including ID column.
        folds_df: DataFrame with 'row_idx' and 'fold_id'.
        variant: Name of the feature variant being used.
        use_gpu: Whether to enable GPU for boosted models.

    Returns:
        CvResult with OOF predictions, averaged test predictions,
        per-fold AUC scores, and timing.
    """
    feature_cols = [
        c for c in train_df.columns
        if c not in {ID_COL, TARGET_COL}
        and train_df[c].dtype.kind in ("i", "f", "u")
    ]
    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[TARGET_COL].values.astype(np.float32)
    test_feat_cols = [c for c in feature_cols if c in test_df.columns]
    X_test = test_df[test_feat_cols].values.astype(np.float32)

    X = np.nan_to_num(X, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    n_folds = folds_df["fold_id"].nunique()
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_aucs: list[float] = []

    start = time.perf_counter()

    for fold_id in range(n_folds):
        train_idx, val_idx = get_fold_indices(folds_df, fold_id)

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        params = dict(model_params)
        if model_name in ("lgbm", "xgb", "catboost"):
            params["use_gpu"] = use_gpu

        model: BaseModel = build_model(model_name, params)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds = model.predict_proba(X_val)
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)

        test_preds += model.predict_proba(X_test) / n_folds

        logger.info(
            "[%s] Fold %d/%d — AUC: %.4f",
            model_name, fold_id + 1, n_folds, fold_auc,
        )

    elapsed = time.perf_counter() - start
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    logger.info(
        "[%s] CV AUC: %.4f +/- %.4f (%.1fs)",
        model_name, mean_auc, std_auc, elapsed,
    )

    result = CvResult(
        model_name=model_name,
        variant=variant,
        oof_preds=oof_preds,
        test_preds=test_preds,
        fold_aucs=fold_aucs,
        mean_auc=mean_auc,
        std_auc=std_auc,
        train_time_sec=elapsed,
    )

    log_data = {
        "model": model_name,
        "variant": variant,
        "fold_aucs": fold_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "train_time_sec": elapsed,
    }
    save_json(log_data, f"logs/training/{model_name}_{variant}.json")

    return result
