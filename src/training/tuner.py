"""Optuna hyperparameter tuner per model."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import OPTUNA_DIR, PipelineConfig
from src.data.loader import ID_COL, TARGET_COL
from src.data.splits import get_fold_indices

logger = logging.getLogger(__name__)


def tune_model(
    model_name: str,
    train_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    config: PipelineConfig,
) -> dict:
    """Run Optuna TPE search for a single model.

    Returns:
        Best hyperparameters dict.
    """
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feature_cols = [
        c for c in train_df.columns if c not in {ID_COL, TARGET_COL}
    ]
    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[TARGET_COL].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    use_gpu = any(
        e.use_gpu for e in config.training.models if e.name == model_name
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        from src.models.base import build_model

        if model_name in ("lgbm", "xgb", "catboost"):
            params["use_gpu"] = use_gpu

        n_folds = folds_df["fold_id"].nunique()
        fold_aucs = []

        for fold_id in range(n_folds):
            train_idx, val_idx = get_fold_indices(folds_df, fold_id)
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = build_model(model_name, params)
            model.fit(X_tr, y_tr, X_val, y_val)
            preds = model.predict_proba(X_val)
            fold_aucs.append(roc_auc_score(y_val, preds))

        return float(np.mean(fold_aucs))

    storage_path = Path(OPTUNA_DIR) / f"{model_name}.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.seed),
        study_name=model_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=config.tuner.n_trials)

    logger.info(
        "[%s] Best trial AUC: %.4f", model_name, study.best_value
    )
    logger.info("[%s] Best params: %s", model_name, study.best_params)

    return study.best_params


def _suggest_params(
    trial: "optuna.Trial",
    model_name: str,
) -> dict:
    """Suggest hyperparameters based on model type."""
    if model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 2000
            ),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 0.3, log=True
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 100
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            ),
        }
    elif model_name == "xgb":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 2000
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            ),
        }
    elif model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 0.3, log=True
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1e-8, 10.0, log=True
            ),
        }
    else:
        raise ValueError(f"No tuning search space for '{model_name}'.")
