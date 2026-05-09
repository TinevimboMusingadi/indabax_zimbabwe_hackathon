"""Gradient-boosted model wrappers with GPU auto-detection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class LGBMModel(BaseModel):
    """LightGBM wrapper."""

    name = "lgbm"

    def __init__(self, use_gpu: bool = True, **kwargs: Any) -> None:
        from lightgbm import LGBMClassifier

        defaults: dict[str, Any] = {
            "n_estimators": 1000,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "is_unbalance": True,
            "objective": "binary",
            "metric": "auc",
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        if use_gpu and _gpu_available():
            defaults["device"] = "gpu"
        defaults.update(kwargs)
        self.model = LGBMClassifier(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                _lgbm_early_stopping(50),
                _lgbm_log_eval(100),
            ]
        self.model.fit(X_train, y_train, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class XGBModel(BaseModel):
    """XGBoost wrapper."""

    name = "xgb"

    def __init__(self, use_gpu: bool = True, **kwargs: Any) -> None:
        from xgboost import XGBClassifier

        defaults: dict[str, Any] = {
            "n_estimators": 1000,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "auc",
            "random_state": 42,
            "verbosity": 0,
        }
        defaults["tree_method"] = "hist"
        if use_gpu and _gpu_available():
            defaults["device"] = "cuda"

        defaults.update(kwargs)

        # Compute scale_pos_weight later in fit if not provided
        self._auto_scale = "scale_pos_weight" not in kwargs
        self.model = XGBClassifier(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        if self._auto_scale:
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            self.model.set_params(
                scale_pos_weight=float(neg / max(pos, 1))
            )

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        self.model.fit(X_train, y_train, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class CatBoostModel(BaseModel):
    """CatBoost wrapper."""

    name = "catboost"

    def __init__(self, use_gpu: bool = True, **kwargs: Any) -> None:
        from catboost import CatBoostClassifier

        defaults: dict[str, Any] = {
            "iterations": 1000,
            "depth": 7,
            "learning_rate": 0.05,
            "auto_class_weights": "Balanced",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": 0,
        }
        if use_gpu and _gpu_available():
            defaults["task_type"] = "GPU"
        defaults.update(kwargs)
        self.model = CatBoostClassifier(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = (X_val, y_val)
            fit_params["early_stopping_rounds"] = 50
        self.model.fit(X_train, y_train, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def _lgbm_early_stopping(rounds: int):  # type: ignore[no-untyped-def]
    """LightGBM early stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=rounds, verbose=False)


def _lgbm_log_eval(period: int):  # type: ignore[no-untyped-def]
    """LightGBM logging callback."""
    from lightgbm import log_evaluation
    return log_evaluation(period=period)
