"""Classical ML model wrappers — LR, RF, Extra Trees."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LRModel(BaseModel):
    """Logistic Regression with balanced class weights."""

    name = "lr"

    def __init__(self, **kwargs: Any) -> None:
        defaults: dict[str, Any] = {
            "solver": "saga",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        }
        defaults.update(kwargs)
        self.model = LogisticRegression(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del X_val, y_val
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class RFModel(BaseModel):
    """Random Forest with balanced class weights."""

    name = "rf"

    def __init__(self, **kwargs: Any) -> None:
        defaults: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": None,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        }
        defaults.update(kwargs)
        self.model = RandomForestClassifier(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del X_val, y_val
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class ExtraTreesModel(BaseModel):
    """Extra Trees with balanced class weights."""

    name = "extra_trees"

    def __init__(self, **kwargs: Any) -> None:
        defaults: dict[str, Any] = {
            "n_estimators": 300,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        }
        defaults.update(kwargs)
        self.model = ExtraTreesClassifier(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del X_val, y_val
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
