"""TabNet wrapper using pytorch-tabnet."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class TabNetModel(BaseModel):
    """TabNet classifier with attentive feature selection."""

    name = "tabnet"

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        max_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self._model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        from pytorch_tabnet.tab_model import TabNetClassifier

        self._model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            seed=42,
            verbose=0,
        )

        fit_kwargs: dict[str, Any] = {
            "X_train": X_train.astype(np.float32),
            "y_train": y_train.astype(np.int64),
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
        }
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [
                (X_val.astype(np.float32), y_val.astype(np.int64))
            ]
            fit_kwargs["eval_metric"] = ["auc"]

        self._model.fit(**fit_kwargs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict_proba(X.astype(np.float32))[:, 1]
