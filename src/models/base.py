"""Abstract base model for all model wrappers."""

from __future__ import annotations

import abc
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """Uniform interface for all models in the pipeline.

    Subclasses must implement fit() and predict_proba().
    """

    name: str = "base"

    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train the model."""

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for the positive class."""

    def save(self, path: str | Path) -> None:
        """Pickle the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved model '%s' -> %s", self.name, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Load a pickled model."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model from %s", path)
        return model


def build_model(name: str, params: dict[str, Any]) -> BaseModel:
    """Factory: instantiate a model wrapper by name."""
    from src.models.boosted import CatBoostModel, LGBMModel, XGBModel
    from src.models.classical import (
        ExtraTreesModel,
        LRModel,
        RFModel,
    )
    from src.models.deep_mlp import MLPModel
    from src.models.tabnet_model import TabNetModel

    registry: dict[str, type[BaseModel]] = {
        "lgbm": LGBMModel,
        "xgb": XGBModel,
        "catboost": CatBoostModel,
        "lr": LRModel,
        "rf": RFModel,
        "extra_trees": ExtraTreesModel,
        "mlp": MLPModel,
        "tabnet": TabNetModel,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[name](**params)
