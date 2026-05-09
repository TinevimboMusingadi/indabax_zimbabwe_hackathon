"""One-Hot Encoder wrapper with sklearn API."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class OHEEncoder:
    """One-hot encode categorical columns, ignoring unseen categories."""

    def __init__(self, cat_cols: list[str] | None = None) -> None:
        self.cat_cols = cat_cols
        self._encoder: OneHotEncoder | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> OHEEncoder:
        del y
        cols = self.cat_cols or self._detect_cat_cols(X)
        self._encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        self._encoder.fit(X[cols].astype(str))
        self._feature_names = list(self._encoder.get_feature_names_out(cols))
        self.cat_cols = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._encoder is None or self.cat_cols is None:
            raise RuntimeError("Call fit() first.")
        encoded = self._encoder.transform(X[self.cat_cols].astype(str))
        ohe_df = pd.DataFrame(
            encoded, columns=self._feature_names, index=X.index
        )
        return pd.concat(
            [X.drop(columns=self.cat_cols), ohe_df], axis=1
        )

    @staticmethod
    def _detect_cat_cols(X: pd.DataFrame) -> list[str]:
        return [
            c for c in X.columns
            if X[c].dtype.name in ("category", "object")
        ]
