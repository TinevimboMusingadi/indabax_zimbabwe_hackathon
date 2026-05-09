"""Ordinal Encoder wrapper — maps categories to integers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder as _SkOrdinalEncoder

logger = logging.getLogger(__name__)


class OrdinalEncoder:
    """Ordinal encode categoricals; unseen categories get -1."""

    def __init__(self, cat_cols: list[str] | None = None) -> None:
        self.cat_cols = cat_cols
        self._encoder: _SkOrdinalEncoder | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> OrdinalEncoder:
        del y
        cols = self.cat_cols or self._detect_cat_cols(X)
        self._encoder = _SkOrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
            dtype=np.float32,
        )
        self._encoder.fit(X[cols].astype(str))
        self.cat_cols = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._encoder is None or self.cat_cols is None:
            raise RuntimeError("Call fit() first.")
        df = X.copy()
        encoded = self._encoder.transform(df[self.cat_cols].astype(str))
        for i, col in enumerate(self.cat_cols):
            df[col] = encoded[:, i]
        return df

    @staticmethod
    def _detect_cat_cols(X: pd.DataFrame) -> list[str]:
        return [
            c for c in X.columns
            if X[c].dtype.name in ("category", "object")
        ]
