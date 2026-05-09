"""Frequency Encoder — replace categories with their train-set counts."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FrequencyEncoder:
    """Map each category to its normalised frequency in the train set."""

    def __init__(self, cat_cols: list[str] | None = None) -> None:
        self.cat_cols = cat_cols
        self._freq_maps: dict[str, dict] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> FrequencyEncoder:
        del y
        cols = self.cat_cols or self._detect_cat_cols(X)
        self.cat_cols = cols
        for col in cols:
            vc = X[col].astype(str).value_counts(normalize=True)
            self._freq_maps[col] = vc.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._freq_maps:
            raise RuntimeError("Call fit() first.")
        df = X.copy()
        for col in self.cat_cols or []:
            mapping = self._freq_maps[col]
            df[col] = (
                df[col].astype(str).map(mapping).fillna(0.0).astype(float)
            )
        return df

    @staticmethod
    def _detect_cat_cols(X: pd.DataFrame) -> list[str]:
        return [
            c for c in X.columns
            if X[c].dtype.name in ("category", "object")
        ]
