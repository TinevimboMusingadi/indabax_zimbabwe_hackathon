"""K-Fold Target Encoder — leak-safe target encoding.

For each fold, encodes using statistics from the OTHER folds only.
Test set is encoded using overall train-set statistics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KFoldTargetEncoder:
    """Target-encode categoricals using CV-safe fold statistics.

    Attributes:
        smoothing: Bayesian smoothing factor to regularise rare categories.
    """

    def __init__(
        self,
        cat_cols: list[str] | None = None,
        smoothing: float = 10.0,
    ) -> None:
        self.cat_cols = cat_cols
        self.smoothing = smoothing
        self._global_maps: dict[str, dict[str, float]] = {}
        self._global_mean: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds_df: pd.DataFrame | None = None,
    ) -> KFoldTargetEncoder:
        """Fit global statistics (used for test-set encoding)."""
        cols = self.cat_cols or self._detect_cat_cols(X)
        self.cat_cols = cols
        self._global_mean = float(y.mean())

        for col in cols:
            self._global_maps[col] = self._smoothed_means(
                X[col].astype(str), y
            )
        return self

    def transform_train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Leak-safe training transform: encode using out-of-fold stats."""
        if not self._global_maps:
            raise RuntimeError("Call fit() first.")

        df = X.copy()
        for col in self.cat_cols or []:
            encoded = np.full(len(df), self._global_mean)
            for fold_id in folds_df["fold_id"].unique():
                val_mask = folds_df["fold_id"] == fold_id
                train_mask = ~val_mask

                oof_means = self._smoothed_means(
                    df.loc[train_mask.values, col].astype(str),
                    y.iloc[train_mask.values],
                )

                val_vals = df.loc[val_mask.values, col].astype(str)
                encoded[val_mask.values] = val_vals.map(oof_means).fillna(
                    self._global_mean
                )

            df[col] = encoded
        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test set using global (full-train) statistics."""
        if not self._global_maps:
            raise RuntimeError("Call fit() first.")

        df = X.copy()
        for col in self.cat_cols or []:
            mapping = self._global_maps[col]
            df[col] = (
                df[col]
                .astype(str)
                .map(mapping)
                .fillna(self._global_mean)
                .astype(float)
            )
        return df

    def _smoothed_means(
        self, col: pd.Series, y: pd.Series
    ) -> dict[str, float]:
        """Compute smoothed target means per category."""
        combined = pd.DataFrame({"cat": col.values, "y": y.values})
        stats = combined.groupby("cat")["y"].agg(["mean", "count"])
        global_mean = y.mean()
        smooth = self.smoothing
        smoothed = (
            (stats["count"] * stats["mean"] + smooth * global_mean)
            / (stats["count"] + smooth)
        )
        return smoothed.to_dict()

    @staticmethod
    def _detect_cat_cols(X: pd.DataFrame) -> list[str]:
        return [
            c for c in X.columns
            if X[c].dtype.name in ("category", "object")
        ]
