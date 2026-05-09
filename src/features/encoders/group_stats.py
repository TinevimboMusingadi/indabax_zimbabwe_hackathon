"""Group-level statistics encoder — province/sector default rates.

Fit on train only, then join to test to prevent target leakage.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class GroupStatsEncoder:
    """Compute group-level target statistics for categorical columns."""

    def __init__(
        self,
        group_cols: list[str] | None = None,
    ) -> None:
        self.group_cols = group_cols or ["province", "employment_sector"]
        self._stats: dict[str, pd.DataFrame] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> GroupStatsEncoder:
        for col in self.group_cols:
            if col not in X.columns:
                continue
            combined = pd.DataFrame({
                col: X[col].astype(str).values,
                "target": y.values,
            })
            stats = (
                combined.groupby(col)["target"]
                .agg(["mean", "count"])
                .reset_index()
            )
            stats.columns = [
                col,
                f"{col}_default_rate",
                f"{col}_loan_count",
            ]
            self._stats[col] = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._stats:
            raise RuntimeError("Call fit() first.")
        df = X.copy()
        for col, stats in self._stats.items():
            if col not in df.columns:
                continue
            key = df[col].astype(str)
            stats_key = stats[col].astype(str)
            lookup = stats.copy()
            lookup[col] = stats_key
            merged = key.to_frame().merge(lookup, on=col, how="left")
            for extra_col in merged.columns:
                if extra_col != col:
                    df[extra_col] = merged[extra_col].values
        return df

    @staticmethod
    def _detect_cat_cols(X: pd.DataFrame) -> list[str]:
        return [
            c for c in X.columns
            if X[c].dtype.name in ("category", "object")
        ]
