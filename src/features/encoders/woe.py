"""Weight of Evidence (WOE) encoder — classic credit-risk encoding."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WOEEncoder:
    """Compute WOE per category: ln(% of non-events / % of events).

    Smoothed to avoid division-by-zero for rare categories.
    """

    def __init__(
        self,
        cat_cols: list[str] | None = None,
        smoothing: float = 0.5,
    ) -> None:
        self.cat_cols = cat_cols
        self.smoothing = smoothing
        self._woe_maps: dict[str, dict[str, float]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> WOEEncoder:
        cols = self.cat_cols or self._detect_cat_cols(X)
        self.cat_cols = cols
        total_events = y.sum()
        total_non_events = len(y) - total_events

        for col in cols:
            combined = pd.DataFrame({
                "cat": X[col].astype(str).values,
                "y": y.values,
            })
            agg = combined.groupby("cat")["y"].agg(["sum", "count"])
            agg.columns = ["events", "total"]
            agg["non_events"] = agg["total"] - agg["events"]

            s = self.smoothing
            pct_events = (agg["events"] + s) / (total_events + s * 2)
            pct_non = (agg["non_events"] + s) / (total_non_events + s * 2)
            woe = np.log(pct_non / pct_events)

            self._woe_maps[col] = woe.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._woe_maps:
            raise RuntimeError("Call fit() first.")
        df = X.copy()
        for col in self.cat_cols or []:
            mapping = self._woe_maps[col]
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
