"""Tests for WOE encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.encoders.woe import WOEEncoder


class TestWOEEncoder:

    def test_fit_transform(self, synthetic_train):
        df = synthetic_train[["province"]].copy()
        y = synthetic_train["Target"]
        enc = WOEEncoder(cat_cols=["province"])
        enc.fit(df, y)
        result = enc.transform(df)

        assert result["province"].dtype == float
        assert not result["province"].isna().any()

    def test_unseen_gets_zero(self, synthetic_train):
        df = synthetic_train[["province"]].copy()
        y = synthetic_train["Target"]
        enc = WOEEncoder(cat_cols=["province"])
        enc.fit(df, y)

        new = pd.DataFrame({"province": ["AlienProvince"]})
        result = enc.transform(new)
        assert result["province"].iloc[0] == 0.0
