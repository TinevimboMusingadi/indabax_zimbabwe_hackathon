"""Tests for ordinal encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.encoders.ordinal import OrdinalEncoder


class TestOrdinalEncoder:

    def test_fit_transform(self, synthetic_train):
        df = synthetic_train.drop(columns=["Target", "ID"])
        dates = [c for c in df.columns if df[c].dtype == "datetime64[ns]"]
        df = df.drop(columns=dates)

        enc = OrdinalEncoder()
        enc.fit(df)
        result = enc.transform(df)

        assert result.shape == df.shape

    def test_unseen_gets_minus_one(self, synthetic_train):
        df = synthetic_train.drop(columns=["Target", "ID"])
        dates = [c for c in df.columns if df[c].dtype == "datetime64[ns]"]
        df = df.drop(columns=dates)

        enc = OrdinalEncoder()
        enc.fit(df)

        new_row = df.iloc[[0]].copy()
        new_row["province"] = "NeverSeen"
        result = enc.transform(new_row)

        assert result["province"].iloc[0] == -1.0
