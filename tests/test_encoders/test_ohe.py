"""Tests for OHE encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.encoders.ohe import OHEEncoder


class TestOHEEncoder:

    def test_fit_transform(self, synthetic_train):
        df = synthetic_train.drop(columns=["Target", "ID"])
        # Drop dates for OHE
        dates = [c for c in df.columns if df[c].dtype == "datetime64[ns]"]
        df = df.drop(columns=dates)

        enc = OHEEncoder()
        enc.fit(df)
        result = enc.transform(df)

        assert result.shape[0] == len(df)
        assert not result.isnull().all(axis=1).any()

    def test_unseen_category(self, synthetic_train):
        df = synthetic_train.drop(columns=["Target", "ID"])
        dates = [c for c in df.columns if df[c].dtype == "datetime64[ns]"]
        df = df.drop(columns=dates)

        enc = OHEEncoder()
        enc.fit(df)

        new_row = df.iloc[[0]].copy()
        new_row["province"] = "UnknownProvince"
        result = enc.transform(new_row)

        assert len(result) == 1
        assert not result.isnull().all(axis=1).any()
