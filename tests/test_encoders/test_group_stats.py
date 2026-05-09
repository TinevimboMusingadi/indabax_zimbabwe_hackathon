"""Tests for group stats encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.encoders.group_stats import GroupStatsEncoder


class TestGroupStatsEncoder:

    def test_adds_stat_columns(self, synthetic_train):
        df = synthetic_train.copy()
        y = df["Target"]
        enc = GroupStatsEncoder(group_cols=["province"])
        enc.fit(df, y)
        result = enc.transform(df)

        assert "province_default_rate" in result.columns
        assert "province_loan_count" in result.columns

    def test_rates_are_valid(self, synthetic_train):
        df = synthetic_train.copy()
        y = df["Target"]
        enc = GroupStatsEncoder(group_cols=["province"])
        enc.fit(df, y)
        result = enc.transform(df)

        rates = result["province_default_rate"].dropna()
        assert (rates >= 0).all()
        assert (rates <= 1).all()
