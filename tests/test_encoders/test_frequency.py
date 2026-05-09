"""Tests for frequency encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.encoders.frequency import FrequencyEncoder


class TestFrequencyEncoder:

    def test_frequencies_are_valid(self, synthetic_train):
        df = synthetic_train[["province"]].copy()
        enc = FrequencyEncoder(cat_cols=["province"])
        enc.fit(df)
        result = enc.transform(df)
        assert (result["province"] >= 0).all()
        assert (result["province"] <= 1).all()

    def test_unseen_gets_zero(self, synthetic_train):
        df = synthetic_train[["province"]].copy()
        enc = FrequencyEncoder(cat_cols=["province"])
        enc.fit(df)

        new = pd.DataFrame({"province": ["NeverSeenBefore"]})
        result = enc.transform(new)
        assert result["province"].iloc[0] == 0.0
