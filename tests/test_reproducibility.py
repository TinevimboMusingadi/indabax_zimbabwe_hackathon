"""Tests for seed-based reproducibility."""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.splits import make_folds
from src.features.base import BaseFeatureEngineer
from src.utils.seeding import seed_everything


class TestReproducibility:

    def test_same_seed_same_folds(self, synthetic_train):
        y = synthetic_train["Target"]
        d1 = tempfile.mkdtemp()
        d2 = tempfile.mkdtemp()

        seed_everything(42)
        f1 = make_folds(y, n_folds=5, seed=42, splits_dir=d1, force=True)

        seed_everything(42)
        f2 = make_folds(y, n_folds=5, seed=42, splits_dir=d2, force=True)

        pd.testing.assert_frame_equal(f1, f2)

    def test_feature_engineering_deterministic(self, synthetic_train):
        seed_everything(42)
        eng1 = BaseFeatureEngineer()
        r1 = eng1.fit_transform(synthetic_train.copy())

        seed_everything(42)
        eng2 = BaseFeatureEngineer()
        r2 = eng2.fit_transform(synthetic_train.copy())

        num_cols = r1.select_dtypes(include=["number"]).columns
        for col in num_cols:
            np.testing.assert_array_almost_equal(
                r1[col].values, r2[col].values,
                err_msg=f"Non-deterministic column: {col}",
            )
