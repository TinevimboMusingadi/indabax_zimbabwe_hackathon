"""Tests for K-fold target encoder — including leakage check."""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.splits import make_folds
from src.features.encoders.target import KFoldTargetEncoder


class TestKFoldTargetEncoder:

    def _get_folds(self, y):
        return make_folds(
            y, n_folds=3, seed=42,
            splits_dir=tempfile.mkdtemp(), force=True,
        )

    def test_fit_transform_train(self, synthetic_train):
        df = synthetic_train.copy()
        y = df["Target"]
        folds_df = self._get_folds(y)

        cat_cols = ["province", "employment_sector"]
        enc = KFoldTargetEncoder(cat_cols=cat_cols, smoothing=10.0)
        enc.fit(df, y, folds_df)
        result = enc.transform_train(df, y, folds_df)

        for col in cat_cols:
            assert result[col].dtype == float

    def test_transform_test(self, synthetic_train, synthetic_test):
        df = synthetic_train.copy()
        y = df["Target"]
        folds_df = self._get_folds(y)

        enc = KFoldTargetEncoder(
            cat_cols=["province"], smoothing=10.0
        )
        enc.fit(df, y, folds_df)
        result = enc.transform(synthetic_test)

        assert result["province"].dtype == float
        assert not result["province"].isna().any()

    def test_leakage_protection(self, synthetic_train):
        """Mutating a held-out row's label must not change its encoding.

        If the target encoder is leak-safe, a row in fold K
        is encoded using only data from folds != K. Flipping that
        row's label should not affect the encoded value.
        """
        df = synthetic_train.copy()
        y = df["Target"].copy()
        folds_df = self._get_folds(y)

        enc = KFoldTargetEncoder(
            cat_cols=["province"], smoothing=10.0
        )
        enc.fit(df, y, folds_df)
        encoded_orig = enc.transform_train(df, y, folds_df)

        test_row = 0
        fold_of_row = folds_df.loc[test_row, "fold_id"]

        y_mutated = y.copy()
        y_mutated.iloc[test_row] = 1 - y_mutated.iloc[test_row]

        enc2 = KFoldTargetEncoder(
            cat_cols=["province"], smoothing=10.0
        )
        enc2.fit(df, y_mutated, folds_df)
        encoded_mut = enc2.transform_train(df, y_mutated, folds_df)

        np.testing.assert_almost_equal(
            encoded_orig["province"].iloc[test_row],
            encoded_mut["province"].iloc[test_row],
            decimal=5,
            err_msg=(
                "Target encoder leaked: encoding changed when "
                "the held-out row's own label was flipped."
            ),
        )
