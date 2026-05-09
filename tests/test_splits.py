"""Tests for stratified k-fold splitting."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.splits import get_fold_indices, make_folds


class TestMakeFolds:

    def test_fold_count(self, synthetic_train):
        y = synthetic_train["Target"]
        folds = make_folds(
            y, n_folds=5, seed=42,
            splits_dir=tempfile.mkdtemp(), force=True,
        )
        assert folds["fold_id"].nunique() == 5

    def test_all_rows_assigned(self, synthetic_train):
        y = synthetic_train["Target"]
        folds = make_folds(
            y, n_folds=5, seed=42,
            splits_dir=tempfile.mkdtemp(), force=True,
        )
        assert len(folds) == len(y)

    def test_disjoint_folds(self, synthetic_train):
        y = synthetic_train["Target"]
        folds = make_folds(
            y, n_folds=3, seed=42,
            splits_dir=tempfile.mkdtemp(), force=True,
        )
        for fold_id in range(3):
            tr_idx, val_idx = get_fold_indices(folds, fold_id)
            assert len(set(tr_idx) & set(val_idx)) == 0

    def test_stratification_preserved(self, synthetic_train):
        y = synthetic_train["Target"]
        folds = make_folds(
            y, n_folds=5, seed=42,
            splits_dir=tempfile.mkdtemp(), force=True,
        )
        overall_rate = y.mean()
        for fold_id in range(5):
            mask = folds["fold_id"] == fold_id
            fold_rate = y.iloc[mask.values].mean()
            assert abs(fold_rate - overall_rate) < 0.05

    def test_determinism(self, synthetic_train):
        y = synthetic_train["Target"]
        d1 = tempfile.mkdtemp()
        d2 = tempfile.mkdtemp()
        f1 = make_folds(y, n_folds=5, seed=42, splits_dir=d1, force=True)
        f2 = make_folds(y, n_folds=5, seed=42, splits_dir=d2, force=True)
        pd.testing.assert_frame_equal(f1, f2)

    def test_persistence(self, synthetic_train):
        y = synthetic_train["Target"]
        d = tempfile.mkdtemp()
        f1 = make_folds(y, n_folds=3, seed=42, splits_dir=d, force=True)
        f2 = make_folds(y, n_folds=3, seed=42, splits_dir=d, force=False)
        pd.testing.assert_frame_equal(f1, f2)
