"""Integration test: synthetic data through the feature pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.data.splits import make_folds
from src.features.base import BaseFeatureEngineer
from src.features.pipeline import build_all_variants


class TestPipelineIntegration:

    def test_build_all_variants(self, synthetic_train, synthetic_test):
        y = synthetic_train["Target"]
        tmp = tempfile.mkdtemp()

        folds_df = make_folds(
            y, n_folds=3, seed=42, splits_dir=tmp, force=True
        )

        config = PipelineConfig(
            data={"processed_dir": tmp},
            features={"variants": ["v1_ohe", "v2_ordinal", "v3_target_woe"]},
        )

        variants = build_all_variants(
            synthetic_train, synthetic_test, folds_df, config
        )

        assert "v1_ohe" in variants
        assert "v2_ordinal" in variants
        assert "v3_target_woe" in variants

        for name, (tr, te) in variants.items():
            assert len(tr) == len(synthetic_train), f"{name} train size"
            assert len(te) == len(synthetic_test), f"{name} test size"
            assert "Target" in tr.columns, f"{name} missing Target"
            assert "ID" in tr.columns, f"{name} missing ID"
            assert "Target" not in te.columns, f"{name} test has Target"

    def test_ordinal_no_nan_in_cats(
        self, synthetic_train, synthetic_test
    ):
        y = synthetic_train["Target"]
        tmp = tempfile.mkdtemp()
        folds_df = make_folds(
            y, n_folds=3, seed=42, splits_dir=tmp, force=True
        )

        config = PipelineConfig(
            data={"processed_dir": tmp},
            features={"variants": ["v2_ordinal"]},
        )

        variants = build_all_variants(
            synthetic_train, synthetic_test, folds_df, config
        )
        tr, te = variants["v2_ordinal"]

        cat_cols = [
            c for c in tr.columns
            if tr[c].dtype in ("int32", "int64")
            and c not in ("Target", "existing_obligations")
        ]
        for col in cat_cols:
            assert not tr[col].isna().any(), (
                f"NaN in train col {col}"
            )
