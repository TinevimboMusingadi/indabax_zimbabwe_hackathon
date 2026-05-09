"""Tests for submission format validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.submission.writer import (
    EXPECTED_TEST_ROWS,
    _validate,
    write_submission,
)


class TestSubmissionValidation:

    def test_valid_submission(self, sample_submission):
        import src.submission.writer as writer_mod
        original = writer_mod.EXPECTED_TEST_ROWS
        writer_mod.EXPECTED_TEST_ROWS = len(sample_submission)
        try:
            _validate(sample_submission)
        finally:
            writer_mod.EXPECTED_TEST_ROWS = original

    def test_wrong_columns(self):
        bad = pd.DataFrame({"wrong": [1], "cols": [2]})
        with pytest.raises(ValueError, match="Expected columns"):
            _validate(bad)

    def test_wrong_row_count(self):
        bad = pd.DataFrame({
            "ID": ["A"],
            "Target": [0.5],
        })
        with pytest.raises(ValueError, match="Expected.*rows"):
            _validate(bad)

    def test_nan_predictions(self):
        ids = [f"X{i:05d}" for i in range(EXPECTED_TEST_ROWS)]
        preds = [0.5] * EXPECTED_TEST_ROWS
        preds[0] = np.nan
        bad = pd.DataFrame({"ID": ids, "Target": preds})
        with pytest.raises(ValueError, match="NaN predictions"):
            _validate(bad)


class TestWriteSubmission:

    def test_writes_valid_csv(self, synthetic_test, sample_submission):
        preds = np.random.random(len(synthetic_test))
        out_dir = tempfile.mkdtemp()

        # Patch expected rows for synthetic data
        import src.submission.writer as writer_mod
        original = writer_mod.EXPECTED_TEST_ROWS
        writer_mod.EXPECTED_TEST_ROWS = len(synthetic_test)
        try:
            path = write_submission(
                test_ids=synthetic_test["ID"],
                predictions=preds,
                sample_sub=sample_submission,
                tag="test",
                submissions_dir=out_dir,
            )
            assert path.exists()
            result = pd.read_csv(path)
            assert list(result.columns) == ["ID", "Target"]
            assert len(result) == len(synthetic_test)
            assert not result["Target"].isna().any()
        finally:
            writer_mod.EXPECTED_TEST_ROWS = original

    def test_id_order_matches_sample(
        self, synthetic_test, sample_submission
    ):
        preds = np.random.random(len(synthetic_test))
        out_dir = tempfile.mkdtemp()

        import src.submission.writer as writer_mod
        original = writer_mod.EXPECTED_TEST_ROWS
        writer_mod.EXPECTED_TEST_ROWS = len(synthetic_test)
        try:
            path = write_submission(
                test_ids=synthetic_test["ID"],
                predictions=preds,
                sample_sub=sample_submission,
                tag="test_order",
                submissions_dir=out_dir,
            )
            result = pd.read_csv(path)
            np.testing.assert_array_equal(
                result["ID"].values,
                sample_submission["ID"].values,
            )
        finally:
            writer_mod.EXPECTED_TEST_ROWS = original
