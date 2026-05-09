"""Submission CSV writer with strict format validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SUBMISSIONS_DIR
from src.data.loader import ID_COL, TARGET_COL

logger = logging.getLogger(__name__)

EXPECTED_TEST_ROWS = 12_977


def write_submission(
    test_ids: pd.Series | np.ndarray,
    predictions: np.ndarray,
    sample_sub: pd.DataFrame,
    tag: str = "final",
    submissions_dir: str | Path | None = None,
) -> Path:
    """Create a validated submission CSV.

    Args:
        test_ids: ID values for each test row.
        predictions: Predicted probabilities.
        sample_sub: Reference SampleSubmission DataFrame for ordering.
        tag: Filename tag.
        submissions_dir: Output directory.

    Returns:
        Path to the written CSV.

    Raises:
        ValueError: If the submission fails validation.
    """
    submissions_dir = (
        Path(submissions_dir) if submissions_dir else SUBMISSIONS_DIR
    )
    submissions_dir.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame({
        ID_COL: np.asarray(test_ids),
        TARGET_COL: np.asarray(predictions, dtype=np.float64),
    })

    sub = _align_to_sample(sub, sample_sub)
    _validate(sub)

    filename = f"sub_{tag}.csv"
    path = submissions_dir / filename
    sub.to_csv(path, index=False)
    logger.info("Submission saved: %s (%d rows)", path, len(sub))

    _update_leaderboard(tag, path)
    return path


def _align_to_sample(
    sub: pd.DataFrame,
    sample_sub: pd.DataFrame,
) -> pd.DataFrame:
    """Reorder submission to match SampleSubmission ID order."""
    expected_ids = sample_sub[ID_COL].values
    sub = sub.set_index(ID_COL).loc[expected_ids].reset_index()
    return sub


def _validate(sub: pd.DataFrame) -> None:
    """Strict format validation."""
    if list(sub.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"Expected columns {[ID_COL, TARGET_COL]}, "
            f"got {list(sub.columns)}"
        )

    if len(sub) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS} rows, got {len(sub)}"
        )

    if sub[TARGET_COL].isna().any():
        raise ValueError("Submission contains NaN predictions.")

    if sub[ID_COL].isna().any():
        raise ValueError("Submission contains NaN IDs.")

    logger.info("Submission validation passed.")


def _update_leaderboard(tag: str, path: Path) -> None:
    """Append entry to results/leaderboard.md."""
    from src.config import RESULTS_DIR

    lb_path = RESULTS_DIR / "leaderboard.md"
    lb_path.parent.mkdir(parents=True, exist_ok=True)

    if not lb_path.exists():
        lb_path.write_text(
            "| Tag | File | Rows |\n"
            "|-----|------|------|\n",
            encoding="utf-8",
        )

    with open(lb_path, "a", encoding="utf-8") as f:
        f.write(f"| {tag} | {path.name} | {EXPECTED_TEST_ROWS} |\n")
