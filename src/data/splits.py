"""Stratified K-fold splitting with persistence."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import DATA_SPLITS, SEED
from src.utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)


def make_folds(
    y: pd.Series,
    n_folds: int = 5,
    seed: int = SEED,
    splits_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Create and persist stratified fold assignments.

    Args:
        y: Target series (index must align with training data).
        n_folds: Number of folds.
        seed: Random seed for reproducibility.
        splits_dir: Directory to save/load fold file.
        force: If True, regenerate even if file exists.

    Returns:
        DataFrame with columns ['row_idx', 'fold_id'].
    """
    splits_dir = Path(splits_dir) if splits_dir else DATA_SPLITS
    folds_path = splits_dir / "folds.parquet"

    if folds_path.exists() and not force:
        logger.info("Loading existing folds from %s", folds_path)
        return load_parquet(folds_path)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_ids = np.zeros(len(y), dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_ids[val_idx] = fold_idx

    folds_df = pd.DataFrame({
        "row_idx": np.arange(len(y)),
        "fold_id": fold_ids,
    })

    _validate_folds(folds_df, y, n_folds)
    save_parquet(folds_df, folds_path)
    logger.info(
        "Created %d-fold split for %d samples.", n_folds, len(y)
    )
    return folds_df


def _validate_folds(
    folds_df: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
) -> None:
    """Validate fold disjointness and approximate stratification."""
    assert len(folds_df) == len(y), "Fold count != target count."
    assert folds_df["fold_id"].nunique() == n_folds, "Wrong fold count."

    overall_rate = y.mean()
    for fold_id in range(n_folds):
        mask = folds_df["fold_id"] == fold_id
        fold_rate = y.iloc[mask.values].mean()
        diff = abs(fold_rate - overall_rate)
        if diff > 0.02:
            logger.warning(
                "Fold %d target rate %.3f differs from overall %.3f "
                "by %.3f (>2%%)",
                fold_id, fold_rate, overall_rate, diff,
            )


def get_fold_indices(
    folds_df: pd.DataFrame,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, val_indices) for a given fold."""
    val_mask = folds_df["fold_id"] == fold_id
    val_idx = folds_df.loc[val_mask, "row_idx"].values
    train_idx = folds_df.loc[~val_mask, "row_idx"].values
    return train_idx, val_idx
