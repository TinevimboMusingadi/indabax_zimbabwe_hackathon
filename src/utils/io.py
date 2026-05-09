"""I/O helpers for parquet and JSON artefacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame to parquet, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved parquet: %s (%d rows)", path, len(df))
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded parquet: %s (%d rows)", path, len(df))
    return df


def save_json(data: dict, path: str | Path) -> Path:
    """Save a dict as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path
