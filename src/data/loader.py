"""Data loading with schema validation and missing-rate audit."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW
from src.data.dates import parse_dates

logger = logging.getLogger(__name__)

DATE_COLS = [
    "date_approved",
    "date_disbursed",
    "first_payment_due",
    "maturity_date",
    "client_dob",
]

CATEGORICAL_COLS = [
    "product_code",
    "payment_frequency",
    "loan_purpose",
    "client_gender",
    "marital_status",
    "employment_sector",
    "collateral_type",
    "disbursement_channel",
    "province",
]

NUMERIC_COLS = [
    "amount_usd",
    "annual_rate_pct",
    "term_months",
    "num_dependents",
    "months_at_employer",
    "monthly_income_usd",
    "existing_obligations",
]

TARGET_COL = "Target"
ID_COL = "ID"

EXPECTED_TRAIN_COLS = (
    [ID_COL] + ["product_code"] + DATE_COLS[:4]
    + NUMERIC_COLS[:3]
    + CATEGORICAL_COLS[1:3]
    + ["client_gender", "client_dob", "marital_status"]
    + NUMERIC_COLS[3:]
    + CATEGORICAL_COLS[5:]
    + [TARGET_COL]
)


def _log_missing_rates(df: pd.DataFrame, label: str) -> None:
    """Log per-column missing rates."""
    missing = df.isnull().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        logger.info("[%s] No missing values.", label)
    else:
        logger.info("[%s] Missing rates:", label)
        for col, rate in missing.items():
            logger.info("  %-25s %.1f%%", col, rate * 100)


def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and cast categoricals."""
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = parse_dates(df[col])

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def load_train(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load and validate Train.csv."""
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    path = raw_dir / "Train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Train.csv not found at {path}. "
            "Place competition files in data/raw/."
        )
    df = pd.read_csv(path)
    logger.info("Loaded Train: %d rows, %d cols", *df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Train.csv missing target column '{TARGET_COL}'.")

    df = _apply_dtypes(df)
    _log_missing_rates(df, "Train")
    return df


def load_test(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load and validate Test.csv."""
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    path = raw_dir / "Test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Test.csv not found at {path}. "
            "Place competition files in data/raw/."
        )
    df = pd.read_csv(path)
    logger.info("Loaded Test: %d rows, %d cols", *df.shape)

    if TARGET_COL in df.columns:
        logger.warning("Test.csv contains '%s' — dropping it.", TARGET_COL)
        df = df.drop(columns=[TARGET_COL])

    df = _apply_dtypes(df)
    _log_missing_rates(df, "Test")
    return df


def load_sample_submission(
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load SampleSubmission.csv for ID ordering and format reference."""
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    path = raw_dir / "SampleSubmission.csv"
    if not path.exists():
        raise FileNotFoundError(f"SampleSubmission.csv not found at {path}.")
    df = pd.read_csv(path)
    logger.info("Loaded SampleSubmission: %d rows", len(df))
    return df
