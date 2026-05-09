"""Date parsing utilities for the loan dataset.

Handles both D/M/YYYY (Train/Test CSV format) and DD-Mon-YYYY
(VariableDefinitions format for client_dob).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def parse_dates(
    series: pd.Series,
    dayfirst: bool = True,
) -> pd.Series:
    """Parse a date column, handling mixed formats gracefully.

    Tries D/M/YYYY first (competition CSV format), then falls back
    to ISO YYYY-MM-DD for any that fail.

    Args:
        series: Raw string date column.
        dayfirst: Whether to interpret the first number as the day.

    Returns:
        Series of pd.Timestamp with NaT for unparseable values.
    """
    original_na = series.isna().sum()

    parsed = pd.to_datetime(
        series, format="%d/%m/%Y", errors="coerce"
    )

    still_missing = parsed.isna() & series.notna()
    if still_missing.any():
        fallback = pd.to_datetime(
            series[still_missing], format="mixed",
            dayfirst=dayfirst, errors="coerce",
        )
        parsed[still_missing] = fallback

    n_failed = parsed.isna().sum() - original_na
    if n_failed > 0:
        logger.warning(
            "Column '%s': %d values could not be parsed as dates.",
            series.name,
            n_failed,
        )
    return parsed
