"""Feature pipeline — produces encoding variants and saves to parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, PipelineConfig
from src.data.loader import ID_COL, TARGET_COL
from src.features.base import BaseFeatureEngineer, get_feature_columns
from src.features.encoders.frequency import FrequencyEncoder
from src.features.encoders.group_stats import GroupStatsEncoder
from src.features.encoders.ohe import OHEEncoder
from src.features.encoders.ordinal import OrdinalEncoder
from src.features.encoders.target import KFoldTargetEncoder
from src.features.encoders.woe import WOEEncoder
from src.utils.io import save_parquet
from src.utils.timer import timer

logger = logging.getLogger(__name__)


def build_all_variants(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds_df: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build requested feature variants, save to parquet, return dict.

    Returns:
        Mapping of variant name to (train_df, test_df) with engineered
        features. Target column is preserved in train_df.
    """
    base_eng = BaseFeatureEngineer()
    y_train = train[TARGET_COL].copy()
    train_ids = train[ID_COL].copy()
    test_ids = test[ID_COL].copy()

    with timer("Base feature engineering"):
        train_fe = base_eng.fit_transform(train)
        test_fe = base_eng.transform(test)

    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    requested = config.features.variants

    if "v1_ohe" in requested:
        with timer("v1_ohe encoding"):
            tr, te = _build_v1_ohe(train_fe, test_fe, y_train)
            tr[TARGET_COL] = y_train.values
            tr[ID_COL] = train_ids.values
            te[ID_COL] = test_ids.values
            _save_variant("v1_ohe", tr, te, config)
            variants["v1_ohe"] = (tr, te)

    if "v2_ordinal" in requested:
        with timer("v2_ordinal encoding"):
            tr, te = _build_v2_ordinal(train_fe, test_fe, y_train)
            tr[TARGET_COL] = y_train.values
            tr[ID_COL] = train_ids.values
            te[ID_COL] = test_ids.values
            _save_variant("v2_ordinal", tr, te, config)
            variants["v2_ordinal"] = (tr, te)

    if "v3_target_woe" in requested:
        with timer("v3_target_woe encoding"):
            tr, te = _build_v3_target_woe(
                train_fe, test_fe, y_train, folds_df, config
            )
            tr[TARGET_COL] = y_train.values
            tr[ID_COL] = train_ids.values
            te[ID_COL] = test_ids.values
            _save_variant("v3_target_woe", tr, te, config)
            variants["v3_target_woe"] = (tr, te)

    return variants


def _build_v1_ohe(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OHE + frequency encoding."""
    freq = FrequencyEncoder()
    freq.fit(train_fe, y)
    freq_cats = freq.cat_cols or []

    ohe = OHEEncoder(cat_cols=freq_cats)
    ohe.fit(train_fe, y)

    tr = ohe.transform(train_fe)
    te = ohe.transform(test_fe)
    return tr, te


def _build_v2_ordinal(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ordinal encoding for tree-based models."""
    enc = OrdinalEncoder()
    enc.fit(train_fe, y)
    return enc.transform(train_fe), enc.transform(test_fe)


def _build_v3_target_woe(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    y: pd.Series,
    folds_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target encoding + WOE + group stats.

    WOE and group-stats are computed on the original categorical columns
    and added as NEW columns (suffixed) alongside the target-encoded ones.
    """
    smoothing = config.features.target_encoder_smoothing
    cat_cols = [
        c for c in train_fe.columns
        if train_fe[c].dtype.name in ("category", "object")
    ]

    # WOE: compute on ORIGINAL categoricals, store as separate columns
    woe = WOEEncoder(cat_cols=cat_cols)
    woe.fit(train_fe, y)
    tr_woe = woe.transform(train_fe[cat_cols].copy())
    te_woe = woe.transform(test_fe[cat_cols].copy())
    tr_woe = tr_woe.rename(columns={c: f"{c}_woe" for c in cat_cols})
    te_woe = te_woe.rename(columns={c: f"{c}_woe" for c in cat_cols})

    # Group stats: compute on ORIGINAL categoricals
    gs = GroupStatsEncoder()
    gs.fit(train_fe, y)
    tr_gs = gs.transform(train_fe[["province", "employment_sector"]].copy())
    te_gs = gs.transform(test_fe[["province", "employment_sector"]].copy())
    gs_new_cols = [
        c for c in tr_gs.columns
        if c not in ("province", "employment_sector")
    ]

    # Target encoding: replaces categoricals in-place with target means
    te_enc = KFoldTargetEncoder(smoothing=smoothing)
    te_enc.fit(train_fe, y, folds_df)
    tr_encoded = te_enc.transform_train(train_fe, y, folds_df)
    te_encoded = te_enc.transform(test_fe)

    # Merge WOE and group-stat columns alongside target-encoded data
    for col in tr_woe.columns:
        tr_encoded[col] = tr_woe[col].values
        te_encoded[col] = te_woe[col].values

    for col in gs_new_cols:
        tr_encoded[col] = tr_gs[col].values
        te_encoded[col] = te_gs[col].values

    return tr_encoded, te_encoded


def _save_variant(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    out_dir = Path(config.data.processed_dir) / name
    save_parquet(train_df, out_dir / "train.parquet")
    save_parquet(test_df, out_dir / "test.parquet")
    logger.info(
        "Variant '%s': train=%d cols, test=%d cols",
        name, train_df.shape[1], test_df.shape[1],
    )
