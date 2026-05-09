"""Pipeline: data loading, splitting, and feature engineering."""

from __future__ import annotations

import logging

from src.config import PipelineConfig
from src.data.loader import load_test, load_train
from src.data.splits import make_folds
from src.features.adversarial import adversarial_validation
from src.features.pipeline import build_all_variants
from src.utils.timer import timer

logger = logging.getLogger(__name__)


def run_prepare_data(config: PipelineConfig) -> dict:
    """Execute the full data preparation pipeline.

    Returns:
        Dict with keys: train, test, folds_df, variants.
    """
    with timer("Loading data"):
        train = load_train(config.data.raw_dir)
        test = load_test(config.data.raw_dir)

    with timer("Creating CV folds"):
        from src.data.loader import TARGET_COL
        folds_df = make_folds(
            train[TARGET_COL],
            n_folds=config.data.n_folds,
            seed=config.seed,
            splits_dir=config.data.splits_dir,
        )

    if config.features.run_adversarial:
        with timer("Adversarial validation"):
            adversarial_validation(train, test)

    with timer("Building feature variants"):
        variants = build_all_variants(train, test, folds_df, config)

    return {
        "train": train,
        "test": test,
        "folds_df": folds_df,
        "variants": variants,
    }
