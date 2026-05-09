"""Pipeline: train all configured models via CV."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import PipelineConfig
from src.training.cv_trainer import CvResult, cv_train
from src.utils.timer import timer

logger = logging.getLogger(__name__)


def run_train_models(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    folds_df: pd.DataFrame,
    config: PipelineConfig,
) -> list[CvResult]:
    """Train every model in config across its assigned variant.

    Returns:
        List of CvResult objects sorted by mean_auc descending.
    """
    results: list[CvResult] = []

    for entry in config.training.models:
        variant_name = entry.variant
        if variant_name not in variants:
            logger.warning(
                "Variant '%s' for model '%s' not built — skipping.",
                variant_name, entry.name,
            )
            continue

        train_df, test_df = variants[variant_name]

        with timer(f"Training {entry.name} on {variant_name}"):
            result = cv_train(
                model_name=entry.name,
                model_params=entry.params,
                train_df=train_df,
                test_df=test_df,
                folds_df=folds_df,
                variant=variant_name,
                use_gpu=config.training.use_gpu,
            )
            results.append(result)

    results.sort(key=lambda r: r.mean_auc, reverse=True)

    logger.info("=== Model Ranking ===")
    for i, r in enumerate(results):
        logger.info(
            "  %d. %s (%s) — AUC: %.4f +/- %.4f",
            i + 1, r.model_name, r.variant,
            r.mean_auc, r.std_auc,
        )

    return results
