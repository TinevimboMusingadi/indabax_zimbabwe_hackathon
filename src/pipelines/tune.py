"""Pipeline: Optuna hyperparameter tuning for selected models."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import PipelineConfig
from src.training.tuner import tune_model
from src.utils.timer import timer

logger = logging.getLogger(__name__)


def run_tuning(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    folds_df: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, dict]:
    """Tune each model in config.tuner.models_to_tune.

    Returns:
        Dict of model_name -> best_params.
    """
    if not config.tuner.enabled:
        logger.info("Tuning disabled in config — skipping.")
        return {}

    best_params: dict[str, dict] = {}

    for model_name in config.tuner.models_to_tune:
        matching = [
            e for e in config.training.models if e.name == model_name
        ]
        if not matching:
            logger.warning(
                "Model '%s' in tuner list but not in training config.",
                model_name,
            )
            continue

        variant_name = matching[0].variant
        if variant_name not in variants:
            logger.warning(
                "Variant '%s' not available for tuning '%s'.",
                variant_name, model_name,
            )
            continue

        train_df, _ = variants[variant_name]

        with timer(f"Tuning {model_name}"):
            bp = tune_model(model_name, train_df, folds_df, config)
            best_params[model_name] = bp

    return best_params
