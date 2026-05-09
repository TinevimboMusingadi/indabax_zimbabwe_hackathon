"""Full pipeline orchestrator — runs Phases 1 through 4."""

from __future__ import annotations

import logging

from src.config import PipelineConfig, load_config
from src.data.loader import TARGET_COL, load_sample_submission
from src.evaluation.metrics import evaluate, find_best_threshold
from src.pipelines.ensemble import run_ensemble
from src.pipelines.prepare_data import run_prepare_data
from src.pipelines.train_models import run_train_models
from src.pipelines.tune import run_tuning
from src.submission.writer import write_submission
from src.utils.logging_setup import setup_logging
from src.utils.seeding import seed_everything
from src.utils.timer import timer

logger = logging.getLogger(__name__)


def run(config: PipelineConfig) -> None:
    """Execute the complete pipeline end-to-end."""
    setup_logging()
    seed_everything(config.seed)

    logger.info("=" * 60)
    logger.info("IndabaX Zimbabwe 2026 — Loan Default Pipeline")
    logger.info("=" * 60)

    # Phase 1+2: Data + Features
    with timer("Phase 1-2: Data preparation"):
        data = run_prepare_data(config)

    train = data["train"]
    test = data["test"]
    folds_df = data["folds_df"]
    variants = data["variants"]

    # Phase 3a: Optional tuning
    if config.tuner.enabled:
        with timer("Phase 3a: Hyperparameter tuning"):
            best_params = run_tuning(variants, folds_df, config)

            for entry in config.training.models:
                if entry.name in best_params:
                    entry.params.update(best_params[entry.name])
                    logger.info(
                        "Updated %s params from tuner.", entry.name
                    )

    # Phase 3b: Training
    with timer("Phase 3b: Model training"):
        results = run_train_models(variants, folds_df, config)

    if not results:
        raise RuntimeError("No models were trained successfully.")

    # Phase 4a: Evaluation
    y_train = train[TARGET_COL].values

    for r in results:
        evaluate(r.model_name, y_train, r.oof_preds)

    best_threshold, best_f1 = find_best_threshold(
        y_train, results[0].oof_preds
    )

    # Phase 4b: Ensemble
    with timer("Phase 4: Ensemble"):
        final_oof, final_test = run_ensemble(
            results, y_train, config
        )

    ensemble_metrics = evaluate(
        "ensemble", y_train, final_oof, threshold=best_threshold
    )
    logger.info("Ensemble metrics: %s", ensemble_metrics)

    # Phase 4c: Submission
    with timer("Phase 4c: Submission"):
        sample_sub = load_sample_submission(config.data.raw_dir)
        from src.data.loader import ID_COL
        write_submission(
            test_ids=test[ID_COL],
            predictions=final_test,
            sample_sub=sample_sub,
            tag="final",
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)
