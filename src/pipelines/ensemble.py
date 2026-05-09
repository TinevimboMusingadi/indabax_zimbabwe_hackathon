"""Pipeline: ensemble OOF and test predictions."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import PipelineConfig
from src.ensemble.optuna_blend import optuna_blend
from src.ensemble.rank_avg import rank_average
from src.ensemble.stacking import stack_predictions
from src.training.cv_trainer import CvResult
from src.utils.timer import timer

logger = logging.getLogger(__name__)


def run_ensemble(
    results: list[CvResult],
    y_train: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine predictions from multiple models.

    Returns:
        (best_oof_preds, best_test_preds) from the best ensemble.
    """
    if len(results) == 1:
        logger.info("Only one model — using its predictions directly.")
        return results[0].oof_preds, results[0].test_preds

    oof_list = [r.oof_preds for r in results]
    test_list = [r.test_preds for r in results]
    methods = config.ensemble.methods

    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if "rank_avg" in methods:
        with timer("Rank average ensemble"):
            ra_oof = rank_average(oof_list)
            ra_test = rank_average(test_list)
            auc = roc_auc_score(y_train, ra_oof)
            logger.info("Rank avg OOF AUC: %.4f", auc)
            candidates["rank_avg"] = (ra_oof, ra_test)

    if "stacking" in methods:
        with timer("Stacking ensemble"):
            st_oof, st_test = stack_predictions(
                oof_list, y_train, test_list
            )
            auc = roc_auc_score(y_train, st_oof)
            logger.info("Stacking OOF AUC: %.4f", auc)
            candidates["stacking"] = (st_oof, st_test)

    if "optuna_blend" in methods:
        n_trials = config.ensemble.optuna_blend_trials
        if n_trials > 0:
            with timer("Optuna blend ensemble"):
                ob_oof, ob_test, _ = optuna_blend(
                    oof_list, y_train, test_list,
                    n_trials=n_trials, seed=config.seed,
                )
                auc = roc_auc_score(y_train, ob_oof)
                logger.info("Optuna blend OOF AUC: %.4f", auc)
                candidates["optuna_blend"] = (ob_oof, ob_test)

    # Pick best by OOF AUC
    best_name = ""
    best_auc = -1.0
    best_preds = (results[0].oof_preds, results[0].test_preds)

    for name, (oof, test) in candidates.items():
        auc = roc_auc_score(y_train, oof)
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_preds = (oof, test)

    logger.info(
        "Best ensemble: %s (OOF AUC: %.4f)", best_name, best_auc
    )
    return best_preds
