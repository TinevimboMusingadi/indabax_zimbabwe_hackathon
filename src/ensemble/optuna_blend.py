"""Optuna-optimised blend weights for ensemble."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def optuna_blend(
    oof_list: list[np.ndarray],
    y_train: np.ndarray,
    test_list: list[np.ndarray],
    n_trials: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Search for optimal blend weights using Optuna.

    Returns:
        (blended_oof, blended_test, weights).
    """
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_models = len(oof_list)

    def objective(trial: optuna.Trial) -> float:
        raw = [
            trial.suggest_float(f"w{i}", 0.0, 1.0)
            for i in range(n_models)
        ]
        total = sum(raw)
        if total < 1e-8:
            return 0.5
        weights = [w / total for w in raw]
        blend = sum(
            w * p for w, p in zip(weights, oof_list)
        )
        return roc_auc_score(y_train, blend)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    raw = [study.best_params[f"w{i}"] for i in range(n_models)]
    total = sum(raw)
    weights = [w / total for w in raw]

    blended_oof = sum(w * p for w, p in zip(weights, oof_list))
    blended_test = sum(w * p for w, p in zip(weights, test_list))

    logger.info("Optuna blend weights: %s", weights)
    logger.info(
        "Optuna blend OOF AUC: %.4f",
        roc_auc_score(y_train, blended_oof),
    )

    return blended_oof, blended_test, weights
