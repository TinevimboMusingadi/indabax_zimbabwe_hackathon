"""Pipeline configuration backed by Pydantic and YAML."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
LOGS_DIR = PROJECT_ROOT / "logs"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
RESULTS_DIR = PROJECT_ROOT / "results"
OPTUNA_DIR = PROJECT_ROOT / "optuna_studies"

SEED = 42


class DataConfig(BaseModel):
    """Settings for data loading and splitting."""

    raw_dir: str = str(DATA_RAW)
    processed_dir: str = str(DATA_PROCESSED)
    splits_dir: str = str(DATA_SPLITS)
    n_folds: int = 5
    seed: int = SEED


class FeatureConfig(BaseModel):
    """Settings for feature engineering."""

    variants: list[str] = Field(
        default=["v1_ohe", "v2_ordinal", "v3_target_woe"]
    )
    target_encoder_smoothing: float = 10.0
    run_adversarial: bool = True


class ModelEntry(BaseModel):
    """A single model specification."""

    name: str
    variant: str = "v2_ordinal"
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Settings for model training."""

    models: list[ModelEntry] = Field(default_factory=list)
    use_gpu: bool = True
    seed: int = SEED


class TunerConfig(BaseModel):
    """Settings for Optuna hyperparameter search."""

    enabled: bool = False
    n_trials: int = 50
    models_to_tune: list[str] = Field(
        default=["lgbm", "xgb", "catboost"]
    )


class EnsembleConfig(BaseModel):
    """Settings for ensemble strategy."""

    methods: list[str] = Field(
        default=["rank_avg", "stacking", "optuna_blend"]
    )
    optuna_blend_trials: int = 200


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    seed: int = SEED
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tuner: TunerConfig = Field(default_factory=TunerConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    logger.info("Loaded config from %s", path)
    return PipelineConfig(**raw)
