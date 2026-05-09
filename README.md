# IndabaX Zimbabwe 2026 — Loan Default Prediction

End-to-end ML pipeline for the Deep Learning IndabaX Zimbabwe 2026 hackathon:
*AI for Financial Inclusion — Predicting Loan Defaults in Zimbabwe's Banking
Sector*.

**Primary metric:** ROC-AUC | **Submission:** `ID,Target` (probabilities)

## Quick Start — Google Colab (recommended)

1. Upload `Train.csv`, `Test.csv`, `SampleSubmission.csv` to
   `Google Drive > My Drive > indabax_data/`.
2. Open `notebooks/01_train_predict_submit.ipynb` in Colab.
3. Set runtime to **GPU (T4)**.
4. Run all cells. A submission CSV will be downloaded automatically.

## Quick Start — Local

```bash
git clone https://github.com/TinevimboMusingadi/indabax_zimbabwe_hackathon.git
cd indabax_zimbabwe_hackathon
pip install -e ".[dev]"

# Place Train.csv, Test.csv, SampleSubmission.csv into data/raw/
python -m src.pipelines.run_full --config configs/fast.yaml
```

## Configs

| Config | Models | Optuna trials | ~Runtime | Environment |
|--------|--------|---------------|----------|-------------|
| `fast.yaml` | LGBM, XGB | 0 (defaults) | ~5 min | CPU |
| `colab_t4.yaml` | LGBM, XGB, CatBoost, TabNet, MLP | 50 | ~30 min | T4 GPU |
| `full.yaml` | All models | 100 | ~90 min | GPU |

## Pipeline Phases

1. **Data loading** — schema validation, date parsing, missing-rate audit.
2. **Splitting** — 5-fold stratified CV, persisted to `data/splits/`.
3. **Feature engineering** — date deltas, ratios, interactions, cyclical
   features. Three encoding variants (OHE, ordinal, target+WOE).
4. **Training** — classical ML + gradient boosting + deep learning with
   fold-level OOF predictions.
5. **Tuning** — Optuna TPE per model (optional, controlled by config).
6. **Ensemble** — rank averaging, stacking, Optuna blend weights.
7. **Submission** — validates format against `SampleSubmission.csv`.

## Reproducibility

- **Global seed:** 42 (threaded through Python, NumPy, PyTorch, sklearn).
- **Fold IDs** persisted in `data/splits/folds.parquet`.
- **Requirements file** with all packages listed in `requirements.txt`.
- All data editing done in code — no manual Excel modifications.

## Project Structure

```
src/
  config.py              # Pydantic config, loads YAML
  data/                  # Loading, date parsing, CV splits
  features/              # Base FE + encoder variants
    encoders/            # OHE, ordinal, frequency, target, WOE, group stats
  models/                # BaseModel ABC + all model wrappers
  training/              # CV trainer, Optuna tuner, calibration
  evaluation/            # Metrics, SHAP explanations
  ensemble/              # Rank avg, stacking, Optuna blend
  submission/            # Format-validated submission writer
  pipelines/             # Orchestrators (prepare_data, train, run_full)
  utils/                 # Seeding, timing, Colab helpers
notebooks/               # Single end-to-end Colab notebook
configs/                 # YAML configs per environment
tests/                   # pytest suite with synthetic fixtures
```

## Team

Built for Deep Learning IndabaX Zimbabwe 2026 — AI for Financial Inclusion.
