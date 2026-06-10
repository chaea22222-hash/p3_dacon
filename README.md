# DACON 5th ETRI Human Understanding AI Baseline

This repository contains a first-submission baseline pipeline for the DACON 5th ETRI Human Understanding AI competition.

The baseline prioritizes a valid, reproducible submission format over leaderboard score. The current version is a lightly improved baseline for team sharing. It:

- inspects the raw CSV/parquet schemas,
- derives the prediction key as `subject_id + lifelog_date`,
- aggregates every parquet item file by daily subject key,
- expands selected nested/list sensor columns into safe daily summaries,
- adds simple `lifelog_date` features,
- trains one target model per label column,
- preserves the sample submission column order exactly.

## Feature Summary

- Numeric columns: daily `mean`, `std`, `min`, `max`, `median`, `count`
- `heart_rate`: list length and numeric heart-rate summaries
- `m_gps`: fast GPS edge summaries such as first speed/altitude/lat/lon and last speed
- `m_wifi`, `m_ble`: detected count and RSSI summaries
- `m_usage_stats`: app count and total-time summaries
- `m_ambience`: list length and top sound score summary
- Date features: day, day of week, weekend flag, month, day index

For binary targets, the training script uses `LightGBMClassifier` when LightGBM is available. Otherwise it falls back to sklearn models.

## Setup

This project uses Python 3.12.

Install uv if not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

## Run

Inspect the dataset:

```bash
uv run python src/inspect_data.py
```

Build features only:

```bash
uv run python src/make_features.py
```

Train models and create the submission from existing features:

```bash
uv run python src/train_baseline.py
```

Run the full pipeline:

```bash
uv run python src/run_first_submission.py
```

## Current Team Baseline: R3

The current team baseline is `r3`, because its Public leaderboard result was
better than the later r4 sensor-rule adjustment.

```bash
python src/train_r3.py
```

See `R3_TEAM_SHARE.md` for the exact preprocessing, interpolation parameters,
validation caveat, and team experiment rules.

## Outputs

The full run writes:

- `outputs/features_train.csv`
- `outputs/features_test.csv`
- `outputs/submission_baseline.csv`
- `outputs/validation_scores.csv`

`outputs/submission_baseline.csv` has exactly the same columns and row order as `data/raw/ch2026_submission_sample.csv`.
