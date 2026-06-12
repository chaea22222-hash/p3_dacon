"""
Reproducible r3 baseline.

r3 uses each subject's target mean as the anchor prediction and blends selected
targets with a temporally weighted average of nearby training labels.

Public leaderboard reference:
- r2: about 0.607
- r3: about 0.604
- r4 sensor-rule adjustment: 0.6060468821, so r3 remains the baseline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import log_loss

load_dotenv()

from pipeline_utils import (
    DATE_COL,
    OUTPUT_DIR,
    SUBJECT_COL,
    ensure_dirs,
    infer_target_columns,
    load_train_sample,
)

SUBMISSION_PATH = OUTPUT_DIR / "submission_r3.csv"
SCORES_PATH = OUTPUT_DIR / "validation_scores_r3.csv"
PROB_CLIP = 1e-6

# alpha is the subject-mean weight. The remaining weight is assigned to the
# temporally local label average.
TARGET_PARAMS: Dict[str, Tuple[int, float, float]] = {
    "Q1": (7, 2.0, 0.7),
    "Q2": (7, 2.0, 0.5),
    "Q3": (14, 2.0, 0.5),
    "S2": (21, 7.0, 0.7),
}


def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    return out


def _subject_means(train: pd.DataFrame, target: str) -> pd.Series:
    return train.groupby(SUBJECT_COL)[target].mean()


def _local_prediction(
    train_subject: pd.DataFrame,
    prediction_date: pd.Timestamp,
    target: str,
    window_days: int,
    decay: float,
    fallback: float,
    exclude_index=None,
) -> float:
    candidates = train_subject
    if exclude_index is not None:
        candidates = candidates[candidates.index != exclude_index]

    distances = (candidates[DATE_COL] - prediction_date).dt.days.abs()
    within = distances <= window_days
    candidates = candidates.loc[within]
    distances = distances.loc[within]

    if candidates.empty:
        return fallback

    weights = np.exp(-distances.to_numpy(dtype=float) / decay)
    return float(np.average(candidates[target].to_numpy(dtype=float), weights=weights))


def predict_target(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    target: str,
    exclude_self: bool = False,
) -> np.ndarray:
    means = _subject_means(train, target)
    grouped = {subject: group for subject, group in train.groupby(SUBJECT_COL)}
    params = TARGET_PARAMS.get(target)
    predictions = []

    for index, row in rows.iterrows():
        subject = row[SUBJECT_COL]
        subject_mean = float(means.loc[subject])

        if params is None:
            prediction = subject_mean
        else:
            window_days, decay, alpha = params
            local = _local_prediction(
                grouped[subject],
                row[DATE_COL],
                target,
                window_days,
                decay,
                fallback=subject_mean,
                exclude_index=index if exclude_self else None,
            )
            prediction = alpha * subject_mean + (1.0 - alpha) * local

        predictions.append(np.clip(prediction, PROB_CLIP, 1.0 - PROB_CLIP))

    return np.asarray(predictions)


def train_and_predict() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    mlflow.set_experiment("r3")
    train, sample = load_train_sample()
    target_cols = infer_target_columns(train, sample)
    train = _prepare_dates(train)
    sample_dates = _prepare_dates(sample)

    submission = sample.copy()
    scores = []

    with mlflow.start_run():
        flat_params = {"prob_clip": PROB_CLIP}
        for t, (window, decay, alpha) in TARGET_PARAMS.items():
            flat_params[f"{t}_window"] = window
            flat_params[f"{t}_decay"] = decay
            flat_params[f"{t}_alpha"] = alpha
        mlflow.log_params(flat_params)

        print(f"r3 target parameters: {TARGET_PARAMS}")
        for target in target_cols:
            oof_pred = predict_target(train, train, target, exclude_self=True)
            test_pred = predict_target(train, sample_dates, target, exclude_self=False)
            score = log_loss(train[target], oof_pred)

            submission[target] = test_pred
            scores.append({"target": target, "log_loss_oof": score})
            mlflow.log_metric(f"log_loss_oof_{target}", score)
            print(
                f"{target}: OOF log_loss={score:.6f}, "
                f"test_range=[{test_pred.min():.4f}, {test_pred.max():.4f}]"
            )

        scores_df = pd.DataFrame(scores)
        mean_oof = scores_df["log_loss_oof"].mean()
        mlflow.log_metric("mean_log_loss_oof", mean_oof)
        mlflow.log_metric("leaderboard_score", float(mean_oof))
        print(f"\nLeaderboard Score (mean OOF log_loss): {mean_oof:.6f}")

    submission = submission[sample.columns]
    scores_df = pd.DataFrame(scores)
    submission.to_csv(SUBMISSION_PATH, index=False)
    scores_df.to_csv(SCORES_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")
    print(f"Saved {SCORES_PATH}")
    return submission, scores_df


if __name__ == "__main__":
    train_and_predict()
