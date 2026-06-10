"""
Improved training script for ETRI lifelog competition.

Root cause of score ~13: original script submits hard labels (0 or 1).
Log Loss requires probabilities. Submitting 0/1 gives -log(epsilon) ≈ 16
for each wrong prediction, producing the observed ~13 leaderboard score.

Analysis summary:
- Global mean prediction: ~0.664 log loss (random baseline)
- Subject mean prediction: ~0.594 log loss  <-- BEST WITH CURRENT FEATURES
- LightGBM with sensor features: ~0.69 (WORSE, overfits with 276 features / 450 samples)
- Within-subject sensor correlations with targets: max 0.16, too weak to help

Key insight: sensor features don't reliably predict day-to-day label variation
within individuals. The strongest signal is each subject's personal baseline.

To improve further below ~0.59, better hand-crafted features are needed
(e.g., actual sleep duration from raw data, specific stress markers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from pipeline_utils import (
    LABEL_META_COLS,
    OUTPUT_DIR,
    SUBJECT_COL,
    ensure_dirs,
    infer_target_columns,
    load_train_sample,
)

SUBMISSION_PATH = OUTPUT_DIR / "submission_improved.csv"
SCORES_PATH = OUTPUT_DIR / "validation_scores_improved.csv"
PROB_CLIP = 1e-6


def train_and_predict() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    train_labels, sample = load_train_sample()
    target_cols = infer_target_columns(train_labels, sample)
    print(f"Target columns: {target_cols}")

    submission = sample.copy()
    scores = []

    for target in target_cols:
        # Compute per-subject probability from training data
        subject_means = train_labels.groupby(SUBJECT_COL)[target].mean()

        # Test set: map each subject to their training mean
        test_pred = sample[SUBJECT_COL].map(subject_means).clip(PROB_CLIP, 1 - PROB_CLIP)
        submission[target] = test_pred.values

        # Approximate OOF log loss (in-sample, slight optimism due to including self in mean)
        train_pred = train_labels[SUBJECT_COL].map(subject_means).clip(PROB_CLIP, 1 - PROB_CLIP)
        ll = log_loss(train_labels[target], train_pred)

        scores.append({"target": target, "log_loss_approx": ll, "n_train": len(train_labels)})
        print(f"{target}: approx_log_loss={ll:.5f}  "
              f"pred_range=[{test_pred.min():.3f}, {test_pred.max():.3f}]")

    mean_ll = sum(r["log_loss_approx"] for r in scores) / len(scores)
    print(f"\nMean approx log_loss: {mean_ll:.5f}")

    submission = submission[sample.columns]
    scores_df = pd.DataFrame(scores)
    submission.to_csv(SUBMISSION_PATH, index=False)
    scores_df.to_csv(SCORES_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH}")
    return submission, scores_df


def main() -> None:
    train_and_predict()


if __name__ == "__main__":
    main()
