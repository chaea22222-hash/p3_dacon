"""
v2: Time-based per-subject split + CatBoost with subject_id as categorical.

Key changes vs v1 (subject mean only):
- Time-based split per subject (앞 80% train / 뒤 20% val) — 시간 누수 방지
- Subject mean computed from training portion only — 미래 정보 누수 방지
- CatBoost: subject_id를 categorical feature로 직접 학습
- Within-subject top-k feature selection (타깃별)
- Final submission: full training data로 재학습
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_utils import (
    FEATURES_TEST_PATH,
    FEATURES_TRAIN_PATH,
    LABEL_META_COLS,
    OUTPUT_DIR,
    SUBJECT_COL,
    DATE_COL,
    ensure_dirs,
    infer_target_columns,
    load_train_sample,
)

SUBMISSION_PATH = OUTPUT_DIR / "submission_v2.csv"
SCORES_PATH     = OUTPUT_DIR / "validation_scores_v2.csv"
PROB_CLIP = 1e-6
TOP_K     = 20   # 타깃별 within-subject 상관 상위 피처 수


def _time_based_split(df: pd.DataFrame, ratio: float = 0.8):
    """피험자별로 날짜 순 정렬 후 앞 ratio → train, 뒤 → val 인덱스 반환."""
    train_idx, val_idx = [], []
    for _, grp in df.groupby(SUBJECT_COL):
        grp_sorted = grp.sort_values(DATE_COL)
        n = len(grp_sorted)
        split = int(n * ratio)
        train_idx.extend(grp_sorted.index[:split].tolist())
        val_idx.extend(grp_sorted.index[split:].tolist())
    return train_idx, val_idx


def _select_features(X_tr: pd.DataFrame, y_tr: pd.Series,
                     subj_tr: np.ndarray, sensor_cols: list, k: int) -> list:
    """Training portion 내 within-subject |correlation| 상위 k개 센서 피처 선택."""
    subjects = np.unique(subj_tr)
    corrs = []
    for sid in subjects:
        mask = subj_tr == sid
        y_s = y_tr.values[mask]
        if y_s.std() < 1e-10:
            corrs.append(np.zeros(len(sensor_cols)))
            continue
        X_s = X_tr[sensor_cols].values[mask]
        c = np.array([
            abs(np.corrcoef(X_s[:, j], y_s)[0, 1]) if X_s[:, j].std() > 1e-10 else 0.0
            for j in range(len(sensor_cols))
        ])
        corrs.append(np.nan_to_num(c))
    avg_corr = np.mean(corrs, axis=0)
    top_idx = np.argsort(avg_corr)[::-1][:k]
    return [sensor_cols[i] for i in top_idx]


def _make_catboost(n_iter: int = 300) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=n_iter,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=5,
        random_seed=42,
        eval_metric="Logloss",
        verbose=False,
    )


def _subject_mean(y: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """각 샘플의 subject 평균을 반환 (leave-self-out 아닌 전체 mean)."""
    tmp = pd.DataFrame({"sid": subj, "y": y})
    means = tmp.groupby("sid")["y"].mean()
    return np.array([means[s] for s in subj])


def train_and_predict():
    ensure_dirs()
    train_labels, sample = load_train_sample()
    target_cols = infer_target_columns(train_labels, sample)
    print(f"Targets: {target_cols}")

    train_features = pd.read_csv(FEATURES_TRAIN_PATH)
    test_features  = pd.read_csv(FEATURES_TEST_PATH)

    train_df = train_labels.merge(train_features, on=LABEL_META_COLS, how="left")
    test_df  = sample[LABEL_META_COLS].merge(test_features, on=LABEL_META_COLS, how="left")

    sensor_cols = [
        c for c in train_features.columns
        if c not in set(LABEL_META_COLS) and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # 결측치 전처리
    imp = SimpleImputer(strategy="median")
    train_df[sensor_cols] = imp.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols]  = imp.transform(test_df[sensor_cols])

    # time-based split
    train_idx, val_idx = _time_based_split(train_df)
    tr = train_df.loc[train_idx].copy()
    va = train_df.loc[val_idx].copy()
    print(f"Time-based split — train: {len(tr)}, val: {len(va)}")

    submission = sample.copy()
    scores = []

    for target in target_cols:
        y_tr = tr[target].values
        y_va = va[target].values
        subj_tr = tr[SUBJECT_COL].values
        subj_va = va[SUBJECT_COL].values
        subj_te = test_df[SUBJECT_COL].values

        # Subject mean (training portion만 사용)
        sm_tr = _subject_mean(y_tr, subj_tr)
        tmp_means = pd.Series(y_tr, index=None).groupby(
            pd.Series(subj_tr)).mean().to_dict()
        global_mean = float(y_tr.mean())
        sm_va = np.array([tmp_means.get(s, global_mean) for s in subj_va])
        sm_te = np.array([tmp_means.get(s, global_mean) for s in subj_te])

        # within-subject 상관 기반 피처 선택
        top_feats = _select_features(tr, pd.Series(y_tr, name=target),
                                     subj_tr, sensor_cols, TOP_K)

        # 피처 구성: [subject_mean] + top-k sensor (within-subject normalized)
        def make_X(df_part, subj_arr, sm_arr, feat_list):
            # within-subject z-score
            X_s = df_part[feat_list].copy()
            X_s["subj_mean"] = sm_arr
            X_s[SUBJECT_COL] = subj_arr
            return X_s

        X_tr_df = make_X(tr, subj_tr, sm_tr, top_feats)
        X_va_df = make_X(va, subj_va, sm_va, top_feats)
        X_te_df = make_X(test_df, subj_te, sm_te, top_feats)

        cat_features = [SUBJECT_COL]

        model = _make_catboost(n_iter=300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_df, y_tr, cat_features=cat_features)

        val_proba = model.predict_proba(X_va_df)[:, 1]
        ll_model  = log_loss(y_va, np.clip(val_proba, PROB_CLIP, 1 - PROB_CLIP))
        ll_sm     = log_loss(y_va, np.clip(sm_va,    PROB_CLIP, 1 - PROB_CLIP))

        print(f"{target}: subj_mean={ll_sm:.5f}  catboost={ll_model:.5f}  "
              f"delta={ll_sm - ll_model:+.5f}")

        # Full data 재학습 후 test 예측
        full_sm_tr = _subject_mean(train_df[target].values, train_df[SUBJECT_COL].values)
        full_means = pd.Series(train_df[target].values).groupby(
            pd.Series(train_df[SUBJECT_COL].values)).mean().to_dict()
        sm_te_full = np.array([full_means.get(s, global_mean) for s in subj_te])

        X_full_df = make_X(train_df, train_df[SUBJECT_COL].values, full_sm_tr, top_feats)
        X_te_full = make_X(test_df, subj_te, sm_te_full, top_feats)

        top_feats_full = _select_features(
            train_df, train_df[target], train_df[SUBJECT_COL].values, sensor_cols, TOP_K
        )
        X_full_df = make_X(train_df, train_df[SUBJECT_COL].values, full_sm_tr, top_feats_full)
        X_te_full = make_X(test_df, subj_te, sm_te_full, top_feats_full)

        model_full = _make_catboost(n_iter=300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_full.fit(X_full_df, train_df[target].values, cat_features=cat_features)

        test_proba = model_full.predict_proba(X_te_full)[:, 1]
        submission[target] = np.clip(test_proba, PROB_CLIP, 1 - PROB_CLIP)

        scores.append({"target": target, "log_loss_subj_mean": ll_sm,
                       "log_loss_catboost": ll_model, "n_val": len(y_va)})

    scores_df = pd.DataFrame(scores)
    print(f"\n=== Summary ===")
    print(scores_df.to_string(index=False))
    print(f"\nMean subj_mean log_loss : {scores_df['log_loss_subj_mean'].mean():.5f}")
    print(f"Mean catboost log_loss  : {scores_df['log_loss_catboost'].mean():.5f}")

    submission = submission[sample.columns]
    submission.to_csv(SUBMISSION_PATH, index=False)
    scores_df.to_csv(SCORES_PATH, index=False)
    print(f"\nSaved {SUBMISSION_PATH}")
    return submission, scores_df


if __name__ == "__main__":
    train_and_predict()
