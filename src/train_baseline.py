from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pipeline_utils import (
    FEATURES_TEST_PATH,
    FEATURES_TRAIN_PATH,
    KEY_COLS,
    LABEL_META_COLS,
    SCORES_PATH,
    SUBMISSION_PATH,
    ensure_dirs,
    infer_target_columns,
    load_train_sample,
    print_frame_overview,
)


def _model_factory(task: str):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        params = {
            "n_estimators": 80,
            "learning_rate": 0.06,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if task == "classification":
            return "lightgbm_classifier", LGBMClassifier(**params)
        return "lightgbm_regressor", LGBMRegressor(**params)
    except Exception:
        if task == "classification":
            return "hist_gradient_boosting_classifier", HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.06,
                random_state=42,
                l2_regularization=0.01,
            )
        try:
            return "hist_gradient_boosting_regressor", HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                random_state=42,
                l2_regularization=0.01,
            )
        except Exception:
            model_cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
            return "random_forest", model_cls(
                n_estimators=150,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )


def _make_pipeline(feature_df: pd.DataFrame, task: str) -> Pipeline:
    numeric_cols = [
        col
        for col in feature_df.columns
        if pd.api.types.is_numeric_dtype(feature_df[col]) and col not in LABEL_META_COLS
    ]
    categorical_cols = [
        col
        for col in feature_df.columns
        if col not in numeric_cols and col not in LABEL_META_COLS
    ]
    transformer = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    model_name, model = _model_factory(task)
    print(f"Using model backend: {model_name}")
    return Pipeline(steps=[("prep", transformer), ("model", model)])


def _coerce_predictions(pred: np.ndarray, sample_col: pd.Series, train_col: pd.Series) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if pd.api.types.is_integer_dtype(train_col) or pd.api.types.is_integer_dtype(sample_col):
        low = float(np.nanmin(train_col.values))
        high = float(np.nanmax(train_col.values))
        pred = np.clip(np.rint(pred), low, high)
        return pred.astype(int)
    return pred


def _is_binary_target(series: pd.Series) -> bool:
    values = sorted(series.dropna().unique().tolist())
    return values in ([0, 1], [0], [1])


def train_and_predict() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    train_labels, sample = load_train_sample()
    target_cols = infer_target_columns(train_labels, sample)
    print(f"Target columns: {target_cols}")

    train_features = pd.read_csv(FEATURES_TRAIN_PATH)
    test_features = pd.read_csv(FEATURES_TEST_PATH)
    print_frame_overview("loaded_features_train", train_features)
    print_frame_overview("loaded_features_test", test_features)

    train_df = train_labels.merge(train_features, on=LABEL_META_COLS, how="left", validate="one_to_one")
    test_df = sample[LABEL_META_COLS].merge(test_features, on=LABEL_META_COLS, how="left", validate="one_to_one")
    print_frame_overview("model_train_table", train_df)
    print_frame_overview("model_test_table", test_df)

    feature_cols = [col for col in train_features.columns if col not in LABEL_META_COLS]
    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    submission = sample.copy()
    scores = []
    for target in target_cols:
        y = train_df[target]
        valid_mask = y.notna()
        X_target = X.loc[valid_mask].copy()
        y_target = y.loc[valid_mask].copy()
        if y_target.nunique(dropna=True) <= 1 or len(y_target) < 5:
            fill_value = y_target.median() if len(y_target) else 0
            valid_pred = np.full(len(y_target), fill_value)
            test_pred = np.full(len(X_test), fill_value)
            model_name = "constant"
        else:
            task = "classification" if _is_binary_target(y_target) else "regression"
            stratify = y_target if y_target.nunique() <= 10 and y_target.value_counts().min() >= 2 else None
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_target,
                y_target,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )
            pipe = _make_pipeline(X_target, task)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_tr, y_tr)
            valid_pred = pipe.predict(X_va)
            test_pred = pipe.predict(X_test)
            y_target = y_va
            model_name = pipe.named_steps["model"].__class__.__name__

        valid_pred_for_metrics = _coerce_predictions(valid_pred, sample[target], train_df[target])
        test_pred = _coerce_predictions(test_pred, sample[target], train_df[target])
        test_pred = np.where(pd.isna(test_pred), int(train_df[target].mode().iloc[0]), test_pred)
        submission[target] = test_pred

        rmse = root_mean_squared_error(y_target, valid_pred_for_metrics)
        mae = mean_absolute_error(y_target, valid_pred_for_metrics)
        r2 = r2_score(y_target, valid_pred_for_metrics) if len(y_target) > 1 else np.nan
        accuracy = accuracy_score(y_target, valid_pred_for_metrics) if _is_binary_target(train_df[target]) else np.nan
        score_row = {
            "target": target,
            "model": model_name,
            "n_train": int(valid_mask.sum()),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "accuracy": accuracy,
        }
        scores.append(score_row)
        print(
            f"{target}: RMSE={rmse:.5f} MAE={mae:.5f} "
            f"ACC={accuracy:.5f} R2={r2:.5f} model={model_name}"
        )

    submission = submission[sample.columns]
    scores_df = pd.DataFrame(scores)
    submission.to_csv(SUBMISSION_PATH, index=False)
    scores_df.to_csv(SCORES_PATH, index=False)
    print_frame_overview("submission", submission)
    print_frame_overview("validation_scores", scores_df)
    print(f"\nSaved {SUBMISSION_PATH}")
    print(f"Saved {SCORES_PATH}")
    return submission, scores_df


def main() -> None:
    train_and_predict()


if __name__ == "__main__":
    main()
