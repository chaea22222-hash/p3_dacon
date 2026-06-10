from __future__ import annotations

import pandas as pd

from pipeline_utils import (
    FEATURES_TEST_PATH,
    FEATURES_TRAIN_PATH,
    KEY_COLS,
    aggregate_item_file,
    align_feature_columns,
    ensure_dirs,
    item_paths,
    load_train_sample,
    merge_feature_frames,
    print_frame_overview,
)


def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    train, sample = load_train_sample()
    print_frame_overview("train_labels", train)
    print_frame_overview("sample_submission", sample)

    feature_frames = [aggregate_item_file(path, KEY_COLS) for path in item_paths()]

    train_base = train[[col for col in ["subject_id", "sleep_date", "lifelog_date"] if col in train.columns]].copy()
    test_base = sample[[col for col in ["subject_id", "sleep_date", "lifelog_date"] if col in sample.columns]].copy()

    train_features = merge_feature_frames(train_base, feature_frames)
    test_features = merge_feature_frames(test_base, feature_frames)
    train_features, test_features = align_feature_columns(train_features, test_features)

    train_features.to_csv(FEATURES_TRAIN_PATH, index=False)
    test_features.to_csv(FEATURES_TEST_PATH, index=False)
    print_frame_overview("features_train", train_features)
    print_frame_overview("features_test", test_features)
    print(f"\nSaved {FEATURES_TRAIN_PATH}")
    print(f"Saved {FEATURES_TEST_PATH}")
    return train_features, test_features


def main() -> None:
    build_features()


if __name__ == "__main__":
    main()
