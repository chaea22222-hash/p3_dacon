from __future__ import annotations

import pandas as pd

from pipeline_utils import (
    DATE_COL,
    ITEM_DIR,
    RAW_DIR,
    TIMESTAMP_COL,
    infer_key_columns,
    item_paths,
    load_train_sample,
    print_frame_overview,
    require_parquet_engine,
)


def main() -> None:
    print(f"Raw directory: {RAW_DIR}")
    print(f"Data item directory: {ITEM_DIR}")

    train, sample = load_train_sample()
    print_frame_overview("train_labels", train)
    print_frame_overview("sample_submission", sample)

    require_parquet_engine()
    paths = item_paths()
    print(f"\nFound {len(paths)} parquet files.")
    for path in paths:
        df = pd.read_parquet(path)
        key_cols = infer_key_columns(df.columns, train.columns)
        print(f"\n{path.name}: shape={df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Inferred prediction keys against labels: {key_cols}")
        if TIMESTAMP_COL in df.columns:
            ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
            print(f"{TIMESTAMP_COL} range: {ts.min()} -> {ts.max()}")
            print(f"Derived {DATE_COL} range: {ts.dt.date.min()} -> {ts.dt.date.max()}")
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
