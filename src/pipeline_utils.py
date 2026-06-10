from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ITEM_DIR = RAW_DIR / "ch2025_data_items"
TRAIN_PATH = RAW_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = RAW_DIR / "ch2026_submission_sample.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_TRAIN_PATH = OUTPUT_DIR / "features_train.csv"
FEATURES_TEST_PATH = OUTPUT_DIR / "features_test.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission_baseline.csv"
SCORES_PATH = OUTPUT_DIR / "validation_scores.csv"

TIMESTAMP_COL = "timestamp"
SUBJECT_COL = "subject_id"
DATE_COL = "lifelog_date"
SLEEP_DATE_COL = "sleep_date"
KEY_COLS = [SUBJECT_COL, DATE_COL]
LABEL_META_COLS = [SUBJECT_COL, SLEEP_DATE_COL, DATE_COL]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "src").mkdir(parents=True, exist_ok=True)


def item_paths() -> List[Path]:
    return sorted(ITEM_DIR.glob("*.parquet"))


def require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "No parquet engine is installed. Install one with "
                "`.venv/bin/python -m pip install pyarrow`."
            ) from exc


def load_train_sample() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    for df in (train, sample):
        for col in (SLEEP_DATE_COL, DATE_COL):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    return train, sample


def infer_key_columns(item_columns: Iterable[str], label_columns: Iterable[str]) -> List[str]:
    item_columns = set(item_columns)
    label_columns = set(label_columns)
    inferred = []
    if SUBJECT_COL in item_columns and SUBJECT_COL in label_columns:
        inferred.append(SUBJECT_COL)
    if TIMESTAMP_COL in item_columns and DATE_COL in label_columns:
        inferred.append(DATE_COL)
    return inferred or [col for col in KEY_COLS if col in label_columns]


def infer_target_columns(train: pd.DataFrame, sample: pd.DataFrame) -> List[str]:
    meta = set(LABEL_META_COLS)
    common_cols = [col for col in sample.columns if col in train.columns]
    targets = [col for col in common_cols if col not in meta]
    if not targets:
        targets = [col for col in train.columns if col not in meta]
    return targets


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return value or "value"


def parse_object(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def stable_text(value) -> str:
    value = parse_object(value)
    if isinstance(value, dict):
        normalized = {str(key): parse_object(val) for key, val in value.items()}
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps(parse_object(value), ensure_ascii=False, sort_keys=True, default=str)
    if value is None:
        return ""
    return str(value)


def _flatten_numeric_values(value):
    value = parse_object(value)
    if value is None:
        return []
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return [float(value)]
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_flatten_numeric_values(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_numeric_values(item))
        return out
    return []


def _sequence_len(value):
    value = parse_object(value)
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    if value is None:
        return np.nan
    return 1


def _safe_float(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _basic_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_count": 0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_count": int(len(arr)),
    }


def summarize_numeric_sequence(value) -> Dict[str, float]:
    value = parse_object(value) if isinstance(value, str) else value
    row = {"len": _sequence_len(value)}
    if not isinstance(value, (list, tuple, np.ndarray)):
        num = _safe_float(value)
        row.update(_basic_stats([num] if num is not None else [], "numeric"))
        return row
    values = []
    for item in value:
        num = _safe_float(item)
        if num is not None:
            values.append(num)
    row.update(_basic_stats(values, "numeric"))
    return row


def summarize_gps_fast(value) -> Dict[str, float]:
    value = parse_object(value) if isinstance(value, str) else value
    row = {"len": _sequence_len(value)}
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) == 0:
        return row
    first = value[0]
    last = value[-1]
    if isinstance(first, dict):
        for key in ("speed", "altitude", "latitude", "longitude"):
            val = _safe_float(first.get(key))
            if val is not None:
                row[f"first_{key}"] = val
    if isinstance(last, dict):
        val = _safe_float(last.get("speed"))
        if val is not None:
            row["last_speed"] = val
    return row


def summarize_pair_fast(value) -> Dict[str, float]:
    value = parse_object(value) if isinstance(value, str) else value
    row = {"len": _sequence_len(value)}
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) == 0:
        return row
    first = value[0]
    if isinstance(first, np.ndarray):
        first = first.tolist()
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        score = _safe_float(first[1])
        if score is not None:
            row["top_pair_score"] = score
    return row


def summarize_nested_value(value) -> Dict[str, float]:
    value = parse_object(value)
    row = {"len": _sequence_len(value)}
    if value is None:
        return row

    if isinstance(value, (int, float, np.integer, np.floating)):
        num = _safe_float(value)
        row.update(_basic_stats([num] if num is not None else [], "numeric"))
        return row

    if isinstance(value, dict):
        value = [value]

    if not isinstance(value, (list, tuple)):
        return row

    numeric_values = []
    pair_scores = []
    dict_values = {
        "rssi": [],
        "total_time": [],
        "speed": [],
        "altitude": [],
        "latitude": [],
        "longitude": [],
    }

    for item in value:
        num = _safe_float(item)
        if num is not None:
            numeric_values.append(num)
            continue
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            score = _safe_float(item[1])
            if score is not None:
                pair_scores.append(score)
            continue
        if isinstance(item, dict):
            for key in dict_values:
                val = _safe_float(item.get(key))
                if val is not None:
                    dict_values[key].append(val)

    if numeric_values:
        row.update(_basic_stats(numeric_values, "numeric"))
    if pair_scores:
        row.update(_basic_stats(pair_scores, "pair_score"))
    for key, values in dict_values.items():
        if values:
            row.update(_basic_stats(values, safe_name(key)))
    return row


def _numeric_summary(value):
    values = _flatten_numeric_values(value)
    if not values:
        return pd.Series({"len": _sequence_len(value), "numeric_mean": np.nan, "numeric_count": 0})
    return pd.Series(
        {
            "len": _sequence_len(value),
            "numeric_mean": float(np.mean(values)),
            "numeric_count": len(values),
        }
    )


def is_nested_object_series(series: pd.Series, max_rows: int = 100) -> bool:
    checked = 0
    for value in series.dropna():
        value = parse_object(value)
        if isinstance(value, (list, tuple, dict, np.ndarray)):
            return True
        checked += 1
        if checked >= max_rows:
            break
    return False


def _dict_numeric_keys(series: pd.Series, max_rows: int = 1000) -> List[str]:
    keys = set()
    checked = 0
    for value in series.dropna():
        value = parse_object(value)
        rows = value if isinstance(value, (list, tuple)) else [value]
        for row in rows:
            if isinstance(row, dict):
                for key, val in row.items():
                    if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                        keys.add(str(key))
        checked += 1
        if checked >= max_rows:
            break
    return sorted(keys)


def _list_pair_score_feature(series: pd.Series) -> pd.Series:
    def mean_score(value):
        value = parse_object(value)
        if not isinstance(value, (list, tuple)):
            return np.nan
        scores = []
        for row in value:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                score = row[1]
                if isinstance(score, (int, float, np.integer, np.floating)) and not pd.isna(score):
                    scores.append(float(score))
        return float(np.mean(scores)) if scores else np.nan

    return series.map(mean_score)


def _dict_key_mean_feature(series: pd.Series, key: str) -> pd.Series:
    def mean_for_key(value):
        value = parse_object(value)
        rows = value if isinstance(value, (list, tuple)) else [value]
        vals = []
        for row in rows:
            if isinstance(row, dict):
                val = row.get(key)
                if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                    vals.append(float(val))
        return float(np.mean(vals)) if vals else np.nan

    return series.map(mean_for_key)


def add_object_summary_columns(df: pd.DataFrame, object_cols: Sequence[str], source: str) -> pd.DataFrame:
    out = df.copy()
    for col in object_cols:
        if col not in out.columns:
            continue
        if source == "ch2025_mGps":
            summarize = summarize_gps_fast
        elif source in {"ch2025_wHr"}:
            summarize = summarize_numeric_sequence
        elif source == "ch2025_mAmbience":
            summarize = summarize_pair_fast
        else:
            summarize = summarize_nested_value
        summaries = pd.DataFrame(out[col].map(summarize).tolist(), index=out.index)
        for summary_col in summaries.columns:
            out[f"{col}__{safe_name(summary_col)}"] = summaries[summary_col]
    return out


def aggregate_item_file(path: Path, key_cols: Sequence[str] = KEY_COLS) -> pd.DataFrame:
    require_parquet_engine()
    source = path.stem
    df = pd.read_parquet(path)
    print(f"\nAggregating {path.name}: shape={df.shape}")
    print(f"Columns: {list(df.columns)}")

    if SUBJECT_COL not in df.columns or TIMESTAMP_COL not in df.columns:
        raise ValueError(f"{path.name} must contain {SUBJECT_COL!r} and {TIMESTAMP_COL!r}.")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[DATE_COL] = df[TIMESTAMP_COL].dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=list(key_cols))

    raw_value_cols = [col for col in df.columns if col not in set(key_cols) | {TIMESTAMP_COL}]
    object_cols = [
        col
        for col in raw_value_cols
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]
    df = add_object_summary_columns(df, object_cols, source)

    value_cols = [col for col in df.columns if col not in set(key_cols) | {TIMESTAMP_COL}]
    numeric_cols = [
        col
        for col in value_cols
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
    ]
    nested_object_cols = {col for col in object_cols if is_nested_object_series(df[col])}
    categorical_cols = [
        col
        for col in value_cols
        if col not in numeric_cols and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]))
        and col not in nested_object_cols
    ]

    pieces = []
    if numeric_cols:
        numeric = df.groupby(list(key_cols), dropna=False)[numeric_cols].agg(
            ["mean", "std", "min", "max", "median", "count"]
        )
        numeric.columns = [f"{source}__{safe_name(col)}__{stat}" for col, stat in numeric.columns]
        pieces.append(numeric)

    if categorical_cols:
        cat_aggs = {}
        for col in categorical_cols:
            as_text = df[col].map(stable_text)
            grouped = as_text.groupby([df[k] for k in key_cols])
            cat_aggs[f"{source}__{safe_name(col)}__nunique"] = grouped.nunique(dropna=True)
            cat_aggs[f"{source}__{safe_name(col)}__top_count"] = grouped.agg(
                lambda s: int(s.value_counts(dropna=True).iloc[0]) if s.notna().any() else 0
            )
        categorical = pd.DataFrame(cat_aggs)
        pieces.append(categorical)

    if not pieces:
        counts = df.groupby(list(key_cols), dropna=False).size().to_frame(f"{source}__row_count")
        pieces.append(counts)

    agg = pd.concat(pieces, axis=1).reset_index()
    agg.columns = [safe_name(col) if col not in key_cols else col for col in agg.columns]
    print(f"Aggregated {path.name}: shape={agg.shape}")
    return agg


def merge_feature_frames(base: pd.DataFrame, feature_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for frame in feature_frames:
        out = out.merge(frame, on=KEY_COLS, how="left")
    out = add_date_features(out)
    return out


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[DATE_COL], errors="coerce")
    out["date__dayofweek"] = dt.dt.dayofweek
    out["date__is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["date__day"] = dt.dt.day
    out["date__month"] = dt.dt.month
    out["date__day_index"] = (dt - pd.Timestamp("2024-01-01")).dt.days
    return out


def align_feature_columns(train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_cols = set(train_features.columns) - set(LABEL_META_COLS)
    test_cols = set(test_features.columns) - set(LABEL_META_COLS)
    for col in sorted(test_cols - train_cols):
        train_features[col] = np.nan
    for col in sorted(train_cols - test_cols):
        test_features[col] = np.nan
    ordered = LABEL_META_COLS + sorted((set(train_features.columns) | set(test_features.columns)) - set(LABEL_META_COLS))
    return train_features[ordered], test_features[ordered]


def print_frame_overview(name: str, df: pd.DataFrame, max_cols: int = 80) -> None:
    print(f"\n{name}: shape={df.shape}")
    cols = list(df.columns)
    print(f"{name} columns ({len(cols)}): {cols[:max_cols]}{' ...' if len(cols) > max_cols else ''}")
    print(df.head(3).to_string(index=False))
