"""
ETRI Lifelog Prediction API (r3 baseline)

POST /predict  — subject_id + lifelog_date → 7 target probabilities
GET  /subjects — 지원 피험자 목록
GET  /model    — 모델 파라미터 정보
"""
from __future__ import annotations

import sys
from pathlib import Path

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline_utils import DATE_COL, SUBJECT_COL, TRAIN_PATH  # noqa: E402

# ── r3 파라미터 ──────────────────────────────────────────────
TARGET_PARAMS: Dict[str, Tuple[int, float, float]] = {
    "Q1": (7,  2.0, 0.7),
    "Q2": (7,  2.0, 0.5),
    "Q3": (14, 2.0, 0.5),
    "S2": (21, 7.0, 0.7),
}
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
PROB_CLIP = 1e-6

TARGET_LABELS = {
    "Q1": "수면 질",
    "Q2": "피로도",
    "Q3": "스트레스",
    "S1": "총 수면 시간",
    "S2": "수면 효율",
    "S3": "수면 지연",
    "S4": "각성 횟수",
}


# ── 데이터 로드 ──────────────────────────────────────────────
def _load_model() -> Tuple[pd.DataFrame, dict, list]:
    train = pd.read_csv(TRAIN_PATH)
    train[DATE_COL] = pd.to_datetime(train[DATE_COL])
    subject_means: Dict[str, Dict[str, float]] = {}
    for sid, grp in train.groupby(SUBJECT_COL):
        subject_means[sid] = {t: float(grp[t].mean()) for t in TARGETS}
    subjects = sorted(train[SUBJECT_COL].unique().tolist())
    return train, subject_means, subjects


TRAIN_DF, SUBJECT_MEANS, SUBJECTS = _load_model()


# ── 예측 로직 (train_r3.py의 predict_target과 동일) ──────────
def _local_pred(subject_rows: pd.DataFrame, pred_date: pd.Timestamp,
                target: str, window: int, decay: float, fallback: float) -> float:
    dists = (subject_rows[DATE_COL] - pred_date).dt.days.abs()
    mask = dists <= window
    cands = subject_rows.loc[mask]
    dists_w = dists.loc[mask]
    if cands.empty:
        return fallback
    weights = np.exp(-dists_w.to_numpy(float) / decay)
    return float(np.average(cands[target].to_numpy(float), weights=weights))


def predict(subject_id: str, lifelog_date: str) -> Dict[str, float]:
    if subject_id not in SUBJECT_MEANS:
        raise ValueError(f"알 수 없는 피험자: {subject_id}")
    pred_dt = pd.Timestamp(lifelog_date)
    subj_rows = TRAIN_DF[TRAIN_DF[SUBJECT_COL] == subject_id]
    result = {}
    for t in TARGETS:
        sm = SUBJECT_MEANS[subject_id][t]
        params = TARGET_PARAMS.get(t)
        if params is None:
            pred = sm
        else:
            window, decay, alpha = params
            local = _local_pred(subj_rows, pred_dt, t, window, decay, sm)
            pred = alpha * sm + (1.0 - alpha) * local
        result[t] = round(float(np.clip(pred, PROB_CLIP, 1.0 - PROB_CLIP)), 6)
    return result


# ── FastAPI ──────────────────────────────────────────────────
app = FastAPI(
    title="ETRI Lifelog Prediction API",
    description="r3 Temporal Weighted Blend — subject mean + 인접 레이블 보간",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    subject_id: str
    lifelog_date: str

    @field_validator("lifelog_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            pd.Timestamp(v)
        except Exception:
            raise ValueError("lifelog_date 형식: YYYY-MM-DD")
        return v


class PredictResponse(BaseModel):
    subject_id: str
    lifelog_date: str
    predictions: Dict[str, float]
    labels: Dict[str, str]
    model: str


@app.get("/")
def root():
    return {"message": "ETRI Lifelog Prediction API", "docs": "/docs"}


@app.get("/subjects")
def get_subjects():
    return {
        "subjects": SUBJECTS,
        "count": len(SUBJECTS),
        "means": SUBJECT_MEANS,
    }


@app.get("/model")
def get_model():
    return {
        "name": "r3 Temporal Weighted Blend",
        "targets": TARGETS,
        "target_labels": TARGET_LABELS,
        "params": {t: {"window": int(p[0]), "decay": p[1], "alpha": p[2]}
                   for t, p in TARGET_PARAMS.items()},
        "subjects_count": len(SUBJECTS),
        "leaderboard_score": 0.6040591633,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    try:
        preds = predict(req.subject_id, req.lifelog_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(
        subject_id=req.subject_id,
        lifelog_date=req.lifelog_date,
        predictions=preds,
        labels=TARGET_LABELS,
        model="r3",
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
