"""
ETRI Lifelog Prediction API — MLflow 연동 버전

기존 엔드포인트 (하위 호환 유지):
  POST /predict  — subject_id + lifelog_date → 7 target probabilities
  GET  /subjects — 지원 피험자 목록
  GET  /model    — 현재 서빙 모델 파라미터 정보

MLflow 연동 신규 엔드포인트:
  GET  /experiments         — 4개 Experiment의 best leaderboard_score 비교
  GET  /experiments/{name}/runs — 특정 Experiment의 Run 목록
  GET  /health              — 서버 + MLflow 연결 상태
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline_utils import DATE_COL, SUBJECT_COL, TRAIN_PATH  # noqa: E402

# ── MLflow 설정 ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# CONTEXT.md 기준 4개 Experiment
EXPERIMENT_NAMES = ["baseline", "improved", "r3", "v2"]

# ── r3 기본 파라미터 (MLflow 조회 실패 시 fallback) ──────────
_DEFAULT_PARAMS: Dict[str, Tuple[int, float, float]] = {
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


# ── MLflow 헬퍼 ──────────────────────────────────────────────
def _mlflow_client() -> mlflow.MlflowClient:
    return mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _best_run(experiment_name: str) -> Optional[dict]:
    """experiment_name 내에서 leaderboard_score가 가장 낮은 Run을 반환."""
    try:
        client = _mlflow_client()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.leaderboard_score ASC"],
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "experiment": experiment_name,
            "leaderboard_score": run.data.metrics.get("leaderboard_score"),
            "params": run.data.params,
            "start_time": run.info.start_time,
        }
    except Exception:
        return None


def _load_params_from_mlflow() -> Dict[str, Tuple[int, float, float]]:
    """
    MLflow r3 experiment의 best run에서 TARGET_PARAMS를 복원.
    파라미터가 없거나 MLflow 연결 실패 시 기본값(_DEFAULT_PARAMS) 사용.
    """
    best = _best_run("r3")
    if best is None or not best.get("params"):
        return _DEFAULT_PARAMS

    params = best["params"]
    result: Dict[str, Tuple[int, float, float]] = {}
    # train_r3.py는 "Q1_window", "Q1_decay", "Q1_alpha" 형태로 로깅
    for t in TARGETS:
        w_key = f"{t}_window"
        d_key = f"{t}_decay"
        a_key = f"{t}_alpha"
        if w_key in params and d_key in params and a_key in params:
            result[t] = (int(params[w_key]), float(params[d_key]), float(params[a_key]))

    # MLflow에 저장된 파라미터가 없으면 기본값 fallback
    return result if result else _DEFAULT_PARAMS


# ── 서버 시작 시 데이터·파라미터 로드 ───────────────────────
def _load_train() -> Tuple[pd.DataFrame, dict, list]:
    train = pd.read_csv(TRAIN_PATH)
    train[DATE_COL] = pd.to_datetime(train[DATE_COL])
    subject_means: Dict[str, Dict[str, float]] = {}
    for sid, grp in train.groupby(SUBJECT_COL):
        subject_means[sid] = {t: float(grp[t].mean()) for t in TARGETS}
    subjects = sorted(train[SUBJECT_COL].unique().tolist())
    return train, subject_means, subjects


TRAIN_DF, SUBJECT_MEANS, SUBJECTS = _load_train()
TARGET_PARAMS = _load_params_from_mlflow()  # MLflow → fallback 순


# ── 예측 로직 (train_r3.py의 predict_target과 동일) ──────────
def _local_pred(
    subject_rows: pd.DataFrame,
    pred_date: pd.Timestamp,
    target: str,
    window: int,
    decay: float,
    fallback: float,
) -> float:
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
    description="r3 Temporal Weighted Blend — MLflow 연동 버전",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 스키마 ───────────────────────────────────────────────────
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


# ── 기존 엔드포인트 (하위 호환) ──────────────────────────────
@app.get("/")
def root():
    return {"message": "ETRI Lifelog Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    """서버 및 MLflow 연결 상태 확인."""
    mlflow_ok = False
    try:
        _mlflow_client().search_experiments(max_results=1)
        mlflow_ok = True
    except Exception:
        pass
    return {
        "status": "ok",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "mlflow_connected": mlflow_ok,
        "params_source": "mlflow" if TARGET_PARAMS != _DEFAULT_PARAMS else "default",
    }


@app.get("/subjects")
def get_subjects():
    return {
        "subjects": SUBJECTS,
        "count": len(SUBJECTS),
        "means": SUBJECT_MEANS,
    }


@app.get("/model")
def get_model():
    """현재 서빙 중인 모델 파라미터 반환. MLflow best run 점수도 포함."""
    best = _best_run("r3")
    leaderboard_score = (
        best["leaderboard_score"] if best else 0.6040591633  # fallback
    )
    return {
        "name": "r3 Temporal Weighted Blend",
        "targets": TARGETS,
        "target_labels": TARGET_LABELS,
        "params": {
            t: {"window": int(p[0]), "decay": p[1], "alpha": p[2]}
            for t, p in TARGET_PARAMS.items()
        },
        "subjects_count": len(SUBJECTS),
        "leaderboard_score": leaderboard_score,
        "params_source": "mlflow" if TARGET_PARAMS != _DEFAULT_PARAMS else "default",
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


# ── MLflow 신규 엔드포인트 ───────────────────────────────────
@app.get("/experiments")
def list_experiments():
    """
    4개 Experiment의 best leaderboard_score 목록.
    대시보드 실험 비교 탭에서 사용.
    """
    results = []
    for name in EXPERIMENT_NAMES:
        best = _best_run(name)
        results.append(
            best
            if best is not None
            else {"experiment": name, "leaderboard_score": None, "run_id": None, "params": {}}
        )
    # leaderboard_score 오름차순 (None은 뒤로)
    results.sort(
        key=lambda x: (x["leaderboard_score"] is None, x["leaderboard_score"] or 999)
    )
    return {"experiments": results}


@app.get("/experiments/{experiment_name}/runs")
def list_runs(experiment_name: str, max_results: int = 20):
    """특정 Experiment의 Run 목록 (leaderboard_score 오름차순)."""
    if experiment_name not in EXPERIMENT_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment_name}")
    try:
        client = _mlflow_client()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return {"experiment": experiment_name, "runs": []}
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.leaderboard_score ASC"],
            max_results=max_results,
        )
        return {
            "experiment": experiment_name,
            "runs": [
                {
                    "run_id": r.info.run_id,
                    "leaderboard_score": r.data.metrics.get("leaderboard_score"),
                    "params": r.data.params,
                    "start_time": r.info.start_time,
                }
                for r in runs
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

DASHBOARD_PATH = PROJECT_ROOT / "dashboard.html"

@app.get("/dashboard")
def serve_dashboard():
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard.html을 프로젝트 루트에 놓아주세요")
    return FileResponse(DASHBOARD_PATH)

    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)