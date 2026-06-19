# 온보딩 2단계: 모델링 확장 가이드

> **언제 읽나**: `docs/onboarding.md`를 이미 읽은 팀원이, 자신의 전처리와 모델을 새로 추가할 때.

---

## 배경

MLops 파이프라인(DVC, MLflow, FastAPI, 대시보드)이 갖춰진 이후, 팀의 다음 목표는 **예측 성능 향상**이다.
각 팀원이 독립적으로 전처리와 모델을 실험하고, 결과를 MLflow와 대시보드에서 함께 비교한다.

---

## 폴더 구조

```
src/
├── pipeline_utils.py        ← 공유 유틸 (수정 금지)
├── make_features.py         ← 기존 baseline 전처리 (수정 금지)
├── train_r3.py              ← 기존 레거시 (수정 금지)
├── train_baseline.py        ↑ 동일
├── train_improved.py        ↑ 동일
├── train_v2.py              ↑ 동일
│
└── 팀원id/                  ← Member Workspace
    ├── 전처리/              ← Member Preprocessing
    │   └── preprocess.py
    ├── model1/              ← Model Workspace
    │   └── train.py
    └── model2/
        └── train.py

outputs/
└── 팀원id/
    ├── features_train.csv   ← 전처리 출력 (모델 간 공유 가능)
    ├── features_test.csv
    └── model1/
        ├── submission.csv
        └── scores.csv
```

---

## 새 전처리 추가하기

`src/팀원id/전처리/preprocess.py` 생성. 최소 구조:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ 추가

import pandas as pd
from pipeline_utils import PROJECT_ROOT, load_train_sample

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "팀원id"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_features():
    train, sample = load_train_sample()
    # ... 자체 전처리 로직 ...
    train_features.to_csv(OUTPUT_DIR / "features_train.csv", index=False)
    test_features.to_csv(OUTPUT_DIR / "features_test.csv", index=False)

if __name__ == "__main__":
    build_features()
```

> **다른 팀원의 전처리 결과를 쓰고 싶다면**: `outputs/다른팀원id/features_train.csv`를 직접 읽으면 된다.

---

## 새 모델 추가하기

`src/팀원id/model1/train.py` 생성. 최소 구조:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ 추가

import mlflow
import pandas as pd
from dotenv import load_dotenv
from pipeline_utils import PROJECT_ROOT, infer_target_columns

load_dotenv()

PREPROCESS_BY = "팀원id"  # 사용할 전처리 소유자
FEATURES_DIR  = PROJECT_ROOT / "outputs" / PREPROCESS_BY
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "팀원id" / "model1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_and_predict():
    mlflow.set_experiment("팀원id/model1")

    train_feat = pd.read_csv(FEATURES_DIR / "features_train.csv")
    test_feat  = pd.read_csv(FEATURES_DIR / "features_test.csv")
    # ... 모델링 로직 ...

    with mlflow.start_run():
        mlflow.log_param("preprocess_by", PREPROCESS_BY)   # 필수
        mlflow.log_metric("leaderboard_score", score)       # 필수
        # 추가 파라미터/지표는 자유롭게

if __name__ == "__main__":
    train_and_predict()
```

### 다른 팀원 전처리로 비교 실험할 때

`PREPROCESS_BY`만 바꿔서 다시 실행하면 된다. 같은 Experiment 안에 Run이 쌓이고,
대시보드에서 `preprocess_by` 파라미터로 필터링해서 비교할 수 있다.

```python
PREPROCESS_BY = "other_member"  # 이것만 변경
```

---

## 브랜치 전략

| 작업 | 브랜치 이름 예시 |
|------|-----------------|
| 내 전처리/모델 작업 | `model/팀원id/모델관련내용` |
| 완료 후 PR | `model/팀원id/...` → `main` |

중간 통합 브랜치 없음. 작업 완료 후 바로 `main`으로 PR.

---

## MLflow 규칙 요약

| 항목 | 규칙 |
|------|------|
| Experiment 이름 | `팀원id/모델이름` |
| 필수 Run parameter | `preprocess_by` (사용한 전처리 팀원id) |
| 필수 Run metric | `leaderboard_score` (낮을수록 좋음) |

대시보드는 MLflow Experiment를 자동 발견하므로, 위 규칙대로만 기록하면 별도 API 수정 없이 바로 대시보드에 나타난다.

---

## sys.path 깊이 참고

| 스크립트 위치 | `src/`를 가리키는 표현 |
|--------------|----------------------|
| `src/팀원id/전처리/preprocess.py` | `parents[2]` |
| `src/팀원id/모델이름/train.py` | `parents[2]` |
