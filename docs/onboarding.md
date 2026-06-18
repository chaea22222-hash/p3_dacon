# 팀원 온보딩 가이드

이 문서는 프로젝트에 처음 합류하는 팀원이 로컬 환경을 세팅하고 첫 실험을 실행하기까지의 과정을 안내합니다.

## 사전 조건

- 6층 서버 계정 및 SSH 접속 가능
- `~/dacon_project/` 심링크 생성 완료 (서버 관리자가 설정)
- `uv` 설치 ([설치 방법](https://astral.sh/uv))

`uv`가 없다면:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. 저장소 클론 및 의존성 설치

📢 [팀 프로젝트 uv 사용 가이드]

패키지 동기화를 할 때는 `deactivate`를 실행해서 기존의 venv 가상환경을 해제한 상태에서 프로젝트 루트 디렉토리로 이동해 **uv sync**만 입력해 주세요.

uv가 알아서 폴더 내의 .venv를 찾아 정확하게 패키지를 일치시켜 줍니다.

추가 패키지 설치가 필요한 경우, `uv add <패키지명>` 으로 설치하시면 됩니다.

코드를 실행할 때도 가상환경을 켤 필요 없이 **uv run 파이썬파일.py**로 실행하면 안전합니다.

```bash
git clone https://github.com/chaea22222-hash/p3_dacon.git
cd p3_dacon
uv sync
```

## 2. DVC로 데이터 받기

DVC remote는 공용 서버의 `~/dacon_project/dvc-storage`를 가리키도록 이미 설정되어 있습니다 (`.dvc/config` 참고).  
`~/dacon_project/` 심링크가 있으면 별도 설정 없이 바로 pull 가능합니다.

```bash
uv run dvc pull
```

성공하면 `data/raw/` 아래에 대회 원본 파일이 생성됩니다.

> 심링크가 없거나 권한 오류가 발생하면 서버 관리자에게 문의하세요.

## 3. MLflow 환경 설정

`.env` 파일을 생성합니다 (git에 포함되지 않습니다):

```bash
cp .env.example .env
```

`.env` 파일 내용:

```
MLFLOW_TRACKING_URI=http://localhost:5000
```

## 4. MLflow 서버 시작

팀 공유 MLflow 서버는 **한 명만 실행**하면 됩니다. 이미 실행 중인지 먼저 확인하세요.

```bash
# 실행 중 여부 확인
tmux ls

# 실행 중이 아닐 때만 시작
bash scripts/start_mlflow.sh
```

브라우저에서 `http://서버IP:5000` 접속하면 MLflow 대시보드를 볼 수 있습니다.

서버 로그 확인: `tmux attach -t mlflow` (나올 때: `Ctrl+B` → `D`)

## 5. 학습 스크립트 실행

피처 생성 후 학습:

```bash
uv run python src/make_features.py
uv run python src/train_r3.py      # 현재 팀 베이스라인
```

또는 전체 파이프라인 한 번에:

```bash
uv run python src/run_first_submission.py
```

실행 후 MLflow 대시보드에서 `leaderboard_score`(낮을수록 좋음)로 실험 결과를 비교할 수 있습니다.

## 사용 가능한 학습 스크립트

| 스크립트 | 방식 | MLflow Experiment |
|---|---|---|
| `train_baseline.py` | sklearn 분류/회귀 모델 | `baseline` |
| `train_improved.py` | 피험자 평균 | `improved` |
| `train_r3.py` | 피험자 평균 + 시간 가중 보간 | `r3` |
| `train_v2.py` | CatBoost + 피험자 평균 (시간 기반 분할) | `v2` |

## 6. 예측 서버 및 XAI 대시보드 실행 (선택)

팀 베이스라인(r3) 예측 결과와 MLflow 실험을 웹 UI로 확인하려면 아래 순서로 실행합니다.

MLflow 서버가 이미 실행 중이어야 합니다 (4단계 참고).

**FastAPI 예측 서버 시작:**

```bash
bash scripts/start_fastapi.sh
```

**브라우저에서 접속:**

```
http://서버IP:8151/dashboard
```

### 대시보드 탭 구성

| 탭 | 내용 |
|----|------|
| 📊 예측 결과 | r3 OOF 예측 확률, 피험자별·요일별·날짜별 분포 |
| 🔬 데이터 분석 | 타겟별 양성률, 상관관계, 결측값, 피험자 시계열 EDA |
| 🔍 XAI 해석 | 특성 기여도(SHAP-style), alpha 블렌딩 시각화, OOF Log-Loss |
| 🧪 MLflow 실험 | 4개 실험 leaderboard_score 비교, Run 히스토리 |
| 🔄 파이프라인 | MLOps 전체 흐름도, r3 수식, 실험 히스토리 |
| 📂 데이터 로드 | CSV 업로드 (없으면 샘플 데이터로 자동 시연) |

API 문서(Swagger UI): `http://서버IP:8151/docs`

## 참고 문서

- [CONTEXT.md](../CONTEXT.md) — 프로젝트 용어 정의
- [docs/adr/](adr/) — 주요 기술 결정 기록
