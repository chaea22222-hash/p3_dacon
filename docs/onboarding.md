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

브라우저에서 `http://localhost:5000` 접속하면 MLflow 대시보드를 볼 수 있습니다.

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

## 참고 문서

- [CONTEXT.md](../CONTEXT.md) — 프로젝트 용어 정의
- [docs/adr/](adr/) — 주요 기술 결정 기록