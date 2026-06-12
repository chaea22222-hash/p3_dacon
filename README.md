# DACON 제5회 ETRI 인간 이해 AI — 팀 MLOps 파이프라인

DACON 제5회 ETRI 인간 이해 AI 대회를 위한 팀 MLOps 파이프라인입니다.

대회 참가와 함께 MLOps 실습 환경 구축을 목표로 합니다. 현재 파이프라인은 다음을 갖추고 있습니다:

- **uv**: 재현 가능한 패키지 환경 관리
- **DVC**: 대회 원본 데이터 버전 관리 (팀 공유 서버 연동)
- **MLflow**: 실험 파라미터·지표 추적 및 팀 간 비교 (`leaderboard_score` 기준)
- 4개 학습 스크립트 (baseline / improved / r3 / v2)

## 피처 요약

- 수치형 컬럼: 일별 `mean`, `std`, `min`, `max`, `median`, `count`
- `heart_rate`: 리스트 길이 및 심박수 수치 요약
- `m_gps`: 첫/마지막 속도·고도·위경도 등 GPS 엣지 요약
- `m_wifi`, `m_ble`: 감지 횟수 및 RSSI 요약
- `m_usage_stats`: 앱 수 및 총 사용 시간 요약
- `m_ambience`: 리스트 길이 및 상위 음향 점수 요약
- 날짜 피처: 일, 요일, 주말 여부, 월, 날짜 인덱스

이진 타깃의 경우 LightGBM이 설치되어 있으면 `LightGBMClassifier`를, 없으면 sklearn 모델을 사용합니다.

## 팀원 온보딩

처음 합류하는 팀원은 **[docs/onboarding.md](docs/onboarding.md)** 를 먼저 읽으세요.
DVC 데이터 pull, MLflow 서버 시작, 학습 스크립트 실행까지의 전체 과정이 안내되어 있습니다.

## 환경 설정

이 프로젝트는 Python 3.12를 사용합니다.

`uv`가 없다면 먼저 설치합니다:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

의존성 설치:

```bash
uv sync
```

데이터 받기 (DVC 원격 접근 필요):

```bash
dvc pull
```

## 실행

데이터셋 검사:

```bash
uv run src/inspect_data.py
```

피처 생성만 실행:

```bash
uv run src/make_features.py
```

기존 피처로 모델 학습 및 제출 파일 생성:

```bash
uv run src/train_baseline.py
```

전체 파이프라인 한 번에 실행:

```bash
uv run src/run_first_submission.py
```

## MLflow

팀 전체가 공용 서버의 단일 MLflow 서버를 공유합니다.

프로젝트 루트에 `.env` 파일을 생성합니다 (git에 포함되지 않음):

```bash
cp .env.example .env
```

서버에 접속된 상태에서 `http://서버IP:5000`으로 MLflow 대시보드에 접근할 수 있습니다.

MLflow 서버 시작 (중복 실행 방지 포함):

```bash
bash scripts/start_mlflow.sh
```

서버 상태 확인: `tmux ls`
서버 로그 확인: `tmux attach -t mlflow` (나올 때: `Ctrl+B` → `D`)

## 현재 팀 베이스라인: R3

현재 팀 베이스라인은 `r3`입니다. r4 센서 규칙 조정보다 퍼블릭 리더보드 결과가 더 좋았기 때문입니다.

```bash
uv run src/train_r3.py
```

정확한 전처리, 보간 파라미터, 검증 주의사항, 팀 실험 규칙은 `R3_TEAM_SHARE.md`를 참고하세요.

## 출력 파일

전체 파이프라인 실행 시 생성되는 파일:

- `outputs/features_train.csv`
- `outputs/features_test.csv`
- `outputs/submission_baseline.csv`
- `outputs/validation_scores.csv`

`outputs/submission_baseline.csv`는 `data/raw/ch2026_submission_sample.csv`와 컬럼 및 행 순서가 정확히 일치합니다.
