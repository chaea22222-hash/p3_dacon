# R3 공통 시작점 공유

## 선택 이유

- r1 Public: 약 `0.620`
- r2 Public: 약 `0.607`
- r3 Public: 약 `0.604`
- r4 Public: `0.6060468821`

r4는 로컬 OOF에서 개선됐지만 Public 점수는 r3보다 악화됐습니다. 따라서 팀의 공통 기준점은 r3로 유지합니다.

## R3 전처리 및 예측 방식

### 사용 데이터

- `data/raw/ch2026_metrics_train.csv`
- `data/raw/ch2026_submission_sample.csv`

r3는 원시 센서 parquet 피처를 사용하지 않습니다. 훈련 레이블의 피험자별 기준값과 날짜상 인접한 훈련 레이블을 사용합니다.

### Key 및 날짜 처리

- 예측 행 식별용 컬럼: `subject_id`, `sleep_date`, `lifelog_date`
- 시간적 거리 계산 기준: `lifelog_date`
- 동일 `subject_id` 안에서만 인접 날짜를 탐색

### 기본 앵커

각 타깃에 대해 동일 피험자의 훈련 레이블 평균을 기본 확률로 사용합니다.

```text
subject_mean = 동일 subject_id의 target 평균
```

### 시간적 인접 레이블 보간

선택된 타깃은 같은 피험자의 인접 훈련 날짜 레이블을 지수 감쇠 가중 평균합니다.

```text
weight = exp(-abs(train_date - prediction_date) / decay)
local_prediction = weighted_average(인접 훈련 레이블)
final_prediction = alpha * subject_mean + (1 - alpha) * local_prediction
```

타깃별 파라미터:

| Target | Window | Decay | Subject mean weight (`alpha`) |
|---|---:|---:|---:|
| Q1 | 7일 | 2 | 0.7 |
| Q2 | 7일 | 2 | 0.5 |
| Q3 | 14일 | 2 | 0.5 |
| S2 | 21일 | 7 | 0.7 |
| S1 | 미적용 | - | 1.0 |
| S3 | 미적용 | - | 1.0 |
| S4 | 미적용 | - | 1.0 |

S1, S3, S4는 시간 보간 없이 피험자 평균만 사용합니다.

### 검증 방식

- 훈련 행을 한 개씩 예측할 때 해당 행은 인접 레이블 계산에서 제외합니다.
- 피험자 평균은 기존 r3 결과 재현을 위해 전체 피험자 평균을 사용합니다.
- 평가지표는 타깃별 Log Loss입니다.
- 이 OOF는 피험자 평균에 자기 레이블이 포함되어 다소 낙관적이므로, 절대 점수보다 실험 간 비교용으로 사용합니다.

## 재실행

프로젝트 루트에서:

```bash
uv run src/train_r3.py
```

생성 파일:

- `outputs/submission_r3.csv`
- `outputs/validation_scores_r3.csv`

## 팀 실험 원칙

- 모든 실험은 `submission_r3.csv`를 기준으로 비교합니다.
- 한 번에 한 가지 변경만 적용합니다.
- 로컬 OOF가 개선돼도 Public에서 악화될 수 있으므로 변경 내용과 Public 점수를 함께 기록합니다.
- r4의 전체 타깃 센서 규칙 조정은 기준선에서 제외합니다.
