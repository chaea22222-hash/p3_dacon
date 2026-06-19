# p3-dacon

DACON 5th ETRI Human Understanding AI 대회를 위한 팀 베이스라인 파이프라인.

## Language

**DVC Remote**:
팀 공유 서버의 `~/dacon_project/dvc-storage`에 위치하는 실제 대용량 데이터 파일 저장소. DVC가 버전을 관리하는 대상.
_Avoid_: 공유 디렉토리, 원격 저장소

**Working Data Directory**:
각 팀원의 로컬 프로젝트 내 `data/raw/`. 대회 제공 원본 파일만 포함하며, DVC가 pull한 데이터가 위치하는 디렉토리. git에서 제외됨.
_Avoid_: 공유 디렉토리, 데이터 폴더

**MLflow Server**:
공용 서버에서 팀 전체가 공유하는 단일 MLflow 인스턴스. artifact store는 `~/dacon_project/mlflow-artifacts`, backend store는 `~/dacon_project/mlflow.db`(SQLite). 접속 주소는 팀원 각자의 `.env`에 `MLFLOW_TRACKING_URI`로 설정. DVC와 무관하게 MLflow가 독립적으로 관리하며, 팀 공유가 목적이므로 각자의 프로젝트 폴더가 아닌 공유 디렉토리(`~/dacon_project/`)에 저장한다.
_Avoid_: MLflow, 트래킹 서버

**Experiment**:
MLflow에서 모델 전략 단위로 구분되는 그룹 네임스페이스. `mlflow.set_experiment("팀원id/모델이름")`으로 지정하며, API가 MLflow에서 자동 발견한다(하드코딩 없음). 같은 Experiment 안에 여러 Run이 누적되며, 전처리 조합마다 별도 Run으로 기록된다.
_Avoid_: 실험 파일, 버전 스크립트

**Run**:
스크립트를 한 번 실행한 단위. `mlflow.start_run()`으로 생성되며, 파라미터·지표·아티팩트를 포함한다. 동일 스크립트의 파라미터 변경 시 새로운 Run이 쌓이며, Experiment 내에서 비교 대상이 된다.
_Avoid_: 실험, 학습 기록

**DVC Metadata**:
`data/raw.dvc`. Working Data Directory의 MD5 해시와 구조 정보를 담은 소형 파일. git으로 추적되며 데이터 버전 이력을 기록.
_Avoid_: dvc 파일, 메타파일

**Member Workspace**:
`src/팀원id/` 디렉토리. 전처리 폴더와 하나 이상의 모델 폴더로 구성된다. 팀원이 독립적으로 실험할 수 있는 단위.
_Avoid_: 팀원 폴더, 개인 디렉토리

**Member Preprocessing**:
`src/팀원id/전처리/` 디렉토리. 해당 팀원이 작성한 전처리 파이프라인. 출력은 `outputs/팀원id/features_train.csv`, `outputs/팀원id/features_test.csv`이며 동일 팀원의 여러 모델과 다른 팀원의 모델이 공유해서 사용할 수 있다.
_Avoid_: 전처리 스크립트, feature 생성기

**Model Workspace**:
`src/팀원id/모델이름/` 디렉토리. 하나의 모델 전략을 구현하는 단위. 출력은 `outputs/팀원id/모델이름/`에 저장된다. MLflow Experiment 이름은 `팀원id/모델이름`을 사용한다.
_Avoid_: 모델 폴더, 학습 스크립트

**preprocess_by**:
MLflow Run parameter. 해당 Run에서 사용한 Member Preprocessing의 팀원id를 기록한다(예: `"preprocess_by": "jinseok"`). 동일 모델에 서로 다른 전처리를 적용한 Run을 대시보드에서 필터링·비교하는 기준이 된다.
_Avoid_: 전처리자, preprocessing_owner

**Leaderboard Score**:
대회 공식 평가지표. K개 타깃에 대한 Binary Log Loss의 평균.
`Score = (1/K) * Σⱼ [ -(1/N) * Σᵢ (yᵢⱼ log pᵢⱼ + (1−yᵢⱼ) log(1−pᵢⱼ)) ]`
낮을수록 좋음. MLflow에 `leaderboard_score` 키로 기록하며, 모든 학습 스크립트가 동일한 키를 사용하여 실험 간 비교의 기준이 된다.
_Avoid_: log_loss, mean_log_loss, score
