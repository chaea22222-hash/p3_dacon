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
MLflow에서 스크립트 단위로 구분되는 그룹 네임스페이스. `mlflow.set_experiment()`로 지정하며 현재 4개(baseline, improved, r3, v2)가 존재한다. 같은 Experiment 안에 여러 Run이 누적된다.
_Avoid_: 실험 파일, 버전 스크립트

**Run**:
스크립트를 한 번 실행한 단위. `mlflow.start_run()`으로 생성되며, 파라미터·지표·아티팩트를 포함한다. 동일 스크립트의 파라미터 변경 시 새로운 Run이 쌓이며, Experiment 내에서 비교 대상이 된다.
_Avoid_: 실험, 학습 기록

**DVC Metadata**:
`data/raw.dvc`. Working Data Directory의 MD5 해시와 구조 정보를 담은 소형 파일. git으로 추적되며 데이터 버전 이력을 기록.
_Avoid_: dvc 파일, 메타파일

**Leaderboard Score**:
대회 공식 평가지표. K개 타깃에 대한 Binary Log Loss의 평균.
`Score = (1/K) * Σⱼ [ -(1/N) * Σᵢ (yᵢⱼ log pᵢⱼ + (1−yᵢⱼ) log(1−pᵢⱼ)) ]`
낮을수록 좋음. MLflow에 `leaderboard_score` 키로 기록하며, 모든 학습 스크립트가 동일한 키를 사용하여 실험 간 비교의 기준이 된다.
_Avoid_: log_loss, mean_log_loss, score
