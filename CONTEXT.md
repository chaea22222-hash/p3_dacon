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
MLflow에 등록되는 단일 학습 실행. 파라미터·지표·모델 아티팩트를 포함하며, 파일명이 다른 별도 스크립트가 아닌 동일 스크립트의 파라미터 변경으로 구분된다.
_Avoid_: 실험 파일, 버전 스크립트

**DVC Metadata**:
`data/raw.dvc`. Working Data Directory의 MD5 해시와 구조 정보를 담은 소형 파일. git으로 추적되며 데이터 버전 이력을 기록.
_Avoid_: dvc 파일, 메타파일
