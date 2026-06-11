# 실험 추적 도구로 MLflow 선택

실험 추적(파라미터·지표·아티팩트 기록 및 팀 간 비교)을 위해 MLflow를 공용 서버에 자체 호스팅하여 사용한다.

W&B는 클라우드 의존성이 생기고 서버 내 데이터 반출 문제가 있으며, CSV 로깅(outputs/validation_scores.csv)은 팀 간 실시간 비교가 불가능하다. MLflow는 공용 서버에서 완전히 자체 호스팅 가능하고 Model Registry까지 연결되어 3단계(모델 버전 관리)와 자연스럽게 이어진다.

FastAPI 서빙(`dashboard` 브랜치)은 파일 경로가 아닌 MLflow Model Registry에서 직접 모델을 로드한다. 이로써 실험(2단계) → 모델 등록(3단계) → 서빙(5단계)이 MLflow를 축으로 단절 없이 연결된다.

MLflow 서버 프로세스는 `tmux` 세션(`mlflow`)으로 유지한다. `systemd`는 관리자 권한이 필요하고 Docker는 설치 여부가 불확실하기 때문이다. 서버 재부팅 시 한 명의 담당자가 수동으로 재시작한다.
