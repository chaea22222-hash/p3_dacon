# MLflow Experiment 자동 발견 방식 채택

API의 `EXPERIMENT_NAMES` 하드코딩을 제거하고, MLflow Client로 모든 Experiment를 동적 조회한다.

팀원이 새 모델을 추가할 때마다 `api/main.py`를 수정해야 하는 병목을 제거하기 위해 선택했다. 허용 목록(allowlist) 방식은 "공식 인정된" 실험만 노출하는 장점이 있으나, 대회 기간 중 실험이 빠르게 늘어나는 환경에서는 관리 비용이 유지 이득을 초과한다. 자동 발견 방식에서는 Experiment 이름 컨벤션(`팀원id/모델이름`)이 품질 게이트 역할을 한다.
