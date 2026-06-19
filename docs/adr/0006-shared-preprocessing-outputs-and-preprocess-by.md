# 팀원별 전처리 출력 공유 및 preprocess_by Run 파라미터

전처리 출력을 `outputs/팀원id/`에 저장해 팀원 간 공유를 허용하고, 모델 Run에는 `preprocess_by` 파라미터로 사용한 전처리 소유자를 기록한다.

"같은 모델, 다른 전처리"와 "같은 전처리, 다른 모델" 두 축의 비교가 핵심 목표다. Experiment 이름에 전처리 정보를 포함하는 안(`팀원id/모델이름/전처리팀원id`)은 Experiment 수가 조합적으로 폭발하고, MLflow의 Experiment/Run 계층 취지(Experiment = 전략, Run = 변형)에도 맞지 않는다. `preprocess_by`를 Run parameter로 기록하면 단일 Experiment 안에서 필터링·비교가 가능하고, 대시보드에서도 전처리 축과 모델 축을 독립적으로 분석할 수 있다.
