# 아키텍처 결정 기록 (ADR)

이 디렉토리는 프로젝트의 주요 기술적 결정들을 기록합니다.

## 결정 과정

이 프로젝트의 ADR들은 [grill-with-docs](https://github.com/anthropics/claude-code) 스킬을 활용하여 작성되었습니다. grill-with-docs는 Claude Code의 인터뷰 기반 설계 세션 스킬로, 오랫동안 미뤄졌던 기술적 결정들(DVC 도입 방식, 실험 추적 도구 선택, 학습 스크립트 구조 등)을 단 한 번의 대화 세션으로 빠르게 정리하고 문서화할 수 있었습니다.

결정을 내리는 과정에서 각 선택지의 trade-off를 명확히 하고, 결정 즉시 CONTEXT.md와 ADR에 기록하는 방식으로 진행했습니다.

## 목록

- [0001](0001-dvc-scope-raw-only.md) — DVC 추적 범위를 원본 데이터로 한정
- [0002](0002-mlflow-experiment-tracking.md) — 실험 추적 도구로 MLflow 선택
- [0003](0003-training-scripts-not-consolidated.md) — 학습 스크립트를 단일 파일로 통합하지 않음
