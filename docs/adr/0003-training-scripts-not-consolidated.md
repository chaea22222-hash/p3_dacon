# 학습 스크립트를 단일 파일로 통합하지 않음

`train_baseline.py`, `train_r3.py`, `train_v2.py`, `train_improved.py`를 하나의 `train.py`로 통합하지 않는다. 각 스크립트에 MLflow logging만 추가한다.

4개 스크립트는 파라미터 차이가 아니라 알고리즘 자체가 다르다(통계 보간 vs sklearn 파이프라인 vs CatBoost). 전략 패턴으로 통합하려면 리팩토링 비용이 크고, 대회 기간 중 해당 비용을 쓰는 것은 비효율적이다. 통합은 대회 종료 후 정리 시점에 검토한다.
