# DVC 추적 범위를 원본 데이터로 한정

DVC remote(`~/dacon_project/dvc-storage`)로 추적하는 대상을 `data/raw/`(대회 제공 원본 파일)로만 한정한다. `outputs/`(features, submission 등 생성 파일)는 DVC로 관리하지 않는다.

outputs는 코드를 실행하면 재현 가능하므로 버전 관리 필요성이 낮고, 대회 맥락에서 결과 공유는 제출 파일을 직접 공유하는 방식이 자연스럽다.
