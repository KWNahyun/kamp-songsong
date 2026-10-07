# Relative selector pilot v1

이 실험은 이미 생성된 최종 query 후보의 순서를 더 잘 정할 수 있는지 확인하는 파일럿이다. 최초 공동학습본은 검출기 재학습 변동이 섞여 보조 기록으로만 남겼고, 주 결과는 각 MAL checkpoint를 비트 단위로 고정한 selector-only 학습이다.

- `UQ`: 각 후보의 query 특징, FDR 분포, 박스 기하, 기존 점수만으로 localization quality를 예측한다.
- `RQS`: `UQ` 입력에 더해 예측 박스가 겹치는 상위 8개 이웃 후보의 특징과 관계를 사용한다.
- 두 선택기의 입력은 모두 `detach`한다. 선택기 loss는 원래 MAL 검출 경로를 바꾸지 않는다.
- quality target은 각 query와 해당 영상 GT 사이의 최대 IoU다.
- 추론 점수는 `sqrt(base probability * predicted quality)`로 미리 고정했다.
- validation을 보고 이웃 수, loss weight, 결합식을 탐색하지 않는다.
- test split은 사용하지 않는다.

비교 질문은 세 가지다.

1. `MAL` 대비 `UQ`가 좋아지는가: 개별 후보의 절대 품질 추정만으로 ranking 문제가 완화되는가?
2. `UQ` 대비 `RQS`가 좋아지는가: 후보 간 관계 정보가 추가 가치를 주는가?
3. `MAL + hard NMS + 0.95 box scale` 대비 `RQS + 같은 후처리`가 좋아지는가: 학습된 선택이 단순 중복 제거보다 나은가?

## 완료 결과

- UQ는 3개 frozen MAL seed의 `Hard NMS 0.7 + 0.95 box scale` AP를 모두 높였다. 평균 변화는 AP +0.66, AP75 +0.60, FP≤6 TP 0이다.
- NMS가 없으면 두 seed에서 FP≤6 TP가 감소했다. UQ는 NMS 대체가 아니라 NMS 이후 quality 보정 대조군이다.
- 단일 seed RQS는 UQ보다 AP와 AP75가 낮았다. 현재 이웃 평균 방식을 확장하지 않는다.
- 세 seed 모두 detector 794개 tensor가 기준 MAL과 bitwise identical임을 확인했다.
- 다음 주력은 후보 좌표를 바꾸는 four-edge boundary refinement다.

상세 해석은 [`상대_후보_품질_모듈_실험_보고서.md`](<../../reports/archive/legacy_reports_20261003/상대_후보_품질_모듈_실험_보고서.md>)를 따른다.
