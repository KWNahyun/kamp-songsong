# Boundary refinement pilot v1

고정 MAL 최종 후보의 좌표만 보정해 IoU≥0.75 후보 coverage와 AP75를 늘릴 수 있는지 확인한다.

- `BR`: query 특징, 집계된 FDR 통계, 점수, 기하를 한 MLP에 넣어 네 경계 residual을 함께 예측하는 대조군.
- `EBR`: FDR의 네 경계 확률분포를 각 edge token으로 처리하고 공유 head로 네 residual을 따로 예측하는 가설 모델.
- 두 모델 모두 MAL 본체를 완전히 고정한다.
- train에서 GT당 기존 IoU가 높은 후보 최대 10개를 선택하고 IoU≥0.1인 query에만 가장 가까운 GT의 경계를 감독한다.
- 경계 이동은 기존 폭·높이의 ±0.75로 제한한다.
- validation은 학습이나 epoch 선택에 사용하지 않고 마지막 30 epoch checkpoint만 평가한다.
- test는 사용하지 않는다.

primary mechanistic endpoint는 score≥0.001 후보의 GT별 최대 IoU≥0.75 coverage다. primary detector comparison은 기존 `MAL + Hard NMS 0.7 + 0.95 scale`과 `refined boxes + Hard NMS 0.7`이다. `refined + NMS + 0.95`는 기존 크기 보정과의 상호작용을 보는 보조 비교다.

## 완료 결과

- BR은 세 seed 모두 평균 최선 IoU와 IoU≥0.75 coverage를 낮췄다. 평균 변화는 AP −0.52, AP75 −1.57, coverage −6.33이다.
- EBR은 seed 20260929 파일럿에서 BR과 같은 coverage 91/142를 보여 확대하지 않았다.
- 원본과 BR 중 하나를 고르는 `BPS`를 추가했으나 최초 seed의 AP75 +2.14·coverage +1은 나머지 두 seed에서 재현되지 않았다. 3-seed 평균은 AP +0.12, AP75 −1.31, coverage −0.67이다.
- 같은 고정 MAL 후보를 재점수화한 기존 UQ가 AP +0.66(3/3 seed 개선), AP75 +0.60으로 가장 일관됐다.
- 임시 데이터의 잠정 구성은 `MAL + UQ + Hard NMS 0.7 + 0.95 scale`이다. BR·EBR·BPS는 현재 단계에서 종료한다.
- validation만 평가했으며 보류 test 78장은 사용하지 않았다.

상세 해석은 [`경계_정제_및_쌍_선택_실험_보고서.md`](<../../reports/archive/legacy_reports_20261003/경계_정제_및_쌍_선택_실험_보고서.md>), 원 수치는 `confirmation_summary.json`과 CSV 파일을 따른다.
