# MAL 후보 선택 진단 v1

저장된 MAL validation 예측 3개 seed만 사용하는 분석이다. 학습과 test 평가는 수행하지 않는다.

- `analyze_candidates.py`: 고정 0.95 박스 보정 아래 원 후보, Hard NMS 0.7, Linear Soft-NMS 0.7, Gaussian Soft-NMS sigma 0.5를 비교한다.
- `postprocess_metrics.csv`: AP·AP75·FP 예산별 TP.
- `candidate_by_gt.csv`: GT별 후보 존재, 점수 leader, 억제 뒤 후보 보존 진단.
- `suppression_events.csv`: 삭제·score 감소 이벤트. GT 필드는 사후 진단 전용이다.
- `oracle_diagnostics.json`: GT IoU를 이용한 비현실적 상한. 추론 결과가 아니다.
- `repeated_failures.csv`, `cases/`: 세 seed 반복 경계 실패와 시각 자료.
- `manifest.json`: 고정 파라미터, 입력, source hash와 한계.

억제 뒤 0.95 보정을 적용하며 파라미터 sweep은 하지 않았다. 결과 해석은 [후보 선택 진단 보고서](<../../reports/archive/legacy_reports_20261003/후보_선택_진단_보고서.md>)를 따른다.
