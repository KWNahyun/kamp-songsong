# v3.1 1차 묶음

2026-09-30. **기존 v3와 다른 목적 함수이며 결과를 섞지 않는다.**

- 고정 자료: kamp500_telea_v1_640, train358/val64, test 미사용.
- baseline/MAL 기존 3개 seed 재사용. 수정 상대 크기 Huber λ0.1 단독/결합 3개 seed씩 새 학습.
- 기존 initialization, optimizer·schedule·matching·batch8·30epoch·FP32 유지.
- `stable_loss.py`: `(pred_wh-gt_wh)/(gt_wh+1e-6)`의 Huber, delta1, 양성 수 정규화. final regular decoder O2O에만 추가.
- `preflight_G.json`, `preflight_GM.json`: 수치 정의역·gradient와 실제 4배치 optimizer update 검증. 이 smoke 가중치는 본 실험에 미사용.
- `trace_decoder.py`: eval 모드 6개 checkpoint ×64장, 원 forward와 최종 tensor 일치 검사. query ID별 경계 변화, 같은 후보 집합의 순위, score floor, NMS 보존을 기록.
- trace의 encoder_all_counterfactual은 선택되지 않은 격자에도 기존 head를 적용한 진단용 후보다. per-GT oracle coverage를 detector Recall이라고 부르지 않는다. trace NMS는 unclipped 정규화 좌표 진단이며 성능 표는 기존 공통 원본 좌표 evaluator를 사용한다.
- `status.json`: 실행 중 상태. 같은 구성 실패 시 그 구성 남은 seed 차단. 무한 재시도 없음.
- 완료된 실행은 `evaluate_v31.py`로 자동 평가, 마지막에 `analyze_v31.py`로 비교·후처리 factorial 집계. 이전 후처리 결과는 `../kamp_ablation_v3/analysis_with_retries/postprocess.csv`에 이미 존재한다.
- `analysis` 안 G01/GM01 표기는 모두 **v3.1 수정 loss**이며 v3와 같은 이름이어도 같은 방법이 아니다.

주 실행은 `run_batch.py`다. 이미 status가 있으면 덮어쓰지 않고 거절한다. 진행 중에는 재실행하지 않는다. 현재 단일 GPU 순차 큐이며 YOLO seed 수정 재학습·새 query 모듈·P2 추가 학습은 포함하지 않는다.

전체 연구 이력은 [통합 보고서](../../프로젝트_진행과정_통합보고서.md), 이전 완료 실험은 [v3 결과](<../../reports/archive/legacy_reports_20261003/Ablation_v3_결과분석_및_다음단계.md>)를 참조한다.
