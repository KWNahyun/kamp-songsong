# 새 데이터 v2 UQ 교차 비교

학습은 2026-10-01 22:19:31 KST에 6회 모두 완료됐다. 공통 평가·동일 query 진단 완료. [결과 보고서](<../../reports/archive/legacy_reports_20261003/새_데이터_v2_UQ_교차비교_결과.md>) 참조. MAL+UQ의 AP 이득은 세 seed 재현됐지만 낮은 FP TP 및 지역 대표 query는 변하지 않았다.

비교: D-FINE / D-FINE+UQ / MAL / MAL+UQ. 기본·MAL checkpoint는 기존 세 seed의 best checkpoint 그대로 고정하며 새 검출기 학습은 하지 않는다. UQ만 train350장으로 각각 30 epoch 학습한다. 마지막 epoch만 평가하며 UQ 학습 또는 epoch 선택에 validation/test를 사용하지 않는다. 기본 checkpoint 자체는 이전 validation으로 선택됐다는 한계가 있다.

추가 모듈은 기존 unary quality head, 62,007 parameters. LR .0002, AdamW, batch8, 24 epoch에서 LR .1배. 점수 결합은 기존 sqrt(base probability × quality probability)로 고정하며 새 최적화 탐색은 하지 않는다. 기본형과 MAL에 동일한 초기 seed를 쓴다. 실행 종료마다 검출기 전체 tensor가 원본 checkpoint와 bitwise 동일한지 검증한다.

총 6회 순차 실행. 실행 오류·비유한 loss·본체 변경 시 정지한다. 예상 약 10~15분.

```bash
python3 /home/viplab/contest/experiments/kamp_v2_uq/watch.py
```

평가: `.detector-venv/bin/python experiments/kamp_v2_uq/evaluate.py dfine_base_UQ_seed20260929 unary`. 출력은 `runs_frozen/<name>/common_eval/`.

주 비교는 동일 NMS .7, scale1.0. native 점수 변화와 후보 순위, NMS 뒤 정밀 후보 생존도 별도 진단한다. 공통 평가 시 실제 box 배열 불변 여부도 확인한다. test 사용 없음.
