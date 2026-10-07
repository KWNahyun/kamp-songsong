# 새 데이터 MAL 재검증

완료: 세 학습 21:56:31 KST 정상 종료. 22시 예약 분석에서 validation 공통 평가 및 decoder 추적 완료. [결과 보고서](<../../reports/archive/legacy_reports_20261003/새_데이터_v2_MAL_재검증_결과.md>) 참조. UQ는 자동 실행하지 않았다.

2026-10-01 21:37 KST 시작. baseline과 같은 v2 train350/val66, COCO 초기값, 640 입력, batch8, 30 epoch, seed 20260929/20260930/20261001. VFL만 MAL로 변경. size loss/P2/UQ 없음. Test 사용 없음.

Train batch 3회의 loss/gradient 검증 통과. 세 실행 순차 처리, 오류 시 정지. 예상 약 20분, 22:00 KST 분석 예약.

```bash
python3 /home/viplab/contest/experiments/kamp_v2_mal/watch.py
```

`queue_status.json`은 진행 상태, `logs/`는 전체 로그, `runs/`는 checkpoint/epoch 지표다. 공통 평가는 `.detector-venv/bin/python experiments/kamp_v2_mal/evaluate.py dfine_M_seed20260929` 형태로 실행한다. 분석에서는 `kamp_v2_baselines`의 같은 seed 기본 D-FINE과 동일 NMS .7·scale1.0으로 비교한다.
