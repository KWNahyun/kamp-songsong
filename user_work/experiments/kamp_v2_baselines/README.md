# KAMP v2 기본 모델·P2 재현

완료: 2026-10-01 16:41 KST에 12개 학습 정상 종료. 이후 validation 66장의 공통 평가, 조건별 실패, 여섯 D-FINE checkpoint의 내부 추적 완료. 결과는 [분석 보고서](<../../reports/archive/legacy_reports_20261003/새_데이터_v2_기본모델_P2_결과_및_실패분석.md>), `analysis/aggregate.csv`, `analysis/completion.json`을 참조한다. 추가 학습은 시작하지 않았다.

2026-10-01 시작. 원본 패키지는 수정하지 않는다.

- 팀 분할 그대로 train 350장/797 bbox, validation 66장/144 bbox 사용.
- test 이미지·라벨은 읽거나 변환하지 않으며 학습 설정에 포함하지 않는다. 팀의 과거 test 활용 이력은 별도 한계로 유지한다.
- 동일 640 중앙 letterbox, batch 8, 30 epoch, AdamW lr 0.0002, AMP 끔, 증강 끔. 이는 1차 통제 비교이며 수렴을 보장하는 최종 학습 길이는 아니다.
- YOLOv8s, YOLOv8s+P2, D-FINE-S, D-FINE-S+P2 순으로 seed 20260929를 실행한 다음 20260930, 20261001을 반복한다. 총 12회.
- KAMP 학습 checkpoint를 재사용하지 않고 기존 COCO 기반 초기 가중치를 사용한다. 초기 가중치는 고정하고 실행 seed/데이터 순서의 변화를 반복 비교한다.
- YOLO dataloader generator에 실행 seed를 명시해 과거 반복 결과 동일 문제를 수정했다.
- 한 GPU 작업씩 실행하며 빈 VRAM 18GB 이상에서 시작한다. 실패나 checkpoint/epoch 검사 실패 시 큐를 정지한다.
- 데이터 감사 및 좌표 변환: data_audit.json. 모델별 완료 checkpoint와 epoch 지표: runs/.
- 학습 중 지표는 프레임워크별 native 평가다. 공통 평가·실패 분석은 학습 완료 후 별도로 수행한다.
- MAL/UQ는 네 기본 모델 실패 분석 이후 단계다.

## 실시간 확인

```bash
python3 /home/viplab/contest/experiments/kamp_v2_baselines/watch.py
```

모델이 바뀌어도 로그를 따라간다. Ctrl+C는 뷰어만 닫고 학습은 계속된다.

- queue_status.json: 전체 12개 실행 상태
- queue_launcher.log: 모델별 시작·종료
- logs/: 실행별 전체 로그
- current.log: 현재 실행 로그 링크
- runs/*/results.csv: YOLO epoch별 지표
- runs/*/log.txt: D-FINE epoch별 지표
