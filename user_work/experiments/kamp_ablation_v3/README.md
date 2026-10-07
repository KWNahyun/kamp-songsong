# 가설 검증용 학습 대기열 v3

> 2026-09-30 결과 분석 완료: [v3 결과와 다음 단계](<../../reports/archive/legacy_reports_20261003/Ablation_v3_결과분석_및_다음단계.md>). 신규 학습 23개 완료·3개 중단. YOLO 추가 seed의 동일 예측 및 크기 loss 수치 안전성 문제를 확인했으므로, 반복성·효과 해석은 결과 보고서를 우선합니다.

2026-09-30. [설계 보고서](<../../reports/archive/legacy_reports_20261003/아키텍처_가설_재설계_보고서.md>)의 1차 loss ablation과 seed 반복만 실행한다. 조건부 새 head·EASE/MDS·추가 P2 조합은 아직 실행 대상이 아니다.

## 구성

| 학습 | seed당 수 | seed 수 | 새 학습 수 |
|---|---:|---:|---:|
| G: 상대 크기 loss λ=0.1 / 0.3 | 2 | 3 | 6 |
| M: VFL→MAL | 1 | 3 | 3 |
| GM: G+M λ=0.1 / 0.3 | 2 | 3 | 6 |
| C0: bbox L1 weight 5→7.5 | 1 | 3 | 3 |
| YOLO/D-FINE 기본형·P2 반복 | 4 | 추가 2 | 8 |
| 합계 | | | 26 |

기본형/P2의 seed 20260929 결과 4개는 기존 실험에서 재사용한다. 신규 seed는 20260930·20261001이며 변경 구성에는 20260929도 포함한다. seed는 실행 날짜가 아니다. 초기 가중치는 기존 공통 init 파일로 고정되므로 **초기화 변동이 아닌 학습 순서·학습 RNG 변동**에 대한 반복이다.

모든 학습은 기존 임시 전처리, train 358/val 64, 640 입력, batch 8, FP32, 30 epoch. GPU 한 작업씩 순차 실행한다. test 78장은 사용하지 않는다. epoch별 native validation 및 best checkpoint 저장은 학습에 포함된다. 별도 공통 평가·failure 분석·모델 순위 해석은 실행하지 않는다.

## 변경의 정확한 범위

- 크기 loss: 정규화된 폭·높이의 log 비율에 Huber(delta=1), eps=1e-6. 마지막 정상 decoder의 O2O 양성 수로 평균. 기존 GO union regression·DN·auxiliary loss는 그대로 유지한다. λ=.1/.3은 사전 고정한 탐색 대조이며 결과를 보고 선택한 값이 아니다.
- MAL: 저자 구현의 gamma=2, mal_alpha=None와 값 및 gradient를 비교. 기존 VFL이 쓰이던 최종/auxiliary/encoder/DN/pre 분류 항목을 모두 MAL로 교체한다. matching·증강·FDR·LQE는 유지한다. 로그의 해당 이름은 loss_mal이다. DEIM 전체 재현이 아니다.
- C0: 기존 모든 bbox L1 항목의 가중치만 1.5배; GIoU/FGL/DDF 가중치는 유지한다. 크기 loss 전용 효과와 회귀 감독 강도 효과를 비교한다.
- 전체 새 실험은 원본 D-FINE 소스를 수정하지 않고 별도 실행 wrapper를 사용한다.

## 기록과 관리

- run_manifest.json: 전체 명령·설정·입력 파일 해시·기존 비교 모델 경로.
- preflight.json/log: 공식 MAL 대비 값·gradient, 크기 loss gradient, 실제 8×640 배치 forward/backward 검사.
- queue_status.json: pending/running/waiting_for_vram/trained/failed 상태 및 PID.
- logs/: 학습별 실행 로그. runs/: 각 모델 가중치·학습 로그.
- 실패한 작업은 실패로 기록하고 다음 작업을 진행한다. 비교 조건을 바꾸는 batch 자동 축소나 자동 재시도는 하지 않는다.
- GPU 여유 메모리가 18,000MiB 미만이면 다음 작업을 대기한다. 다른 사용자의 프로세스는 종료하지 않는다.
- 중복 실행 방지 lock과 파일 해시 검사를 사용한다. PC 종료·재부팅에는 계속 실행되지 않으며, 실패/중단 후에는 기존 기록을 확인해 별도 재개 처리가 필요하다.

아직 결과 분석 전이다. 전체 완료 뒤 동일 평가기로 원본 출력과 동일 후처리를 비교해야 한다.
