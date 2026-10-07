# KAMP 임시 전처리·P2 Ablation 실험 기록

갱신: 2026-09-30T10:53:46

**범위:** 30 epoch·단일 seed의 탐색 실험. 아래는 모델 선택에 사용한 validation 결과이며 최종 test 또는 안전성 검증 결과가 아니다.

## 데이터와 비교 조건

- 주 500장: train 358장/877 bbox, val 64장/142 bbox, test 78장/128 bbox. 날짜 그룹 분리, 모든 분할에 3개 장비 포함.
- 색상 채널 차이 >30 마스크, 팽창 없음, Telea radius 2, 회색조 3채널. 정답으로 입력을 수정하지 않음. 색상 표시의 원 신호 복원은 입증되지 않음.
- 두 계열 모두 동일한 640×640 중앙 letterbox PNG. 좌표 역변환 후 공통 pycocotools COCO evaluator로 평가.
- seed 20260929, 30 epoch, batch 8, FP32, COCO 사전학습. 데이터 증강 없음. test 예측·임계값 조정은 하지 않음.
- YOLO와 D-FINE은 각 loss/matching/EMA/학습률 schedule이 다르다. 계열 간 차이를 순수 아키텍처 효과로 단정하지 않음. P2 전후에는 각 계열 내 recipe를 동일하게 유지.
- YOLO: AdamW lr 0.0002, weight decay 0.0001, 선형 LR 감소, warmup 1 epoch. D-FINE: AdamW head lr 0.0002/backbone 0.0001, epoch 24에서 0.1배, warmup 45 step, EMA warmup 100. 정확한 설정은 args.yaml 및 dfine_s.yml 참조.
- 검증 AP50:95로 계열별 best checkpoint 선정. YOLO 기본 rectangular 검증은 672 입력을 만들므로 FixedSquareTrainer로 640 고정. 수정 전 YOLO 실행은 archive/에 보존하고 결과에서 제외.
- YOLO native AP 계산과 COCO AP 계산은 다를 수 있다. YOLO native 예측 JSON을 COCO로 평가한 값, 공통 예측의 640 좌표 평가, 원본 좌표 평가가 일치함을 coordinate_crosscheck.json에서 확인했다. 최종 표에는 공통 COCO 수치만 사용한다.
- YOLO 기준 재실행 일부와 D-FINE 기준 학습은 GPU를 공유했다. 학습 wall time을 속도 비교 지표로 사용하지 않는다.

## 무엇을 P2 ablation이라고 부르는가

기존 P3/P4/P5 경로를 보존하고 C2와 업샘플한 P3를 결합한 P2 출력을 추가했다. 기존 경로를 전부 다시 구성하는 공식 YOLO P2 YAML의 완전 재현은 아니다. 두 구현 모두 새 P2의 bottom-up 피드백은 없으며, 추가 경로의 필요성을 먼저 보는 최소 변경이다.

- YOLOv8s: 기존 layer 0–21과 3개 검출 분기를 유지. P3 업샘플+C2 concat+C2f로 P2를 만들고 네 번째 검출 분기를 추가. 출력 stride 순서는 8/16/32/4.
- D-FINE-S: 기존 hybrid encoder 유지. C2 projection과 P3 업샘플을 1×1/3×3 conv로 결합하고 decoder에 네 번째 level을 추가. query 수 300 유지. P2 anchor 폭/높이 prior는 stride 비례 .025로 정의하며, 추가 level index 때문에 .40이 되지 않도록 수정.
- D-FINE decoder sampling/attention 행렬에서 기존 세 level의 per-head 가중치를 복사하고 새 P2 부분만 초기화. YOLO에서도 기존 head 가중치를 이동해 보존. 공유 tensor 일치와 P2 gradient/shape 확인 결과는 initialization_audit.json에 기록.
- P2는 특징 해상도뿐 아니라 파라미터·후보 수·decoder sampling도 늘린다. 개선이 있어도 해상도 단독 효과라고 부르지 않는다. 동일 용량 conv 대조군 및 여러 seed는 아직 미실시.

| 계열 | 기본 파라미터 | P2 파라미터 | 증가 |
|---|---:|---:|---:|
| yolo | 11,135,987 | 11,473,204 | 3.0% |
| dfine | 10,220,441 | 11,014,769 | 7.8% |

## 공통 평가 결과

AP는 0–100, Recall은 %로 표시한다. FP/영상 목표는 validation에서 고른 탐색 운영점이며 안전 기준이 아니다. 모든 영상이 양성이므로 정상 제품 오경보율도 아니다.

| 모델 | AP50 | AP75 | AP50:95 | Recall @ FP/영상≤0.1 | Recall @ FP/영상≤0.5 |
|---|---:|---:|---:|---:|---:|
| yolov8s | 91.75 | 12.12 | 36.60 | 94.37 | 95.07 |
| yolov8s_p2 | 90.18 | 12.71 | 35.99 | 64.79 | 94.37 |
| dfine_s | 92.36 | 18.29 | 35.69 | 79.58 | 95.07 |
| dfine_s_p2 | 92.95 | 16.73 | 37.25 | 87.32 | 94.37 |

**yolov8s의 P2 변화:** AP50:95 -0.62점, FP/영상≤0.1에서 Recall -29.58%p. 단일 seed와 6개 검증 날짜의 결과이므로 통계적으로 확정된 개선으로 쓰지 않는다.

**dfine_s의 P2 변화:** AP50:95 +1.56점, FP/영상≤0.1에서 Recall +7.75%p. 단일 seed와 6개 검증 날짜의 결과이므로 통계적으로 확정된 개선으로 쓰지 않는다.

## 작은 이물 조건: 원본 짧은 변 <8px

검증 세트에서 이 구간은 17 bbox다. 모델마다 0.1 FP/영상 이하를 만족하는 운영점에서 비교하며 다른 크기 구간도 metrics.json에 포함한다.

| 모델 | GT | TP | FN | Recall |
|---|---:|---:|---:|---:|
| yolov8s | 17 | 10 | 7 | 58.82% |
| yolov8s_p2 | 17 | 10 | 7 | 58.82% |
| dfine_s | 17 | 11 | 6 | 64.71% |
| dfine_s_p2 | 17 | 8 | 9 | 47.06% |

## 같은 GT에서 P2 전후가 어떻게 바뀌었는가

각 모델의 FP/영상≤0.1 운영점에서 일대일 매칭한다. 임계값은 각각 val에서 선정하므로 독립 시험에서 확정한 개선은 아니다.

| 계열 | 둘 다 검출 | P2에서 새로 검출 | P2에서 새로 누락 | 둘 다 누락 |
|---|---:|---:|---:|---:|
| yolov8s | 90 | 2 | 44 | 6 |
| dfine_s | 104 | 20 | 9 | 9 |

## 실패 분석과 해석 범위

common_eval/gt_diagnostics.csv에는 GT별 최종 예측의 최대 IoU와 대응 점수를 기록했다. 여기서는 최종 출력에서 위치 대응이 없는 경우와 대응은 있지만 점수가 낮은 경우를 나눈다. encoder 후보 소실·top-K 탈락을 직접 계측한 결과는 아니므로 no_final_iou_match를 feature 소실로 해석하지 않는다. 진단 필드는 중복 예측 간 일대일 매칭을 대체하지 않으며 실제 TP/FN은 공통 평가기의 일대일 규칙을 따른다.

AP50은 높은데 AP75가 낮으면, 후보 발견보다 작은 박스의 정확한 경계가 병목일 가능성을 조사한다. P2가 작은 구간 recall을 개선하지 않거나 AP75를 낮추면 P2를 필수 구조로 확정하지 않는다. 학습 curve와 라벨 경계·잔존 표시를 함께 검토해야 한다.

이 임시 데이터의 성능에는 복원 흔적·유사 제품 배경·날짜 간 반복 제품 가능성이 섞일 수 있다. 팀원 최종 데이터에서도 같은 원본 ID split으로 baseline과 P2를 재실험한 뒤 채택 여부를 결정한다. validation 모델 선택과 임계값 탐색을 수행했으므로 이 수치를 독립 일반화 성능으로 제시하지 않는다.

## 사용 파일

[전처리 데이터 설명](<../../../data/processed/kamp500_telea_v1/README.md>) · [원본 ID 분할표](<../../../data/processed/kamp500_telea_v1/manifest.csv>) · [학습 입력 설정](<../../../data/processed/kamp500_telea_v1_640/data.yaml>)

[실험 폴더](<../../../experiments/kamp_pilot_v1>) · [실행 상태](<../../../experiments/kamp_pilot_v1/queue_status.json>) · [초기화 검사](<../../../experiments/kamp_pilot_v1/initialization_audit.json>)

D-FINE 공식 저장소 commit과 초기 가중치 SHA256은 run_manifest.json, 패키지 버전은 requirements.lock.txt, 명령은 run_queue.py, 변환 코드는 scripts/build_kamp_pilot.py 및 build_kamp_letterbox.py에 있다. 원본 코드·데이터는 변경하지 않았다. D-FINE 추가 기능은 별도 wrapper로 적용한다.
