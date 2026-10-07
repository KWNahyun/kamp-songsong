# KAMP X-ray 로컬 재현 패키지

D-FINE-S + MAL + Unary Quality, seed20260930. NMS0.7, box scale1.0.
진행 이력은 [통합 보고서](../프로젝트_진행과정_통합보고서.md), 제출 본문은 [제출용 결과보고서](../제출용_결과보고서_사용자담당.md)에서 별도로 관리한다.

## 현재 검증 범위

제공된 마커 제거 uint8 흑백 PNG → 640 letterbox → 추론 → 원본 좌표 prediction → COCO 평가.
Val66장 전체의13256개 NMS 후 prediction이 기존 결과와 좌표·score까지 정확히 일치했다.
AP39.275808, AP50 97.878244, AP75 19.660062, TP142/144, FP6이다.
본 수치는 대표 단일 seed이며3seed 평균과 다르다. test를 새로 추론하지 않았다.

원본 BMP의 마커 제거는 이 패키지에서 아직 제공하지 않는다. 팀의 전처리 구현을 받아 연결해야 한다.
마커가 포함된 입력을 단순 grayscale 변환하는 것은 허용된 입력 준비 절차가 아니다.
MAL→UQ 재학습 진입점도 포함하며 소량 배치 역전파를 검증했다. 전체 30+30 epoch 재학습은 이번 패키지 검증에서 실행하지 않았다.

## 환경

현재 확인 환경: Python3.13, PyTorch2.10.0+cu128, NVIDIA CUDA GPU.
[환경 고정 목록](environment/requirements.lock.txt), [런타임](environment/runtime.json).
현재 로컬에서는 `/home/viplab/contest/.detector-venv/bin/python`을 사용한다.
새 환경 설치 시 CUDA에 맞는 torch/torchvision을 먼저 설치하고 lock 목록의 의존성을 설치한다.
별도 가상환경 `.final-repro-venv`에서 lock 설치 및 의존성 검사와 전체 검증 추론을 통과했다. 같은 컴퓨터·GPU·드라이버 조건의 검증이며 다른 운영체제까지 보장하지 않는다.

## 추론

패키지 폴더에서 실행한다. 출력 폴더는 새 이름이어야 한다.

```bash
python predict.py --images /path/to/kamp_xray_v2/images/val --preprocessed --output outputs/my_val
```

- GT/TXT 없이 추론한다. 8bit 단채널 또는 동일값3채널의 마커 제거 영상을 입력한다.
- `predictions.json`: NMS 후 score>=.001인 모든 prediction. 원본 픽셀 xywh, class0.
- `image_summary.json`: 이미지 ID/파일명, 고정 임계값, 경보 개수, 입력 해시.
- `overlays/`: 경보 threshold 이상 박스.
- `run_metadata.json`: checkpoint 해시, 시간, 설정.
- 임계값0.19337229430675507은 과거 validation에서 정한 비교 운영점이다. 안전 확률이 아니다.
- 무검출 출력은 `no_detection_review_required`; 자동 정상 통과를 승인하지 않는다.

## 평가

```bash
python evaluate.py --predictions outputs/my_val/predictions.json --annotations /path/to/kamp_xray_v2/coco/instances_val.json --output outputs/my_val_metrics
python scripts/analyze.py
```

평가기는 파일명으로 image ID를 매핑하며 단일 defect class의 category ID를 annotation에 맞춘다.
팀원 COCO의 class1과 기존 실험의 class0을 혼동하지 않는다. 중복 파일명은 오류 처리한다.
AP에는 score floor 이후 전체 후보를 사용하고 TP/FP에는 고정 validation threshold를 적용한다.
IoU>=.5의 score순 일대일 매칭, COCO AP50:.95(maxDets100), 후보 상한300.
분석 스크립트는 포함된 **기존 val prediction**으로 FN/FP·조건표·FROC를 재생성한다.
새 threshold 선택이나 test tuning은 수행하지 않는다.

## 구성과 한계

- `manifests/model.json`: checkpoint/config 및 대표 선정 근거.
- `manifests/source_provenance.json`: 원본 코드 위치/해시. 경로는 출처 기록이며 런타임 의존성이 아니다.
- `reference_results`: 검증용 val 결과와 기존 test 요약/protocol. test를 다시 최적화하지 않는다.
- `analysis`: 오류·조건표·FROC·재현 검증.
- `vendor/dfine`: 사용한 D-FINE source와 Apache-2.0 LICENSE.
- UQ는 좌표 보정 또는 calibrated uncertainty가 아니라 IoU target의 quality re-scoring head다.
- 정상 제품 데이터가 없어 정상 오경보율/안전 자동통과를 검증하지 못했다.
- 재학습은 CUDA backward 비결정성으로 bitwise 동일 결과를 보장하지 않는다.
- 데이터/가중치의 외부 공개 권한을 이 패키지가 부여하지 않는다. 로컬 심사용이며 GitHub 업로드 안 함.


## 제출 그림과 조건 분석

`figures/`에 학습/검증 분포, 영상 특성, 조건별 재현율, 실제 성공/실패 사례, 반복 성능 비교, FROC의 PNG와 벡터 PDF가 있다. 해석과 캡션은 별도 제출 보고서를 참고한다.

`python scripts/publication_analysis.py`로 생성한다. 이 스크립트는 현재 상위 프로젝트의 `kamp_xray_v2` 영상, `experiments/kamp_v2_baselines` 정답 및 기존 성능 CSV에 의존한다. 패키지만 복사한 환경에서 독립 실행되는 범위는 고정 추론·평가이며, 원자료 기반 전체 그림 재생성까지 독립화한 것은 아니다. 이번 분석은 test를 읽지 않는다.


## 기존 MAL→UQ 학습 재현 경로

아래 명령은 사용자가 전체 재학습을 원할 때 실행한다. 이번 정리 작업에서는 전체 학습을 시작하지 않았다. 모델 구조·손실·분할·30 epoch 설정은 기존 실험을 유지한다. 기존 COCO 기반 단일 클래스 초기화 파일을 함께 보존했으며, 최종 학습 checkpoint로 검출기를 초기화하지 않는다.

```bash
python scripts/prepare_training.py --dataset /path/to/kamp_xray_v2 --output outputs/retrain/data640
python scripts/configure_training.py --data640 outputs/retrain/data640 --output outputs/retrain/train.yml
python train_mal.py -c outputs/retrain/train.yml -t checkpoints/dfine_s_coco_init.pth --device cuda:0 --seed 20260930
python train_uq.py --config outputs/retrain/train.yml --base-checkpoint outputs/retrain/mal_run/best_stg1.pth --output outputs/retrain/uq_run --seed 20260930
```

- 데이터 준비는 팀원 처리본과 split_manifest.csv의 train/val만 읽는다. test 픽셀·라벨은 읽지 않는다. 원본 BMP의 색상 표시 제거는 포함하지 않는다.
- 고정 640 정사각 letterbox와 원본 라벨 변환을 재현한다. 생성한 영상·TXT 832개와 두 COCO 정답 JSON이 기존 학습 입력과 일치했다.
- MAL은 기존 validation 기준 best_stg1 checkpoint를 사용한다. UQ는 이 detector를 고정해 30 epoch 학습한 last.pth를 사용하며 validation으로 UQ epoch를 고르지 않는다.
- UQ `--smoke`는 2배치의 실행 검사이다. smoke checkpoint를 최종 모델로 사용하지 않는다.
- `checkpoints/maluq_seed20260930.pth`와 추론 manifest는 고정 모델이다. 위 재학습 결과로 자동 교체하지 않는다. CUDA 비결정성 때문에 새 학습 결과가 기존 가중치와 완전히 같다고 보장하지 않는다.
- 기존 출력 폴더를 재사용하지 말고 새 이름을 사용한다. 준비·UQ 스크립트는 덮어쓰기를 거부한다.

## 별도 환경 검증 결과

[검증 JSON](analysis/fresh_environment_verification.json)에 환경과 판정 결과를 저장했다. MAL 8장 배치에서 유한 손실 및 488개 gradient tensor를 확인했고 optimizer 갱신은 하지 않았다. UQ 2배치에서는 head만 갱신되고 detector 가중치의 정확한 동일성을 확인했다. 별도 환경의 검증 66장·13,256개 예측은 기존 출력과 완전히 일치했다. AP 39.275808, AP75 19.660062, TP 142, FP 6이다.

환경 설치에는 로컬 패키지 캐시를 사용할 수 있으나 가상환경 자체는 새로 생성했고 기존 site-packages를 상속하지 않았다. GPU 드라이버/운영체제는 동일하다. 전체 학습의 수치 재현과 다른 하드웨어의 이식성은 이번 점검 범위가 아니다.


## 보고서 그림 및 수치 감사

`python scripts/report_synthesis.py`는 기존 결과 CSV/JSON에서 방법론·대조군 차이·운영점 전이·재검사 그림을 생성한다. `python scripts/audit_report.py`는 보고서 핵심 수치와 그림 경로를 원본 결과에 대조한다. 두 스크립트는 상위 실험 폴더에 의존하는 로컬 보고서 작업용이며 새 학습·test 추론을 실행하지 않는다. 전달용 최소 추론 경로와 구분한다. 전처리 구현 연결은 사용자가 마지막에 수행한다.


## 전달 파일 및 제출 상태

현재는 로컬 전달 후보이며 최종 제출 ZIP은 아직 만들지 않았다. [출처 안내](THIRD_PARTY_NOTICES.txt), [파일별 포함/제외 목록](manifests/delivery_inventory.csv), [제출 감사](analysis/delivery_audit.json)를 확인한다.

- `saved_test/predictions.json`: 대표 checkpoint의 **기존** test 예측을 변경 없이 복사했다. 새 추론·임계값 최적화는 하지 않았다. 원본 픽셀 xywh, class0, NMS0.7 이후 score≥0.001 후보이며 운영 임계값 적용 전 결과이다.
- `saved_test/image_index.json`: image_id와 파일명·해상도 대응. `metrics.json`은 대표 단일 seed의 기존 결과이며 3개 seed 평균이 아니다.
- `requirements.txt`: 기존 lock 파일을 참조하는 설치 진입점이다. 전체 실험 환경 목록에는 YOLO 비교용 의존성도 포함되며 최소 의존성 목록이라고 주장하지 않는다.
- `outputs/`는 점검용 자료이다. smoke checkpoint·재생성 학습 입력·중복 overlay를 그대로 최종 제출물에 포함하지 않는다. 파일은 삭제하지 않고 로컬 보존했다.
- `analysis/`와 `manifests/`에는 근거용 원래 로컬 경로가 있다. 외부 전달 전에 식별정보/블라인드 규정 검토를 한다. 이 자료를 지운 뒤 출처를 잃지 않도록 내부 원본은 보존한다.
- 데이터의 공개 재배포 허용을 소스코드 라이선스로 대체하지 않는다. 이번 작업은 공개 업로드가 아니다.

남은 필수 항목은 사용자의 전처리 최종 연결, 학습용 데이터의 코드 ZIP 포함, 팀 통합 보고서 PDF, 발표 PDF/PPT, 설문 완료화면, 최종 블라인드 점검이다. 이 항목이 끝나기 전에는 제출 완료로 표시하지 않는다.


## 탐색적 운영 분석

`python scripts/operational_validation.py`는 기존 validation 예측만으로 그룹 제외 임계값 전이, 검토 영역 포함도·면적, YOLO 보완 후보를 분석한다. 상위 실험 폴더의 YOLO 저장 출력에 의존한다. 결과는 `analysis/operational/`에 저장한다. 목표를 만족하지 못한 규칙의 집계는 비워 두고 최저 점수 진단을 별도 열에 표시한다. 고정 모델·운영 임계값을 바꾸지 않으며 새 test 평가 또는 통계적 안전 보장이 아니다. 사람 검토 시간·실제 회수율은 측정하지 않았다.


## 검사 보조 화면과 사람 검토 비교

[사용 안내](review_study/README.md)를 따라 `review_study/`만 로컬 서버로 제공한다. `inspector.html`은 분석자 미리보기, `study.html`은 참가자 비교용이다. 정답 키가 있는 `analysis/review_study_admin/`는 참가자에게 제공하지 않는다. 화면에는 제공 데이터 영상이 포함되어 있으므로 공개 재배포하지 않는다. 현재 실제 사람 실험 결과는 없으며 소프트웨어 자동 점검만 완료했다. 평가 도구는 `scripts/score_review_study.py`다. Playwright는 화면 점검에만 사용했으며 모델 추론 의존성이 아니다.
