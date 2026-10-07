# 사용자 연구 브랜치

모든 작업 코드·보고서·그림과 예시 사진은 **[user_work/README.md](user_work/README.md)**에서 확인하세요.

---

# KAMP X-ray foreign-object detection

KAMP 제6회 경진대회 ④번, **X-ray 영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석**을 위한 통합 코드 저장소입니다. 데이터 처리와 합성 X-ray 생성 파이프라인, YOLO/D-FINE 비교 실험, 최종 D-FINE-S + MAL + UQ 추론·평가 경로를 한 저장소에서 관리합니다.

> 공식 TXT는 학습·평가 정답입니다. BMP의 색상 사각형은 이물 자체의 신호가 아니므로 입력 특징으로 사용하지 않습니다. 마커를 제거한 이미지를 사용하고, 전처리 방법과 영향을 보고서에 기록합니다.

## 빠른 시작: 고정 최종 모델 추론과 평가

아래는 대회 결과를 재현하는 가장 짧은 경로입니다. 대표 모델 가중치 `repro/checkpoints/maluq_seed20260930.pth`와 MAL 재학습 초기화 가중치 `repro/checkpoints/dfine_s_coco_init.pth`는 저장소에 포함되어 있다. KAMP 원본 데이터와 마커 제거 입력 영상은 대회 제공 조건에 따라 별도로 준비한다.

```bash
# 1) 저장소와 Python 환경 준비
git clone https://github.com/KWNahyun/kamp-songsong.git
cd kamp-songsong
./repro/run.sh setup .venv
# CUDA/드라이버에 맞는 torch·torchvision을 먼저 설치한 뒤:
.venv/bin/python -m pip install -r repro/requirements.txt

# 2) 마커 제거 PNG를 입력해 추론
PYTHON=.venv/bin/python ./repro/run.sh predict \
  --images /path/to/kamp_xray_v2/images/val \
  --output outputs/val_prediction

# 4) 저장된 예측을 공식 COCO 라벨로 평가
PYTHON=.venv/bin/python ./repro/run.sh evaluate \
  --predictions outputs/val_prediction/predictions.json \
  --annotations /path/to/kamp_xray_v2/coco/instances_val.json \
  --output outputs/val_metrics
```

`predict`는 TXT를 읽지 않습니다. `evaluate` 단계에서만 TXT에서 변환한 COCO 정답을 읽습니다. 출력에는 전체 후보(`predictions.json`), 경보 요약(`image_summary.json`), 시각화(`overlays/`)와 실행 설정이 저장됩니다.

## 저장소 구성

| 경로 | 역할 |
| --- | --- |
| `prep/` | 원본 인벤토리, 마커 제거, group split, KAMP 전달 패키지 생성 |
| `prep/synth/` | 옮겨심기·gVXR 기반 합성 X-ray 생성 |
| `eval/` | 공통 평가, 조건별 실패·임계값·합성 스트레스 분석 |
| `eda/` | 데이터 특성·중복·마커·결함 다양성 분석 |
| `baseline/`, `yolov3/` | 초기 YOLO 기준선과 참고 구현 |
| `repro/` | 최종 D-FINE-S + MAL + UQ 고정 모델의 추론·평가·재학습 패키지 |
| `REPORT.md` | 나현 데이터·합성 파이프라인의 상세 연구 기록 |
| `docs/` | 재현성·데이터 경로·실행 순서 안내 |

## 데이터 준비와 마커 제거

`prep/build_dataset.py`는 원본 BMP에서 색상 마커를 제거하고, 장비·날짜 그룹이 train/val/test에 섞이지 않도록 고정 분할을 만듭니다. `prep/build_handoff.py`는 학습용 PNG, 공식 TXT, YOLO와 COCO 형식 라벨, split manifest를 포함한 `kamp_xray_v2` 패키지를 생성합니다.

```bash
# 원본 자료와 EDA 산출물이 준비된 작업 루트에서
export KAMP_ROOT="$PWD"
python prep/build_dataset.py
python prep/build_handoff.py
```

실행 전 [데이터 파이프라인 안내](docs/DATA_AND_SYNTHESIS.md)를 읽으세요. 원본 데이터는 `.gitignore`에 의해 저장소에 포함되지 않습니다.

## 합성 X-ray 생성과 조건별 스트레스 테스트

`prep/synth/`에는 실제 결함 패치를 옮겨심는 방식과 gVXR 기반 물리 시뮬레이션을 이용해 위치·크기·형상·재질·대비를 바꾸는 코드가 있습니다.

```bash
export KAMP_ROOT="$PWD"
# gVXR와 원본 calibration/erased background 산출물이 준비된 뒤 실행
python prep/synth/build_synth_test.py
```

합성 test는 실제 모델의 최종 점수를 대체하지 않습니다. 고정된 실제 검증 임계값을 적용해 취약 조건을 찾고, 임계값 변화에 따른 경보 부담을 분석하는 용도입니다. 자세한 생성 의존성·평가 방법·학습 전후 비교는 [데이터·합성 파이프라인](docs/DATA_AND_SYNTHESIS.md)을 참고하세요.

## MAL+UQ 재학습

```bash
PYTHON=.venv/bin/python ./repro/run.sh prepare-data \
  --dataset /path/to/kamp_xray_v2 \
  --output outputs/data640
PYTHON=.venv/bin/python ./repro/run.sh configure-train \
  --data640 outputs/data640 \
  --output outputs/train.yml
PYTHON=.venv/bin/python ./repro/run.sh train-mal \
  --config outputs/train.yml \
  --init repro/checkpoints/dfine_s_coco_init.pth \
  --seed 20260930
PYTHON=.venv/bin/python ./repro/run.sh train-uq \
  --config outputs/train.yml \
  --base-checkpoint outputs/mal_run/best_stg1.pth \
  --output outputs/uq_run \
  --seed 20260930
```

MAL은 D-FINE의 기존 matching을 유지하면서 DEIM의 Matchability-Aware Loss를 적용한 구현입니다. UQ는 최종 query 특징·경계 분포 통계·기존 점수·박스 기하 정보를 사용해 후보별 위치 품질을 예측하고 검출 점수를 재조정합니다. 세 seed 결과를 함께 보고하며, 재학습 가중치가 배포용 대표 checkpoint를 자동으로 교체하지 않습니다.

## 재현성 범위

- 고정 대표 가중치의 추론·평가 경로는 마커 제거 입력과 지정한 설정에서 재현하도록 구성했습니다.
- 대표 checkpoint와 MAL 재학습 초기화 checkpoint는 저장소에 포함한다. KAMP 데이터, 원본 BMP, 합성 이미지와 중간 산출물은 공개 권한을 확인하지 못했으므로 저장소에 포함하지 않는다.
- D-FINE은 Apache-2.0, DEIM/MAL은 해당 upstream 고지에 따라 출처를 기록했습니다. 자세한 사항은 [`repro/THIRD_PARTY_NOTICES.txt`](repro/THIRD_PARTY_NOTICES.txt)를 참고하세요.
- GPU, CUDA, torch 버전은 `repro/environment/`에 기록했습니다. 전체 학습은 CUDA 연산 특성상 bitwise 동일성을 보장하지 않으므로 seed 3회 평균과 표준편차로 비교합니다.

## 핵심 참고문헌

- Peng et al. (2025). *D-FINE: Redefine Regression Task of DETRs as Fine-Grained Distribution Refinement*. ICLR.
- Huang et al. (2025). *DEIM: DETR with Improved Matching for Fast Convergence*. CVPR.
- Andriiashen et al. (2023). *CT-based Data Generation for Foreign Object Detection on a Single X-ray Projection*. Scientific Reports.
- Andriiashen et al. (2024). *X-ray Image Generation as a Method of Performance Prediction for Real-time Inspection*. Journal of Nondestructive Evaluation.

각 논문 아이디어와 프로젝트 구현 범위는 보고서에서 구분해 설명합니다.
