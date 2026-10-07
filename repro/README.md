# 최종 MAL+UQ 재현 패키지

`repro/`는 대표 모델 **D-FINE-S + MAL + UQ + NMS(0.7)** 의 추론, 공식 COCO 평가, 그리고 MAL/UQ 재학습 경로를 담는다. 대표 추론 가중치와 MAL 재학습 초기화 가중치는 `checkpoints/`에 포함되어 있으며, KAMP 데이터만 별도로 준비한다.

## 준비

```bash
./repro/run.sh setup .venv
# 환경의 CUDA/드라이버에 맞는 torch와 torchvision을 먼저 설치
.venv/bin/python -m pip install -r repro/requirements.txt
```

`environment/runtime.json`은 결과를 얻은 실행 환경이고, `environment/requirements.lock.txt`는 당시 전체 패키지 기록이다. GPU 환경마다 CUDA용 PyTorch wheel이 다르므로 설치 명령은 의도적으로 분리했다.

## 입력과 가중치

- 입력은 마커 제거 이미지로 구성한 `kamp_xray_v2` 패키지다.
- 대표 가중치와 MAL 재학습 초기화 가중치는 `repro/checkpoints/`에 포함되어 있다. 파일명과 SHA-256은 `checkpoints/README.md`에서 확인할 수 있다.
- `manifests/split_manifest.csv`와 `manifests/model.json`에는 고정 split, 이미지 크기, NMS IoU, box scale, 검증 operating point가 기록돼 있다.

## 고정 모델 평가

```bash
PYTHON=.venv/bin/python ./repro/run.sh predict \
  --images /path/to/kamp_xray_v2/images/val \
  --output outputs/val_prediction

PYTHON=.venv/bin/python ./repro/run.sh evaluate \
  --predictions outputs/val_prediction/predictions.json \
  --annotations /path/to/kamp_xray_v2/coco/instances_val.json \
  --output outputs/val_metrics
```

`predict`는 이미지와 고정 가중치만 사용하며 TXT/COCO 정답을 읽지 않는다. `evaluate`가 저장된 예측과 공식 정답을 분리해 AP, AP50, AP75, recall을 계산한다. 시각화와 이미지별 경보 요약도 `predict` 출력에 함께 생성된다.

## 재학습

`prepare-data`와 `configure-train`을 먼저 실행한 뒤, 세 난수 초기값으로 MAL과 UQ를 학습한다. 각 seed의 수치와 평균·표준편차를 함께 보고, 하나의 새 가중치가 기존 대표 checkpoint를 자동으로 대체하지 않도록 한다.

```bash
PYTHON=.venv/bin/python ./repro/run.sh prepare-data --dataset /path/to/kamp_xray_v2 --output outputs/data640
PYTHON=.venv/bin/python ./repro/run.sh configure-train --data640 outputs/data640 --output outputs/train.yml
PYTHON=.venv/bin/python ./repro/run.sh train-mal --config outputs/train.yml --init repro/checkpoints/dfine_s_coco_init.pth --seed 20260930
```

합성 데이터 생성과 조건별 스트레스 평가는 저장소 최상단의 `prep/synth/`, `eval/synth/`, 그리고 [데이터·합성 파이프라인](../docs/DATA_AND_SYNTHESIS.md)에서 안내한다.
