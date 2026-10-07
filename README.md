# KAMP X-ray 이물질 탐지 — 소스코드·데이터 재현 패키지

## 가장 먼저: 실행 순서 (아나콘다 기준)

**이 폴더에서 터미널 열기 → 가상환경 생성·활성화 → PyTorch 설치 → `requirements.txt` 설치 → `run_all.py` 실행 → 결과 확인**

통합 실행 파일은 **`run_all.py`**입니다. `run.sh all`도 같은 파일을 호출하지만, 아래 안내에서는 `run_all.py`를 직접 실행합니다. Anaconda Prompt 또는 터미널에서 **이 README와 `run_all.py`가 있는 폴더로 이동한 뒤**, CPU/GPU 중 사용할 환경의 명령을 위에서 아래로 실행하세요. 환경 생성과 패키지 설치는 처음 한 번만 필요하며, 이후에는 해당 환경을 활성화하고 실행 명령을 사용합니다.

### CPU에서 예측·평가하기

```bash
# 1. 아나콘다 가상환경 생성 및 활성화
conda create -n kamp-xray-cpu python=3.13 -y
conda activate kamp-xray-cpu

# 2. PyTorch와 의존성 패키지 설치
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements.txt
python -m pip check

# 3. 데이터 검사 → val/test 예측 → 평가 → CSV 저장을 한 번에 실행
python run_all.py --mode predict --device cpu --output outputs/frozen_cpu
```

### NVIDIA GPU에서 예측·평가하기

CUDA 12.8용 PyTorch를 사용할 수 있는 NVIDIA 드라이버 환경에서 실행합니다.

```bash
# 1. 아나콘다 가상환경 생성 및 활성화
conda create -n kamp-xray-gpu python=3.13 -y
conda activate kamp-xray-gpu

# 2. CUDA용 PyTorch와 의존성 패키지 설치
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check

# 3. 데이터 검사 → val/test 예측 → 평가 → CSV 저장을 한 번에 실행
python run_all.py --mode predict --device cuda:0 --output outputs/frozen_gpu
```

**학습부터 다시 실행하려면:** 위 GPU 환경의 설치·활성화를 마친 뒤, 예측 실행 명령 대신 다음을 사용합니다. MAL+UQ를 3개 seed로 재학습하고 val/test 예측·평가까지 진행합니다. 이 전체 학습 경로의 완주는 아직 검증되지 않았습니다.

```bash
python run_all.py --mode train --device cuda:0 --output outputs/retrain
```

테스트 예측 CSV는 지정한 출력 폴더의 `test/prediction/predictions.csv`, 평가 수치는 `test/metrics/metrics.json`에서 확인합니다. 재학습 결과는 `outputs/retrain/seed_<seed>/test/` 아래에 저장됩니다. 출력 폴더가 이미 있으면 중단되므로, 재실행할 때에는 `--output`에 새로운 폴더 이름을 지정하세요.

> <sub><strong>환경 참고:</strong> 같은 데이터와 가중치라도 CPU/GPU, CUDA·PyTorch 버전과 연산 방식에 따라 예측값·평가지표·재학습 결과가 달라질 수 있습니다. 결과를 비교할 때 실행 환경도 함께 확인하세요.</sub>

---

X-ray 영상에서 이물질을 찾고, 탐지 성능을 평가하는 프로젝트입니다. 소스코드와 고정 데이터 분할을 함께 담아, 설치부터 결과 확인까지 순서대로 실행할 수 있도록 정리했습니다.

**실제 테스트 예측 생성 완료.** GitHub의 두 가중치 해시를 확인하고 검증 66장·테스트 84장의 추론과 공식 COCO 평가를 끝냈습니다. 바로 볼 파일은 [테스트 예측 CSV](predictions/test_predictions.csv), [테스트 평가](predictions/test_metrics.json), [박스 표시 영상](predictions/overlays/)입니다. 원래 검증 AP와 이번 결과는 약 0.55포인트 차이가 있으며, 전체 3개 seed 재학습은 아직 미검증입니다. [검수 보고서](REVIEW_REPORT.md)에 구분해 기록했습니다.

## 1. 무엇이 들어 있나요?

| 파일·폴더 | 내용 |
| --- | --- |
| `run_all.py` | 데이터 검사, 고정 모델 예측·평가, MAL+UQ 3개 seed 재학습을 한 명령으로 실행 |
| `requirements.txt` | 최종 모델 경로의 Python 패키지 목록 |
| `repro/` | D-FINE-S + MAL + UQ 구현, 모델 설정, 고정 분할과 가중치 해시 |
| `repro/vendor/dfine/` | 원래 저장소에 지정된 D-FINE 전체 소스코드. 폴더에 포함되어 별도 다운로드 불필요 |
| `data/kamp_xray_v2/` | 마커 제거 PNG, YOLO 라벨, COCO 라벨, 고정 분할표 |
| `predictions/` | 이번에 실제 생성한 테스트 예측 JSON·CSV, 84장 요약, 평가와 박스 표시 영상 |
| `results/frozen_cpu/` | 이번 val/test 통합 실행의 전체 원본 결과와 완료 기록 |
| `verification/` | 이번 검수의 데이터 검사 결과와 설치 환경 기록 |
| `prep/`, `eda/`, `eval/`, `baseline/`, `yolov3/` | 원래 프로젝트의 전처리·분석·비교 실험 코드와 기존 산출물 |
| `docs/EXPERIMENT_SCOPE.md` | 통합 실행 범위와 별도 자원이 필요한 과거 실험 목록 |
| `REPORT.md`, `REPORT.pdf` | 원래 상세 연구 보고서. 이번에 다시 계산한 결과가 아님 |

GitHub 기준 버전은 `7dd38486dde36839708d77a387d8967efba766e3`입니다. 패키지에 적용한 수정은 `verification/source_changes.patch`에 기록했습니다.

## 2. 데이터 구성

| 분할 | 이미지 | 이물질 박스 | 용도 |
| --- | ---: | ---: | --- |
| train | 350장 | 797개 | 학습 |
| val | 66장 | 144개 | 모델 선택·검증 |
| test | 84장 | 206개 | 최종 평가 |

같은 장비·날짜 그룹이 서로 다른 분할에 섞이지 않습니다. 공식 TXT로부터 만든 YOLO/COCO 라벨을 사용하며, 색상 사각형을 제거한 흑백 PNG를 모델에 입력합니다. 분할을 임의로 다시 만들지 마세요.

`train` 모드는 학습용 640×640 데이터를 만들 때 train과 val만 읽습니다. 테스트 정답은 학습에 사용하지 않으며, 저장된 테스트 예측을 평가할 때만 읽습니다. 전체 파일 검사 단계에서는 test의 파일·라벨 일치 여부도 검사합니다.

## 3. 실행 환경 준비

이번 구성요소 검사는 **Python 3.13.5, PyTorch 2.10.0, torchvision 0.25.0, macOS CPU**에서 수행했습니다. 원래 결과를 얻은 환경은 `repro/environment/runtime.json`에 기록되어 있으며 CUDA 12.8을 사용합니다. 전체 학습에는 NVIDIA CUDA GPU가 필요합니다.

환경 생성·활성화와 설치 명령은 문서 맨 위의 **아나콘다 실행 순서**를 따르세요. CPU 예측은 `kamp-xray-cpu`, GPU 예측·재학습은 `kamp-xray-gpu` 환경을 사용합니다.

`requirements.txt`는 이미 선택한 CPU/CUDA용 torch를 덮어쓰지 않도록 torch 설치를 분리했습니다. 과거 YOLO/PIDray/gVXR 실험에는 추가 의존성과 데이터가 필요합니다. 이 파일은 최종 MAL+UQ 경로의 설치 목록입니다.

## 4. 먼저 데이터 검사

```bash
python run_all.py --mode check --output outputs/check
```

고정 분할, 그룹 누수, 파일 누락, PNG 크기, YOLO/COCO 박스 일치, 분할 간 동일 파일을 검사합니다. `outputs/check/dataset_audit.json`에 검사 결과를 저장합니다. 이 명령의 성공은 데이터 검사 통과를 의미하며, 모델 재현 성공을 의미하지 않습니다.

## 5. 고정된 최종 모델로 예측·평가

확인된 원래 가중치를 다음 위치에 포함했습니다.

```text
repro/checkpoints/maluq_seed20260930.pth
```

예상 SHA-256:

```text
ddbdbc3638e661003b49922832edb84a44c8d0b95e30d33906ff4cae984c6461
```

```bash
python run_all.py --mode predict --output outputs/frozen --device cuda:0
# CPU에서 실행한다면 --device cpu
```

이 한 명령은 데이터 검사 → val 예측 → val 평가 → test 예측 → test 평가 → CSV 내보내기를 실행합니다. 예측 단계는 정답 파일을 읽지 않습니다.

| 결과 위치 | 내용 |
| --- | --- |
| `outputs/frozen/test/prediction/predictions.json` | 테스트 전체 검출 후보. 원본 영상 좌표의 `[x, y, width, height]` |
| `outputs/frozen/test/prediction/predictions.csv` | 표로 열어 볼 수 있는 동일 예측결과 |
| `outputs/frozen/test/prediction/image_summary.json` | 모든 테스트 이미지별 검출 수와 입력 해시. 검출이 없는 영상도 포함 |
| `outputs/frozen/test/prediction/overlays/` | 고정 임계값 이상 검출 박스를 표시한 영상 |
| `outputs/frozen/test/metrics/metrics.json` | AP, AP50, AP75, TP/FP/FN, recall 등 |
| `outputs/frozen/completed.json` | 모든 단계가 성공했을 때만 생기는 완료 기록 |

예측의 클래스 ID는 `0=defect`입니다. 평가 시 파일명으로 원래 COCO 이미지 ID와 클래스 ID에 대응시킵니다. 전체 후보의 최저 점수는 0.001, NMS IoU는 0.7, 경보 표시 임계값은 원래 val에서 고정한 0.19337229430675507입니다. 점수는 보정된 불량 확률이 아닙니다.

## 6. MAL+UQ 3개 seed 재학습

다음 초기 가중치도 포함했습니다. 일반 COCO 다운로드 파일과 다른, 단일 클래스용 프로젝트 초기 가중치입니다.

```text
repro/checkpoints/dfine_s_coco_init.pth
SHA-256: 0ac6124d45341889b9999b2294540cc6105add3679fe6182ea692c9e6917f531
```

```bash
python run_all.py --mode train --output outputs/retrain --device cuda:0
```

학습용 데이터 준비 → seed 20260929·20260930·20260931 각각 MAL 30 epoch → 검출기 고정 후 UQ 30 epoch → val/test 예측·평가 → seed 평균과 표본 표준편차 집계를 순서대로 실행합니다. 학습은 새 출력 폴더에 저장됩니다.

재학습 모델의 TP/FP/recall은 원래 대표 모델의 검증 임계값을 그대로 적용해 계산하며, seed마다 임계값을 다시 최적화하지 않습니다. AP는 후보 전체를 사용합니다. 이 재학습 경로의 완주와 과거 결과 일치는 아직 검증되지 않았습니다.

모든 실행은 새 출력 폴더를 사용해야 합니다. 이미 존재하는 출력 폴더를 지정하면 중단됩니다. 실패한 실행에는 `completed.json`이 없으므로 완료 결과로 취급하지 마세요.

## 7. 이번 실제 실행 결과

| 분할 | AP | AP50 | AP75 | recall | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | --- |
| val 66장 | 38.7285 | 97.8630 | 18.2255 | 98.6111% | 142 / 8 / 2 |
| test 84장 | 38.5547 | 94.4548 | 15.0455 | 95.6311% | 197 / 16 / 9 |

AP는 COCO AP@[0.50:0.95]이며 0~100 척도입니다. TP/FP/FN과 recall은 고정 경보 임계값 0.19337229430675507, IoU 0.5에서 계산했습니다. 검출이 없는 영상이 있어도 자동 정상 판정을 의미하지 않습니다.

원래 manifest의 val AP는 39.2758, 이번 값은 38.7285입니다. 포장 과정의 수정 때문에 달라진 것인지 확인하기 위해 원래 GitHub 예측 코드의 CUDA 호출만 CPU로 바꿔 같은 환경에서 실행했습니다. 검증 후보 전체가 현재 코드와 정확히 같았습니다. 차이의 원인이 실행 장치·라이브러리 또는 원래 입력 데이터/정답의 차이인지 아직 확정하지 못했습니다.

전체 3개 seed 재학습과 GPU 대조는 미완료입니다. 과거 YOLO/PIDray 비교, 합성 스트레스 전체까지 재현한 폴더는 아닙니다. [실험 범위](docs/EXPERIMENT_SCOPE.md)를 확인하세요. 원래 REPORT §5의 YOLOv8s 최종 모델과 현재 GitHub README의 MAL+UQ 모델은 다르므로, 이번 폴더는 업로드된 MAL+UQ 대표 가중치를 기준으로 생성했습니다.

데이터·코드·가중치 출처는 `repro/THIRD_PARTY_NOTICES.txt`와 `verification/results.json`에 기록했습니다.
