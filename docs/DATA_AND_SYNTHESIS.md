# 데이터·합성 X-ray 파이프라인

## 1. 실제 KAMP 데이터

원본 BMP는 색상 사각형을 포함할 수 있습니다. 공식 안내에 따라 위치·개수의 정답은 TXT를 사용하고, 색상 사각형은 입력 특징으로 사용하지 않습니다.

1. `eda/extract.py`로 원본 파일 인벤토리와 마커 후보를 조사합니다.
2. `prep/build_dataset.py`로 마커 제거 PNG와 장비·날짜 단위 group split을 만듭니다.
3. `prep/build_handoff.py`로 `kamp_xray_v2` 패키지를 만듭니다.
4. `kamp_xray_v2/tools/verify_split.py`로 분할 누수를 확인합니다.

데이터 파이프라인 스크립트는 과거 팀 작업 경로를 기본값으로 담고 있습니다. 실행 전 다음 환경 변수를 현재 저장소의 절대 경로로 설정합니다.

```bash
export KAMP_ROOT="$(pwd)"
```

원본 BMP, 공식 TXT, 마커 제거 영상, split manifest를 임의로 공개 저장소에 올리지 않습니다.

## 2. 합성 X-ray 생성

### 옮겨심기

`prep/synth/build_replica.py`, `prep/synth/build_tp_train.py`, `prep/synth/build_synth_test.py`는 실제 결함 주변에서 추정한 광학밀도(OD) 패치를 마커 제거 배경에 곱셈 감쇠 방식으로 넣습니다. 원래 위치와 무작위 위치, 농도 배수를 비교할 수 있습니다.

### gVXR 물리 시뮬레이션

`prep/synth/gvxr_core.py`는 gVXR를 통해 구·정육면체·불규칙 조각·판·선형 형상과 SUS304·철·알루미늄·유리·돌·뼈·플라스틱의 감쇠를 생성합니다. `build_synth_test.py`는 위치·크기·형상·재질 축의 평가 전용 세트를, `build_gvxr_train.py`는 훈련용 변형을 생성합니다.

gVXR는 GPU/OpenGL 컨텍스트가 필요합니다. gVXR 공식 Python 배포판은 PyPI `gvxr` 패키지로 제공되며, 별도 가상환경에서 다음처럼 설치할 수 있습니다.

```bash
python -m venv .venv-gvxr
. .venv-gvxr/bin/activate
pip install gvxr numpy scipy pandas opencv-python
```

실행 전 `eval/synth/gvxr/calibration.csv`, erased background, 실제 결함 OD 맵과 원본 split 산출물이 준비되어야 합니다. 이 중간 산출물은 데이터·경로 의존성이 크므로 생성 스크립트와 결과를 함께 보관합니다.

## 3. 합성 결과 평가

`prep/handoff_src/synth_test/eval_synth.py`와 `eval/synth/`의 스크립트는 모델 예측을 조건별로 집계합니다. 합성 이물 중심 8 px 이내 또는 제공 평가 영역에 예측 중심이 들어오면 해당 이물을 찾은 것으로 계산합니다. 이 지표는 공식 COCO AP와 다르므로 같은 표에 섞지 않습니다.

합성 평가의 권장 순서는 다음과 같습니다.

1. 실제 train/val로 모델과 임계값을 먼저 고정합니다.
2. 합성 test에는 고정된 checkpoint와 임계값만 적용합니다.
3. 크기·위치·형상·재질·대비 조건별 도달률과 이물을 지운 배경의 경보 비율을 함께 봅니다.
4. 합성 조건을 학습에 넣은 경우에는 실제 데이터 AP 및 배경 경보와 같이 비교하고, 기존 실제 test를 독립 최종 검증으로 다시 주장하지 않습니다.

## 4. 학습 전후 보완 실험

현재 프로젝트의 보완 실험은 실제 train 350장과 합성 val 파생 120장을 섞어 6개 모델×3 seed를 학습했다. 합성 test 파생 영상은 학습에서 제외했다. 대표 MAL+UQ는 합성 도달률 42.01%에서 64.19%로 상승했지만, 실제 test AP는 37.46에서 36.27로 낮아졌다. 따라서 최종 실제 데이터 모델은 기존 MAL+UQ 가중치로 유지하고, 합성 학습은 취약 조건과 분포 균형을 진단하는 보조 결과로 사용한다.

## 출처

- gVXR: [PyPI](https://pypi.org/project/gvxr/), [official site](https://gvirtualxray.sourceforge.io/)
- Andriiashen, V., et al. (2023). *Scientific Reports*, 13, 1881.
- Andriiashen, V., et al. (2024). *Journal of Nondestructive Evaluation*, 43, 79.
