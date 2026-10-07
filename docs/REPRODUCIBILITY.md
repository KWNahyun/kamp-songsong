# 대회 제출용 재현성 안내

대회 제출에서는 데이터와 가중치의 공유 권한을 보존하면서도, 제3자가 동일한 절차와 지표를 따라갈 수 있어야 한다. 이 저장소는 다음 정보를 코드와 manifest로 분리해 제공한다.

| 항목 | 저장 위치 | 재현 시 확인 방법 |
| --- | --- | --- |
| 데이터 분할 | `repro/manifests/split_manifest.csv` | 장비·날짜 group이 고정 split에 따라 분리되는지 확인 |
| 전처리 | `prep/build_dataset.py`, `prep/build_handoff.py` | BMP의 색상 마커를 입력 특징으로 사용하지 않고 마커 제거 이미지 생성 |
| 모델·후처리 | `repro/configs/`, `repro/manifests/model.json` | D-FINE-S, MAL, UQ, NMS IoU 0.7, box scale 1.0 확인 |
| 가중치 식별 | `repro/checkpoints/README.md` | 승인된 checkpoint의 SHA-256을 대조 |
| 실행 환경 | `repro/environment/runtime.json`, `requirements.lock.txt` | Python, PyTorch, CUDA, 주요 라이브러리 확인 |
| 추론·공식 평가 | `repro/run.sh`, `predict.py`, `evaluate.py` | 이미지→예측 저장→COCO 평가를 분리해 실행 |

## 권장 검증 순서

1. 승인된 마커 제거 데이터와 대표 checkpoint의 해시를 확인한다.
2. `predict`로 정답을 읽지 않는 예측 파일을 만든다.
3. `evaluate`로 공식 COCO 정답에 대해 AP/AP50/AP75/recall을 계산한다.
4. 결과 수치가 보고서의 대표 checkpoint 결과와 차이 날 경우 CUDA·torch 버전, 입력 이미지의 해시, split manifest와 NMS 설정을 먼저 비교한다.

고정 checkpoint의 추론·평가 재현과 전체 재학습의 bitwise 재현은 구분한다. CUDA의 비결정적 연산 특성 때문에 재학습은 동일 프로토콜 및 seed 반복으로 재현성을 확인한다.
