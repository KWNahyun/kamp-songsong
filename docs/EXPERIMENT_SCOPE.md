# 일괄 실행의 범위

`run_all.py`는 **최종 D-FINE-S+MAL+UQ 경로**를 통합합니다. 저장소의 과거 연구 전체를 자동 재현하는 실행기로 검증된 상태는 아닙니다.

| 실험 | 원래 코드 | 필요한 자원 | 이번 확인 상태 |
| --- | --- | --- | --- |
| 고정 MAL+UQ 추론·평가 | `repro/predict.py`, `repro/evaluate.py` | 대표 가중치, 가공 데이터 | 대표 checkpoint 실제 CPU 추론·평가 완료; 원래 val 지표와 차이 있음 |
| MAL+UQ 3개 seed 재학습 | `repro/train_mal.py`, `repro/train_uq.py` | 단일 클래스 초기 가중치, CUDA GPU | 실제 checkpoint와 train 샘플의 손실·역전파 검사 통과; 전체 GPU 학습 보류 |
| YOLOv8s/D-FINE-S, scratch/COCO/PIDray 비교 | `prep/run_nb2.sh`, `prep/run_pidray_pretrain.sh` | baseline 초기 가중치, PIDray 데이터, `data/nb2/`, CUDA GPU | 미실행; 원래 `/data/knhyun/KAMP` 경로가 남아 있음 |
| 초기 YOLOv3-SPP 비교 | `prep/run_train.sh`, `yolov3/` | 원본/가공 데이터, 초기 가중치, 과거 버전 의존성 | 미실행 |
| 마커 제거·의사 라벨 비교 | `prep/run_marker_factor.sh`, `prep/run_unlabeled_factor.sh` | 원본 BMP, GT 없는 영상, 기존 모델, 중간 산출물 | 미실행 |
| gVXR/옮겨심기 합성 스트레스 | `prep/synth/`, `eval/synth/` | gVXR/OpenGL, calibration, erased background, OD 맵, 평가 모델 | 미실행 |
| 전처리 데이터 생성 | `prep/build_dataset.py`, `prep/build_handoff.py` | 원본 BMP/TXT, EDA 인벤토리와 마커 정보 | 재생성 미실행; 기존 가공 데이터 500장 검사 통과 |

최종 제출 모델이 YOLOv8s라면 해당 가중치와 실행 경로를 추가로 확정해야 합니다. 원래 상세 보고서의 YOLOv8s 최종 모델과 새 README의 MAL+UQ 최종 모델은 동일한 결과가 아닙니다.

원래 연구의 CSV·그림·보고서는 참고 자료로 포함했습니다. 해당 파일이 존재한다는 이유로 이번 실행에서 재현되었다고 판정하지 않습니다.
