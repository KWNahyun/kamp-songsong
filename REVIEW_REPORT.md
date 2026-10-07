# 재현성 검수 보고서 — 가중치 포함 재생성

검수일: 2026-10-07 (한국 시간)  
GitHub: `KWNahyun/kamp-songsong`, commit `7dd38486dde36839708d77a387d8967efba766e3`

## 현재 판정

**실제 고정 모델 예측·평가 완료. 전체 재학습과 원래 지표 일치는 미검증.**

이번에는 ZIP을 만들지 않고 이 폴더에 소스, requirements, 학습 350장·검증 66장·테스트 84장, 가중치 2개, 개선 README와 실제 테스트 예측결과를 포함했습니다.

- 테스트 예측: `predictions/test_predictions.csv`, `predictions/test_predictions.json`
- 테스트 84장 처리 기록: `predictions/image_summary.json`
- 표시 영상: `predictions/overlays/`
- 테스트 평가: `predictions/test_metrics.json`
- 통합 실행 완료: `results/frozen_cpu/completed.json`

## 가중치 검증

| 파일 | 확인한 SHA-256 | 판정 |
| --- | --- | --- |
| `maluq_seed20260930.pth` | `ddbdbc3638e661003b49922832edb84a44c8d0b95e30d33906ff4cae984c6461` | model manifest와 일치 |
| `dfine_s_coco_init.pth` | `0ac6124d45341889b9999b2294540cc6105add3679fe6182ea692c9e6917f531` | source provenance와 일치 |

## 실제 추론·공식 COCO 평가

실행 환경은 Python 3.13.5, torch 2.10.0, torchvision 0.25.0, macOS CPU입니다. 추론은 정답 파일을 읽지 않고 수행했으며, 저장된 후보를 공식 COCO 라벨에 대조해 평가했습니다. 후보 점수 0.001 이상, NMS IoU 0.7, 원래 val 경보 임계값 0.19337229430675507을 유지했습니다.

| 분할 | 이미지 | AP | AP50 | AP75 | TP | FP | FN | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 66 | 38.7285 | 97.8630 | 18.2255 | 142 | 8 | 2 | 98.6111% |
| test | 84 | 38.5547 | 94.4548 | 15.0455 | 197 | 16 | 9 | 95.6311% |

AP는 COCO AP@[0.50:0.95], 0~100 척도입니다. TP/FP/FN 및 recall은 경보 임계값을 적용하고 IoU 0.5로 대응한 값입니다. 정상 영상이 없어 normal specificity는 계산할 수 없습니다.

원래 manifest의 val AP는 **39.27580797691046**, 이번 val AP는 **38.72851832282536**입니다. 차이는 **-0.54728965408510 AP 포인트**입니다. 원래 AP75는 19.660061547449224, 이번 AP75는 18.22549493920374입니다. 원래 수치와의 완전 일치를 주장할 수 없습니다.

GitHub 원래 `predict.py`의 CUDA 호출만 CPU로 바꿔 동일 환경에서 별도로 실행했습니다. 검증 후보 13,300개의 점수·좌표·파일명 전체가 현재 패키지의 예측과 정확히 같았습니다. 따라서 이번 포장 코드 수정이 이 차이를 만들었다는 증거는 없습니다. 원래 GPU 실행 및 원래 입력·정답의 픽셀/파일 해시를 추가로 대조해야 원인을 확정할 수 있습니다. 임계값이나 정답을 변경해 점수를 맞추지 않았습니다.

## 그 밖의 검사

- 기존 검수의 데이터 500장, 고정 split hash, 37개 장비/날짜 그룹 분리, 이미지 디코딩, YOLO/COCO 좌표 일치, 분할 간 픽셀 중복 검사가 통과했습니다.
- 두 가중치가 포함된 현재 폴더의 데이터 검사도 다시 통과했습니다.
- 실제 초기 checkpoint를 로드해 실제 train 이미지의 MAL loss와 역전파를 확인했습니다.
- 실제 대표 checkpoint를 로드한 UQ loss·역전파 검사에서 selector gradient를 확인하고, 한 번의 갱신 후 검출기 tensor 전체가 동일함을 확인했습니다. 공식 평가 fixture 검사도 통과했습니다.

## 아직 완료하지 못한 범위

전체 MAL+UQ 3개 seed 학습, GPU에서의 고정 모델 대조, 과거 YOLO/PIDray/합성 스트레스 전체 실험은 완료되지 않았습니다. 원래 REPORT §5는 YOLOv8s seed 1을 최종 모델로, 현재 GitHub README는 MAL+UQ를 최종 모델로 적고 있습니다. 이번 폴더의 실제 예측은 업로드된 MAL+UQ checkpoint 기준입니다.

자동 승인 검토가 학습 데이터·소스·가중치를 연구 GPU 서버 `ailab5090-jeongwon`으로 복사하는 작업을 거절했습니다. 이유는 그 자료를 해당 서버로 내보내는 명시적 사용자 승인이 없다는 것입니다. GPU 대조와 전체 학습 검수에는 이 복사 작업에 대한 승인 또는 원래 GPU 서버에서 생성한 검증 자료가 추가로 필요합니다.
