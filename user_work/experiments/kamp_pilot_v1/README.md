# KAMP pilot v1: D-FINE-S / YOLOv8s ± P2

데이터와 초기화 검사 완료 후 실행한 30 epoch, 단일 seed 탐색 실험이다. 실행 상태는 `queue_status.json`, 결과는 각 모델 `common_eval/metrics.json`과 프로젝트 루트의 `임시_전처리_P2_Ablation_실험_보고서.md`에서 확인한다. `watch_report.py`는 현재 큐 종료까지만 보고서를 갱신하는 유한 실행 프로세스다.

- Python: `/home/viplab/contest/.detector-venv/bin/python`
- 재현 패키지: `requirements.lock.txt` (Python 3.13, PyTorch 2.10.0 CUDA12.8)
- 데이터: `/home/viplab/contest/data/processed/kamp500_telea_v1_640`
- 원본 ID/분할: `/home/viplab/contest/data/processed/kamp500_telea_v1/manifest.csv`
- COCO 초기 가중치 해시, 공식 D-FINE commit: `run_manifest.json`
- 초기화·P2 shape/gradient 검사: `prepare_models.py`, `initialization_audit.json`

## 모델별 명령

현재 실행을 덮어쓰지 않도록 재실험은 경로와 실험 버전을 바꿔서 수행한다. 이미 실행 기록이 있는 `run_queue.py`는 재실행을 거부한다. 아래 명령은 실행 방식의 기록이며 진행 중인 큐와 동시에 다시 실행하지 않는다.

```bash
# YOLO 기본 / P2
python train_yolo.py
python train_yolo.py --p2

# D-FINE 기본 / P2 (공식 코드의 tuning 로더로 초기화만 읽음)
python train_dfine.py -c dfine_s.yml -t dfine_s_init.pth --device cuda:0 --seed 20260929
python train_dfine.py --p2 -c dfine_s_p2.yml -t dfine_s_p2_init.pth --device cuda:0 --seed 20260929

# Val만 공통 평가. Test 옵션은 제공하지 않음.
python evaluate_common.py --model yolov8s
python evaluate_common.py --model yolov8s_p2
python evaluate_common.py --model dfine_s
python evaluate_common.py --model dfine_s_p2
python summarize.py
```

D-FINE 공식 체크포인트 및 직접 생성한 초기화/학습 체크포인트만 로드한다. PyTorch 2.10의 기본 weights_only 변경과 공식 D-FINE 로더 호환을 위해 현재 큐는 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`을 지정했다. 임의 출처의 체크포인트를 이 실행 경로에 넣지 않는다.

## 실험 해석

P2 분기는 기존 P3/P4/P5를 유지하고 추가한다. YOLO는 C2f, D-FINE은 projection+fusion conv로 구현했다. 각 계열 내 전후 비교이며 두 P2 구현의 세부 구조가 동일하다는 의미는 아니다. D-FINE query 개수는 300 고정이며, 기존 attention sampling 가중치를 보존하고 P2 sampling 부분을 추가했다.

두 모델의 기존 공통 tensor는 동일 초기값이다. P2로 파라미터·연산량·후보 수가 함께 바뀌므로 순수 해상도 효과만 분리한 실험은 아니다. 작은 검증 표본·단일 seed·임시 표시 처리본이라는 한계가 있으며 최종 데이터에서 재검증해야 한다. test 78장은 이번 실험에서 평가하지 않는다.

YOLO 첫 실행은 검증 672 패딩 문제를 수정하기 전 기록으로 archive/에 격리했다. 채택하는 YOLO 재실행의 로그와 가중치는 yolov8s/에 있다. 재실행 일부 구간만 D-FINE과 GPU를 공유했으므로 wall time 비교는 하지 않는다.

## 후속 오류 분석

프로젝트 루트의 `P2_결과_해석과_다음_설계_보고서.md`와 `failure_analysis/`를 참고한다. `analyze_failures.py`와 `analyze_sensitivity.py`는 기존 val 예측만 분석한다. 추가 NMS/크기 보정 결과는 원본 결과와 분리했고 기본 모델 설정을 변경하지 않았다. `inference_postprocess.py`는 GT를 받지 않는 선택적 출력 처리 함수다. 추가 학습과 test 평가는 이 후속 단계에서 실행하지 않았다.
