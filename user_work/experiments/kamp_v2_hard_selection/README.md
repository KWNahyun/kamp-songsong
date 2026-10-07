# 고정 특징 기반 상대 선택 목적 통제 실험

2026-10-02. detector는 고정하고 기존 UQ head를 15 epoch 추가 학습한다. 기본/MAL × 3 seed × quality/hard_pair/listwise = 18 runs. Train350, val66, test 미사용.

- 입력: 앞선 후보 진단에서 저장한 283차원 특징. 동일 모델 출력 검증 완료.
- head: 기존 UQ와 같은 62,007 parameters, 각 parent/seed의 기존 UQ 가중치로 시작. 처음 val 영상의 기존 출력 재현 assertion.
- AdamW lr1e-4 wd1e-4 batch16, 15 epoch 마지막 가중치 고정. 세 목적 사이 초기값·자료 순서 동일. validation epoch/하이퍼파라미터 탐색 없음.
- quality: 기존 modulated quality BCE만 계속 학습. 추가 학습량 대조군.
- hard_pair: quality + .1×pair logistic. 두 후보 GT IoU≥.5, 차이≥.05, 같은 GT 할당. 이미지당 최대64쌍, 고정 랜덤추출.
- listwise: quality + .1×list cross entropy. 목표 softmax(IoU/.1). credible candidate(max IoU≥.5)가 있는 예측 클러스터를 학습한다.
- 후보 클러스터는 GT 없이 원래 score 순 greedy leader 및 predicted-box IoU≥.7로 생성. 각 그룹 최고점수32개까지. Train GT는 target과 감독 대상 결정에만 사용.
- 추론에는 GT/cluster를 요구하지 않으며 unary head의 sqrt(base score×quality)를 사용한다. 따라서 신규 relation architecture가 아니라 감독 목적의 통제 실험이다.
- 평가: 기존 base score≥.001인 유효 고정 후보 support, 새 score≥.001, 원본 좌표, NMS .7, scale1.0. 추가 head는 bbox를 수정하지 않는다.
- AP75, AP 및 FP≤6 TP, oracle GT 인접 최고score 대표의 IoU 변화 확인. validation은 이미 반복 분석에 사용됐다. 정상 제품 안전율 주장 금지.
- 보조 loss 계수 .1은 사전 고정이며 loss 간 gradient 크기를 같게 맞춘 실험은 아니다. 부정적 결과는 다른 계수/구조의 불가능성을 뜻하지 않는다.

로그: `run.log`, 각 run의 `train.jsonl`. 결과 `results.json`, 완료 `completion.json`.
