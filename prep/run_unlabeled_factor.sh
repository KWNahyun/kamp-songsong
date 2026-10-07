#!/bin/bash
# 실험 ② GT 없는 데이터 (분할 v2) 전체 실행: 베이스라인 B(seed 0~2) 완료 대기 → 평가 → 의사 라벨 → 학생 6개 → 학습량 대조군 3개 → 평가
# 사전 준비: prep/build_dataset.py, prep/build_unlabeled.py, 베이스라인 학습(v2/B_s0~2)
# 학생: B 설정 + 의사 라벨 (합성 증강 없음), 100 epoch (데이터 약 4배 → 이미지 처리량은 B 300 epoch 의 약 1.4배)
set -e
K=/data/knhyun/KAMP; cd $K
PY=${PY:-$K/.venv/bin/python}   # 프로젝트 전용 환경
log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_run() {  # $1 = run 이름
  until grep -q "epochs completed" runs/$1/train.log 2>/dev/null; do
    if grep -qE "Traceback|non-finite" runs/$1/train.log 2>/dev/null; then log "FAIL $1"; exit 1; fi
    sleep 30
  done
  log "done $1"
}

for s in 0 1 2; do wait_run v2/B_s$s; done
log "B 평가"
CUDA_VISIBLE_DEVICES=0 $PY eval/evaluate.py --runs v2/B_s0 v2/B_s1 v2/B_s2 --inputs clean --out eval/out_v2 2>&1 | grep -E "clean  (val|test)"

log "의사 라벨 생성 (교사 = 각 seed 의 B)"
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=0 $PY prep/build_pseudo_v2.py --teacher v2/B_s$s --tag s$s 2>&1 | grep -vE "meshgrid|_VF|Model Summary"; done

for p in p990 p1000; do
  log "학생 학습 $p"
  for s in 0 1 2; do bash prep/run_train.sh v2/U1_${p}_s$s $s pl_v2_s${s}_$p 320 640 416 100 16 "" $s & done
  wait
  for s in 0 1 2; do wait_run v2/U1_${p}_s$s; done
done

log "학습량 대조군 (GT train 반복, 학생 p990 과 같은 목록 크기·epoch)"
$PY prep/build_ctrl_v2.py
for s in 0 1 2; do bash prep/run_train.sh v2/U0ctrl_s$s $s ctrl_v2_s$s 320 640 416 100 16 "" $s & done
wait
for s in 0 1 2; do wait_run v2/U0ctrl_s$s; done

log "평가"
RUNS=""; for c in B U0ctrl U1_p990 U1_p1000; do for s in 0 1 2; do RUNS="$RUNS v2/${c}_s$s"; done; done
CUDA_VISIBLE_DEVICES=0 $PY eval/evaluate.py --runs $RUNS --inputs clean --out eval/out_v2 2>&1 | grep -E "clean  (val|test)"
CUDA_VISIBLE_DEVICES=0 $PY eval/pseudo/pl_quality_v2.py v2/B_s0 v2/B_s1 v2/B_s2 2>&1 | grep -vE "meshgrid|_VF"
CUDA_VISIBLE_DEVICES=0 $PY eval/pseudo/eval_unlabeled.py $RUNS --roles data/splits/unlabeled_roles.csv --metrics eval/out_v2 --out eval/pseudo_v2 2>&1 | grep -vE "meshgrid|_VF|Model Summary|완료"
log "ALL DONE"
