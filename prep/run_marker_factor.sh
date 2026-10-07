#!/bin/bash
# ① 마커 처리 요인 실험 (분할 v2, seed 0~2). 인페인팅 조건 = 베이스라인 v2/B_s* (기존)
#   원본(marked) / 마스킹(masked) 각 3회 학습 (나머지 조건은 베이스라인과 동일, 300 epoch)
#   평가: 모든 모델 × 입력 3종(인페인팅/원본/마스킹) 공식 TXT, 흔적 검사(인페인팅 링 / 회색 링)
set -e
K=/data/knhyun/KAMP; cd $K
PY=${PY:-$K/.venv/bin/python}   # 프로젝트 전용 환경
log() { echo "[$(date +%H:%M:%S)] $*"; }
wait_run() { until grep -q "epochs completed" runs/$1/train.log 2>/dev/null; do
  if grep -qE "Traceback|non-finite" runs/$1/train.log 2>/dev/null; then log "FAIL $1"; exit 1; fi; sleep 30; done; log "done $1"; }

for v in marked masked; do
  log "학습 $v"
  for s in 0 1 2; do bash prep/run_train.sh v2/M_${v}_s$s $s $v 320 640 416 300 16 "" $s & done
  wait
  for s in 0 1 2; do wait_run v2/M_${v}_s$s; done
done

RUNS="v2/B_s0 v2/B_s1 v2/B_s2 v2/M_marked_s0 v2/M_marked_s1 v2/M_marked_s2 v2/M_masked_s0 v2/M_masked_s1 v2/M_masked_s2"
log "평가 (입력 3종)"
CUDA_VISIBLE_DEVICES=0 $PY eval/evaluate.py --runs $RUNS --inputs clean marked masked --out eval/out_v2 2>&1 | grep -E " test "
log "흔적 검사"
TRACE_OUT=$K/eval/marker_v2 CUDA_VISIBLE_DEVICES=1 $PY eval/inpaint_trace_test.py --mode inpaint $RUNS 2>&1 | grep -vE "meshgrid|_VF|Model Summary" &
TRACE_OUT=$K/eval/marker_v2 CUDA_VISIBLE_DEVICES=2 $PY eval/inpaint_trace_test.py --mode mask $RUNS 2>&1 | grep -vE "meshgrid|_VF|Model Summary" &
wait
log "ALL DONE"
