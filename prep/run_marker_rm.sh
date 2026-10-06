#!/bin/bash
# ① 마커 처리 요인 확장: 제거 방법 4종(rm_ns / rm_dil / rm_bgfill / rm_biharm) × seed 0~2 학습 (나머지 조건은 베이스라인과 동일)
#   A. 기존 모델 9개(인페인팅 Telea / 원본 / 마스킹) × 새 입력 4종 교차 평가
#   B. GPU 1: rm_ns, rm_dil 학습 / GPU 2: CV_f0 교사 종료 대기 → 의사 라벨 방법 비교(pl_methods_cv) → rm_bgfill, rm_biharm 학습
#   C. 전체 21개 모델 × 입력 7종 평가, 흔적 검사 6종(각 제거 방법으로 가짜 링 지우기 + 회색 링)
K=/data/knhyun/KAMP; cd $K
PY=${PY:-$K/.venv/bin/python}
log() { echo "[$(date +%H:%M:%S)] $*"; }
done_run() { grep -q "epochs completed" runs/$1/train.log 2>/dev/null; }
OLD="v2/B_s0 v2/B_s1 v2/B_s2 v2/M_marked_s0 v2/M_marked_s1 v2/M_marked_s2 v2/M_masked_s0 v2/M_masked_s1 v2/M_masked_s2"
NEW=""; for v in rm_ns rm_dil rm_bgfill rm_biharm; do for s in 0 1 2; do NEW="$NEW v2/M_${v}_s$s"; done; done
train_list() { local G=$1; shift; for v in "$@"; do for s in 0 1 2; do
  done_run v2/M_${v}_s$s && continue
  log "학습 v2/M_${v}_s$s (GPU $G)"; bash prep/run_train.sh v2/M_${v}_s$s $G $v 320 640 416 300 16 "" $s
  done_run v2/M_${v}_s$s || { log "FAIL v2/M_${v}_s$s"; tail -5 runs/v2/M_${v}_s$s/train.log; }
done; done; }

log "A. 교차 평가 (기존 9개 × 새 입력 4종)"
CUDA_VISIBLE_DEVICES=1 $PY eval/evaluate.py --runs $OLD --inputs rm_ns rm_dil rm_bgfill rm_biharm --out eval/out_v2 2>&1 | grep -E " test " 
log "B. 학습"
train_list 1 rm_ns rm_dil &
( until done_run v2/CV_f0; do sleep 30; done
  log "의사 라벨 방법 비교"; CUDA_VISIBLE_DEVICES=2 $PY eval/pseudo/pl_methods_cv.py 2>&1 | grep -vE "meshgrid|_VF|Model Summary"
  train_list 2 rm_bgfill rm_biharm ) &
wait
log "C. 평가 (21개 × 입력 7종)"
CUDA_VISIBLE_DEVICES=1 $PY eval/evaluate.py --runs $OLD $NEW --inputs clean marked masked rm_ns rm_dil rm_bgfill rm_biharm --out eval/out_v2 2>&1 | grep -E " test " &
log "흔적 검사"
for mode in inpaint mask rm_ns rm_dil rm_bgfill rm_biharm; do
  G=$([ $mode = inpaint ] || [ $mode = rm_ns ] || [ $mode = rm_bgfill ] && echo 2 || echo 1)
  TRACE_OUT=$K/eval/marker_v2 CUDA_VISIBLE_DEVICES=$G $PY eval/inpaint_trace_test.py --mode $mode $OLD $NEW > eval/marker_v2/trace_$mode.log 2>&1 &
  [ $mode = mask ] || [ $mode = rm_dil ] && wait
done
wait
log "ALL DONE"
