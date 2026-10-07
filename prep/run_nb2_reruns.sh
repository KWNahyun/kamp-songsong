#!/bin/bash
# 새 베이스라인으로 기존 실험 다시 돌리기 (추론만): 기본 평가 → 마커 흔적 검사 → 실패 조건 → 합성 검증 V3·V4 → 스트레스 곡선
# 사용법: run_nb2_reruns.sh <yolov8s|dfine_s> <gpu>   (해당 모델 seed 3개 학습이 끝난 뒤)
set -e
K=/data/knhyun/KAMP; PY=$K/.venv/bin/python; M=$1; G=$2
export NB2_MODEL=$M CUDA_VISIBLE_DEVICES=$G
cd $K; L=$K/eval/out_nb2/rerun_$M.log; mkdir -p $K/eval/out_nb2 $K/eval/marker_nb2
F="meshgrid|_VF|Model Summary|HGNetV2|network|Downloading|%\|"
echo "== 1 기본 평가 $(date +%T)" | tee $L
$PY eval/nb2_eval.py $M 2>&1 | grep -vE "$F" | tee -a $L
echo "== 2 마커 흔적 검사 $(date +%T)" | tee -a $L
TRACE_OUT=$K/eval/marker_nb2/$M $PY eval/inpaint_trace_test.py --mode inpaint 2>&1 | grep -vE "$F" | tee -a $L
echo "== 3 실패 조건 $(date +%T)" | tee -a $L
$PY eval/failure/collect_heldout.py 2>&1 | grep -vE "$F" | tee -a $L
$PY eval/failure/failure_analysis.py 2>&1 | grep -vE "$F" | tee -a $L
echo "== 4 합성 검증 V3·V4 $(date +%T)" | tee -a $L
$PY eval/synth/validate_replica.py --infer 2>&1 | grep -vE "$F" | tee -a $L
echo "== 5 스트레스 곡선 $(date +%T)" | tee -a $L
$PY eval/synth/stress_v0.py 2>&1 | grep -vE "$F" | tee -a $L
echo "== DONE $M $(date +%T)" | tee -a $L
