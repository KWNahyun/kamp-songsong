#!/bin/bash
# 위치 축 스트레스 (eval/synth/stress_pos.py): 모델 변형마다 기증 결함 train / val·test(_hd) 두 가지
# 사용법: run_pos_stress.sh <gpu> <모델 ...>   (모델: v3 = YOLOv3 베이스라인, yolov8s, dfine_s, yolov8s_tpB 등)
K=/data/knhyun/KAMP; G=$1; shift; cd $K
F="meshgrid|_VF|Model Summary|HGNetV2|network|Downloading"
for m in "$@"; do
  [ "$m" = v3 ] && m=""
  for dn in train heldout; do
    echo "== ${m:-yolov3} $dn $(date +%T)"
    CUDA_VISIBLE_DEVICES=$G NB2_MODEL=$m STRESS_DONOR=$dn $K/.venv/bin/python eval/synth/stress_pos.py 2>&1 | grep -vE "$F" | tail -25
  done
done
echo "== DONE $(date +%T)"
