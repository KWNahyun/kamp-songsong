#!/bin/bash
# 사용법: run_train.sh <name> <gpu> <data> <img_min> <img_max> <img_test> [epochs] [batch] [init_weights|none] [seed]
# init_weights 기본값 = COCO 사전학습(외부 데이터), 'none' = 무작위 초기화(처음부터 학습)
# (effective batch 는 train.py 가 accumulate 로 64 에 맞춤)
set -e
NAME=$1; GPU=$2; VARIANT=$3; IMIN=$4; IMAX=$5; ITEST=$6; EPOCHS=${7:-300}; BS=${8:-16}; INIT=${9:-}; SEED=${10:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=/data/knhyun/KAMP
PY=${PY:-$K/.venv/bin/python}   # 프로젝트 전용 환경 (numpy 1.26.4, torch 2.7.1+cu118)
[ -z "$INIT" ] && INIT=$K/weights/yolov3-spp-coco.pt
[ "$INIT" = none ] && INIT=''
mkdir -p $K/runs/$NAME/weights && cd $K/runs/$NAME
CUDA_VISIBLE_DEVICES=$GPU $PY $K/yolov3/train.py \
  --cfg $K/yolov3/yolov3-spp.cfg --data $K/data/$VARIANT.data \
  --weights "$INIT" \
  --epochs $EPOCHS --batch-size $BS --img-size $IMIN $IMAX $ITEST \
  --cache-images --single-cls --seed $SEED > train.log 2>&1
echo "DONE $NAME"
