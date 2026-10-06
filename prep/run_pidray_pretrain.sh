#!/bin/bash
# 사전학습 조건 ③ 1단계: COCO 사전학습 → PIDray 흑백 (외부 데이터). 2단계(KAMP)는 run_nb2.sh <모델> <gpu> <seed> pidray
# 사용법: run_pidray_pretrain.sh <yolov8s|dfine_s> <gpu>   → runs/nb2/pretrain_pidray_<모델>/
set -e
K=/data/knhyun/KAMP; PY=$K/.venv/bin/python; M=$1; GPU=$2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $K
if [ "$M" = yolov8s ]; then
  # 50 epoch, 배치 32, 증강 기본값 (색상·채도만 끔: 흑백). GPU 는 device 로 물리 번호 전달 (run_nb2.sh 참고)
  $PY -c "
from ultralytics import YOLO
YOLO('$K/weights/yolov8s.pt').train(data='$K/data/external/pidray_gray/data.yaml', imgsz=640, epochs=50, batch=32,
    hsv_h=0.0, hsv_s=0.0, seed=0, deterministic=True, workers=6, device=$GPU,
    project='$K/runs/nb2', name='pretrain_pidray_yolov8s', exist_ok=True, plots=True, verbose=False)
" > $K/runs/nb2/pretrain_pidray_yolov8s.log 2>&1
else
  cd $K/third_party/D-FINE
  CUDA_VISIBLE_DEVICES=$GPU $PY train.py -c $K/configs_nb2/dfine_s_pidray.yml --use-amp --seed=0 \
    -t $K/weights/dfine_s_coco.pth -u output_dir=$K/runs/nb2/pretrain_pidray_dfine_s > $K/runs/nb2/pretrain_pidray_dfine_s.log 2>&1
fi
echo "DONE pretrain_pidray $M"
