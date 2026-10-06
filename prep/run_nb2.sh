#!/bin/bash
# 새 베이스라인(nb2) KAMP 학습: 입력 640, seed 0~2. KAMP 단계의 해상도·증강·epoch 는 조건과 무관하게 같음
# 사용법: run_nb2.sh <yolov8s|dfine_s> <gpu> <seed> [조건]
#   조건 coco    (기본) ② COCO 사전학습 → KAMP                    → runs/nb2/<모델>_s<seed>
#        scratch       ① 사전학습 없음 (무작위 초기화)             → runs/nb2/<모델>_scratch_s<seed>
#        pidray        ③ COCO → PIDray(run_pidray_pretrain.sh) → KAMP → runs/nb2/<모델>_pidray_s<seed>
#        tp*           옮겨 심기 학습 데이터(prep/synth/build_tp_train.py 의 data/nb2/tp/<조건>) + COCO 초기값 → runs/nb2/<모델>_<조건>_s<seed>
set -e
K=/data/knhyun/KAMP; PY=$K/.venv/bin/python; M=$1; GPU=$2; S=$3; C=${4:-coco}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $K
N=${M}_s$S; [ "$C" != coco ] && N=${M}_${C}_s$S
V8_DATA=$K/data/nb2/kamp_v2.yaml; V8_INIT="'$K/weights/yolov8s.pt'"; V8_PRE=True
DF_CFG=$K/configs_nb2/dfine_s_kamp.yml; DF_INIT="-t $K/weights/dfine_s_coco.pth"; DF_OPT=""
case $C in
  coco) ;;
  scratch) V8_INIT="'yolov8s.yaml'"; V8_PRE=False; DF_CFG=$K/configs_nb2/dfine_s_kamp_scratch.yml; DF_INIT="" ;;
  pidray)
    V8_INIT="'$K/runs/nb2/pretrain_pidray_yolov8s/weights/best.pt'"
    P=$K/runs/nb2/pretrain_pidray_dfine_s; W=$P/best_stg2.pth; [ -f $W ] || W=$P/best_stg1.pth
    DF_INIT="-t $W" ;;
  tp*)
    V8_DATA=$K/data/nb2/tp/$C/kamp_tp.yaml
    DF_OPT="train_dataloader.dataset.img_folder=$K/data/nb2/tp/$C train_dataloader.dataset.ann_file=$K/data/nb2/tp/$C/dfine_train.json" ;;
  *) echo "알 수 없는 조건 $C"; exit 1 ;;
esac
if [ "$M" = yolov8s ]; then
  # GPU: ultralytics 는 device 값으로 CUDA_VISIBLE_DEVICES 를 덮어쓰므로 물리 번호를 device 로 직접 넘김 (환경변수 쓰지 않음)
  # 증강: 확대·축소 0.2 (기본 0.5 는 2px 결함을 1px 로 뭉갬), 색상·채도 증강 끔 (흑백), 나머지 기본값
  $PY -c "
from ultralytics import YOLO
YOLO($V8_INIT).train(data='$V8_DATA', imgsz=640, epochs=300, batch=16, patience=300, pretrained=$V8_PRE,
    scale=0.2, hsv_h=0.0, hsv_s=0.0, seed=$S, deterministic=True, workers=4, device=$GPU,
    project='$K/runs/nb2', name='$N', exist_ok=True, plots=True, verbose=False)
" > $K/runs/nb2/$N.log 2>&1
elif [ "$M" = dfine_s ]; then
  cd $K/third_party/D-FINE
  CUDA_VISIBLE_DEVICES=$GPU $PY train.py -c $DF_CFG --use-amp --seed=$S $DF_INIT \
    -u output_dir=$K/runs/nb2/$N $DF_OPT > $K/runs/nb2/$N.log 2>&1
fi
echo "DONE $N"
