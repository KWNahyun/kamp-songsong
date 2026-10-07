from pathlib import Path
import os, argparse
os.environ['YOLO_CONFIG_DIR']='/home/viplab/contest/experiments/kamp_pilot_v1/ultralytics_settings'
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data import build_yolo_dataset

class FixedSquareTrainer(DetectionTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        # Default validation rect=True adds 32px even to 640-square inputs.
        # Fix both train and validation geometry to the shared 640 PNG.
        return build_yolo_dataset(self.args,img_path,batch,self.data,mode=mode,rect=False,stride=32)

if __name__=='__main__':
    root=Path('/home/viplab/contest')
    parser=argparse.ArgumentParser(); parser.add_argument('--p2',action='store_true'); parser.add_argument('--seed',type=int,required=True); parser.add_argument('--name',required=True); args=parser.parse_args()
    import inspect, torch
    import ultralytics.data.build as build
    import ultralytics.models.yolo.detect.train as detect_train
    source=inspect.getsource(build.build_dataloader)
    assert '6148914691236517205 + RANK' in source
    source=source.replace('6148914691236517205 + RANK', str(args.seed)+' + RANK')
    namespace=dict(build.__dict__)
    exec(compile(source, '<seeded_build_dataloader>', 'exec'),namespace)
    detect_train.build_dataloader=namespace['build_dataloader']
    print('DATALOADER_SEED',args.seed,flush=True)
    name='yolov8s_p2' if args.p2 else 'yolov8s'
    model=YOLO(str(root/f'experiments/kamp_pilot_v1/{name}_init.pt'))
    model.train(trainer=FixedSquareTrainer,data=str(root/'experiments/kamp_v2_baselines/data640/data.yaml'),
        project=str(root/'experiments/kamp_v2_baselines/runs'),name=args.name,
        epochs=30,patience=0,imgsz=640,batch=8,device=0,workers=2,seed=args.seed,
        optimizer='AdamW',lr0=.0002,lrf=.1,weight_decay=.0001,nbs=8,
        warmup_epochs=1,warmup_bias_lr=0.0,amp=False,deterministic=True,
        mosaic=0.0,mixup=0.0,copy_paste=0.0,degrees=0.0,translate=0.0,
        scale=0.0,shear=0.0,perspective=0.0,flipud=0.0,fliplr=0.0,
        hsv_h=0.0,hsv_s=0.0,hsv_v=0.0,close_mosaic=0,multi_scale=False,
        rect=False,plots=True,save=True,exist_ok=False)
