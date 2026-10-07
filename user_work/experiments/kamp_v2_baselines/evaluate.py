import sys,json
from pathlib import Path
E=Path(__file__).parent;P=E.parent/'kamp_pilot_v1'
sys.path.insert(0,str(P))
import evaluate_common as c
import torch,numpy as np
from PIL import Image
name=sys.argv[1]
c.EXP=E/'runs'
c.DATA=E/'data640'
c.ORIG=E/'original'
def predict(name,ims):
 if name.startswith('yolo'):
  from ultralytics import YOLO
  m=YOLO(str(c.EXP/name/'weights/best.pt'))
  for im in ims:
   r=m.predict(str(c.DATA/'images/val'/im['file_name']),imgsz=640,device=0,conf=.001,iou=.7,max_det=300,rect=False,verbose=False)[0]
   yield im,r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy()
 else:
  if '_p2_' in name:
   import dfine_p2;dfine_p2.install()
  from src.core import YAMLConfig
  cfg=YAMLConfig(str(E/'configs'/f'{name}.yml'));m=cfg.model
  state=torch.load(c.EXP/name/'best_stg1.pth',map_location='cpu',weights_only=False)
  m.load_state_dict(state['ema']['module'] if 'ema' in state else state['model']);m.cuda().eval()
  for im in ims:
   x=torch.from_numpy(np.array(Image.open(c.DATA/'images/val'/im['file_name'])).copy()).permute(2,0,1)[None].float().cuda()/255
   with torch.no_grad():r=cfg.postprocessor(m(x),torch.tensor([[640,640]],device='cuda'))[0]
   keep=r['scores']>=.001
   yield im,r['boxes'][keep].cpu().numpy(),r['scores'][keep].cpu().numpy()
c.predict=predict;c.main(name)
