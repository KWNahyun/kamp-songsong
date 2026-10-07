"""Frozen KAMP v2 inference on marker-removed grayscale images. No GT is read."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import cv2,numpy as np,torch
R=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['dfine_mal_UQ'],default='dfine_mal_UQ');p.add_argument('--seed',type=int,choices=[20260930],default=20260930);p.add_argument('--preprocessed',action='store_true',required=True,help='Confirm inputs are the marker-removed grayscale images, not marked originals');a=p.parse_args()
 paths=sorted([a.images] if a.images.is_file() else [p for p in a.images.iterdir() if p.suffix.lower() in ['.png','.bmp','.jpg','.jpeg']]);assert paths,'No input images';assert not a.output.exists(),'Use a new output directory to preserve earlier outputs';a.output.mkdir(parents=True);(a.output/'overlays').mkdir()
 j=json.loads((R/'manifests/model.json').read_text());name=j['name'];assert a.variant=='dfine_mal_UQ' and a.seed==j['seed'];j['checkpoint']=str(R/j['checkpoint']);j['config']=str(R/j['config']);assert hashlib.sha256(Path(j['checkpoint']).read_bytes()).hexdigest()==j['sha256']
 sys.path[:0]=[str(R/'vendor/dfine'),str(R/'src')]
 from metrics import nms
 torch.set_num_threads(4)
 if a.variant.startswith('yolo'):
  from ultralytics import YOLO
  model=YOLO(j['checkpoint'])
 else:
  if a.variant=='dfine_s_p2':
   import dfine_p2;dfine_p2.install()
  if a.variant=='dfine_mal_UQ':
   from selector_patch import install_model
   install_model('unary')
  from src.core import YAMLConfig
  cfg=YAMLConfig(j['config']);model=cfg.model;ck=torch.load(j['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model']);model.to('cpu').eval()
 start=time.time();allpred=[];summaries=[]
 for iid,path in enumerate(paths,1):
  original=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
  if original is None or original.dtype!=np.uint8:raise ValueError(f'Expected decodable uint8 image: {path}')
  if original.ndim==3:
   if original.shape[2]!=3 or not np.array_equal(original[:,:,0],original[:,:,1]) or not np.array_equal(original[:,:,0],original[:,:,2]):raise ValueError('Marked/color input is outside the frozen preprocessing contract: '+str(path))
   gray=original[:,:,0]
  elif original.ndim==2:gray=original
  else:raise ValueError('Unexpected image channels')
  h,w=gray.shape;nw,nh=round(w*640/max(w,h)),round(h*640/max(w,h));px,py=(640-nw)//2,(640-nh)//2;canvas=np.full((640,640,3),114,np.uint8);canvas[py:py+nh,px:px+nw]=cv2.resize(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),(nw,nh),interpolation=cv2.INTER_LINEAR)
  if a.variant.startswith('yolo'):
   q=model.predict(canvas,imgsz=640,device=0,conf=.001,iou=.7,max_det=300,rect=False,verbose=False)[0];boxes=q.boxes.xyxy.cpu().numpy();scores=q.boxes.conf.cpu().numpy()
  else:
   x=torch.from_numpy(canvas.copy()).permute(2,0,1)[None].float().to('cpu')/255
   with torch.no_grad():q=cfg.postprocessor(model(x),torch.tensor([[640,640]],device='cpu'))[0]
   valid=q['scores']>=.001;boxes=q['boxes'][valid].cpu().numpy();scores=q['scores'][valid].cpu().numpy()
  preds=[]
  for b,score in zip(boxes,scores):
   xx=np.clip((b[[0,2]].astype(float)-px)/(nw/w),0,w);yy=np.clip((b[[1,3]].astype(float)-py)/(nh/h),0,h)
   if xx[1]>xx[0] and yy[1]>yy[0]:preds.append(dict(image_id=iid,file_name=path.name,category_id=0,bbox=[float(xx[0]),float(yy[0]),float(xx[1]-xx[0]),float(yy[1]-yy[0])],score=float(score)))
  preds=nms(preds,.7);alarms=[q for q in preds if q['score']>=j['val_threshold']];allpred.extend(preds);overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
  for q in alarms:
   x,y,bw,bh=q['bbox'];cv2.rectangle(overlay,(round(x),round(y)),(round(x+bw),round(y+bh)),(0,130,255),1);cv2.putText(overlay,f"{q['score']:.3f}",(max(0,round(x)),max(10,round(y)-3)),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,80,220),1,cv2.LINE_AA)
  cv2.imwrite(str(a.output/'overlays'/f'{iid:04d}_{path.stem}.png'),overlay);summaries.append(dict(image_id=iid,file_name=path.name,width=w,height=h,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),threshold=j['val_threshold'],detections=len(alarms),status='detections_present' if alarms else 'no_detection_review_required',auto_pass_authorized=False))
  print(json.dumps(summaries[-1]),flush=True)
 (a.output/'predictions.json').write_text(json.dumps(allpred,indent=2));(a.output/'image_summary.json').write_text(json.dumps(summaries,indent=2));(a.output/'run_metadata.json').write_text(json.dumps(dict(run=name,checkpoint_sha256=j['sha256'],val_threshold=j['val_threshold'],images=len(paths),seconds=time.time()-start,GT_used=False,preprocessing='provided marker-removed uint8 grayscale; original-resolution input; 640 central letterbox',NMS=.7,box_scale=1,score_floor=.001,score_is_calibrated_probability=False),indent=2))
if __name__=='__main__':main()
