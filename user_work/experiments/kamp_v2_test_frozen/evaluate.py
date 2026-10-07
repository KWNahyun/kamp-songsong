from pathlib import Path
import json,sys,hashlib
import torch,numpy as np
from PIL import Image
R=Path('/home/viplab/contest');E=Path(__file__).parent;sys.path[:0]=[str(E.parent/'kamp_pilot_v1'),str(R/'models/D-FINE'),str(E.parent/'kamp_v2_uq')]
from analyze_failures import nms,match,coco_metrics,dumpcsv
name=sys.argv[1];protocol=json.loads((E/'frozen_protocol.json').read_text());j=next(j for j in protocol['jobs'] if j['name']==name);G=json.loads((E/'test_original.json').read_text());T=json.loads((E/'transforms.json').read_text());out=E/'runs'/name;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
assert hashlib.sha256(Path(j['checkpoint']).read_bytes()).hexdigest()==j['sha256']
if name.startswith('yolo'):
 from ultralytics import YOLO
 model=YOLO(j['checkpoint'])
else:
 if j['variant']=='dfine_s_p2':
  import dfine_p2;dfine_p2.install()
 if j['variant']=='dfine_mal_UQ':
  from selector_patch import install_model
  install_model('unary')
 from src.core import YAMLConfig
 cfg=YAMLConfig(j['config']);model=cfg.model;ck=torch.load(j['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model']);model.cuda().eval()
pred=[]
for im in G['images']:
 if name.startswith('yolo'):
  q=model.predict(str(E/'images'/im['file_name']),imgsz=640,device=0,conf=.001,iou=.7,max_det=300,rect=False,verbose=False)[0];boxes=q.boxes.xyxy.cpu().numpy();scores=q.boxes.conf.cpu().numpy()
 else:
  x=torch.tensor(np.array(Image.open(E/'images'/im['file_name'])).copy()).permute(2,0,1)[None].float().cuda()/255
  with torch.no_grad():q=cfg.postprocessor(model(x),torch.tensor([[640,640]],device='cuda'))[0]
  keep=q['scores']>=.001;boxes=q['boxes'][keep].cpu().numpy();scores=q['scores'][keep].cpu().numpy()
 t=T[im['file_name']]
 for b,s in zip(boxes,scores):
  xx=np.clip((b[[0,2]].astype(float)-t['px'])/t['sx'],0,im['width']);yy=np.clip((b[[1,3]].astype(float)-t['py'])/t['sy'],0,im['height'])
  if xx[1]>xx[0] and yy[1]>yy[0]:pred.append(dict(image_id=im['id'],category_id=0,bbox=[float(xx[0]),float(yy[0]),float(xx[1]-xx[0]),float(yy[1]-yy[0])],score=float(s)))
pp=nms(pred,.7);ap=coco_metrics(G,pp);ground={im['id']:[a for a in G['annotations'] if a['image_id']==im['id']] for im in G['images']};events=[e for e in match(pp,ground) if e['score']>=j['val_threshold']];matched={e['gt_id'] for e in events if e['tp']};tp=len(matched);fp=sum(not e['tp'] for e in events);alarm={e['image_id'] for e in events};m=dict(name=name,variant=j['variant'],seed=j['seed'],AP=ap[0]*100,AP50=ap[1]*100,AP75=ap[2]*100,val_threshold=j['val_threshold'],TP=tp,FP=fp,FN=206-tp,recall=tp/206,FP_per_image=fp/84,alarm_images=len(alarm),images=84,GT=206,test_threshold_optimized=False)
conditions=[]
for machine in ['M1','M2','M3']:
 ids={im['id'] for im in G['images'] if im['machine']==machine};aa=[a for a in G['annotations'] if a['image_id'] in ids];conditions.append(dict(machine=machine,images=len(ids),GT=len(aa),TP=sum(a['id'] in matched for a in aa),FP=sum(not e['tp'] and e['image_id'] in ids for e in events)))
m['conditions']=conditions;(out/'metrics.json').write_text(json.dumps(m,indent=2));(out/'predictions.json').write_text(json.dumps(pp));dumpcsv(out/'gt_matches.csv',[dict(gt_id=a['id'],image_id=a['image_id'],matched=a['id'] in matched) for a in G['annotations']]);print(json.dumps(m),flush=True)
