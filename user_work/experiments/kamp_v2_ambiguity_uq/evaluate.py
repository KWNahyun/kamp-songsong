"""Validation only; frozen NMS/scale, common original-pixel evaluation."""
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image
R=Path('/home/viplab/contest');E=Path(__file__).parent;B=R/'experiments/kamp_v2_baselines'
sys.path[:0]=[str(R/'models/D-FINE'),str(R/'experiments/kamp_pilot_v1')]
from src.core import YAMLConfig
from analyze_failures import nms,match,operating,coco_metrics,dumpcsv,iou
sys.path.insert(0,str(R/'experiments/kamp_v2_uq'))
from selector_patch import install_model
install_model('unary')
name=sys.argv[1];job=next(j for j in json.loads((E/'manifest.json').read_text())['jobs'] if j['name']==name);out=E/'runs_frozen'/name/'common_eval';out.mkdir(exist_ok=True)
torch.set_num_threads(4);cfg=YAMLConfig(job['config']);m=cfg.model.cuda().eval();cp=E/'runs_frozen'/name/'last.pth';ck=torch.load(cp,map_location='cpu',weights_only=False);m.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model'])
gt=json.loads((B/'original/annotations/val.json').read_text());trans=json.loads((B/'data640/transforms.json').read_text());raw=[]
for im in gt['images']:
 arr=np.asarray(Image.open(B/'data640/images/val'/im['file_name']));x=torch.from_numpy(arr.copy()).permute(2,0,1)[None].float().cuda()/255
 with torch.no_grad():p=cfg.postprocessor(m(x),torch.tensor([[640,640]],device='cuda'))[0]
 keep=p['scores']>=.001;t=trans[im['file_name']]
 for box,score in zip(p['boxes'][keep].cpu().numpy(),p['scores'][keep].cpu().numpy()):
  xx=np.clip((box[[0,2]].astype(float)-t['pad_left'])/t['scale_x'],0,im['width']);yy=np.clip((box[[1,3]].astype(float)-t['pad_top'])/t['scale_y'],0,im['height'])
  if xx[1]>xx[0] and yy[1]>yy[0]:raw.append(dict(image_id=im['id'],category_id=0,bbox=[float(xx[0]),float(yy[0]),float(xx[1]-xx[0]),float(yy[1]-yy[0])],score=float(score)))
(out/'predictions_original.json').write_text(json.dumps(raw));pp=nms(raw,.7);(out/'predictions_nms.json').write_text(json.dumps(pp));ap=coco_metrics(gt,pp);ap_raw=coco_metrics(gt,raw);ground={im['id']:[a for a in gt['annotations'] if a['image_id']==im['id']] for im in gt['images']};events=match(pp,ground);op=operating(events,.1,len(gt['images']));ids=op.pop('matched');selected=op.pop('events');op['recall']=op['TP']/144;op['FP_per_image']=op['FP']/66
manifest={r['image_id']:r for r in csv.DictReader((R/'kamp_xray_v2/split_manifest.csv').open())};images={im['id']:im for im in gt['images']};rows=[]
for a in gt['annotations']:
 ps=[p for p in raw if p['image_id']==a['image_id']];pn=[p for p in pp if p['image_id']==a['image_id']];best=max(ps,key=lambda p:iou(a['bbox'],p['bbox']),default=None);near=[p for p in ps if iou(a['bbox'],p['bbox'])>=.3];leader=max(near,key=lambda p:p['score'],default=None);bestn=max(pn,key=lambda p:iou(a['bbox'],p['bbox']),default=None);stem=Path(images[a['image_id']]['file_name']).stem
 row=dict(gt_id=a['id'],image_id=a['image_id'],machine=manifest[stem]['machine'],group=manifest[stem]['group'],width=a['bbox'][2],height=a['bbox'][3],TP=a['id'] in ids,best_iou=iou(a['bbox'],best['bbox']) if best else 0,leader_iou=iou(a['bbox'],leader['bbox']) if leader else 0,best_nms_iou=iou(a['bbox'],bestn['bbox']) if bestn else 0)
 for tag,p in [('best',best),('leader',leader)]:
  if p:
   x,y,w,h=a['bbox'];u,v,bw,bh=p['bbox'];row.update({f'{tag}_{k}':val for k,val in dict(left=u-x,top=v-y,right=u+bw-x-w,bottom=v+bh-y-h,center_x=u+bw/2-x-w/2,center_y=v+bh/2-y-h/2,width=bw-w,height=bh-h).items()})
  else:row.update({f'{tag}_{k}':None for k in ['left','top','right','bottom','center_x','center_y','width','height']})
 rows.append(row)
dumpcsv(out/'gt_diagnostics.csv',rows);dumpcsv(out/'events.csv',events)
conditions={machine:dict(GT=sum(r['machine']==machine for r in rows),TP=sum(r['machine']==machine and r['TP'] for r in rows)) for machine in ['M1','M2','M3']}
result=dict(name=name,variant=job['variant'],seed=job['seed'],AP=100*ap[0],AP50=100*ap[1],AP75=100*ap[2],raw_AP=100*ap_raw[0],raw_AP75=100*ap_raw[2],**op,conditions=conditions,precise_candidate=sum(r['best_iou']>=.75 for r in rows),precise_leader=sum(r['leader_iou']>=.75 for r in rows),precise_after_nms=sum(r['best_nms_iou']>=.75 for r in rows),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),epoch=ck.get('last_epoch',ck.get('epoch')),split='val',test_used=False,UQ=True,NMS=.7,box_scale=1,operating_point='validation-selected FP <= 6 over 66 images; not fixed operational deployment threshold')
(out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
