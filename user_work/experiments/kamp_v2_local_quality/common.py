"""Cached frozen features: matched quality continuation vs hard-pair/listwise losses."""
from pathlib import Path
import sys,json,time,csv,contextlib,io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou
R=Path('/home/viplab/contest');E=Path(__file__).parent;U=E.parent/'kamp_v2_uq';C=E.parent/'kamp_v2_candidate_probe'
sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import coco_metrics,nms,match,operating
torch.set_num_threads(4)
G={s:json.loads((U/f'data640/annotations/{s}.json').read_text()) for s in ['train','val']}
original=json.loads((U/'original/annotations/val.json').read_text());trans=json.loads((U/'data640/transforms.json').read_text());ground={im['id']:[a for a in original['annotations'] if a['image_id']==im['id']] for im in original['images']}
class Head(nn.Module):
 def __init__(self):
  super().__init__();self.embedding=nn.Sequential(nn.LayerNorm(283),nn.Linear(283,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU());self.quality_head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
 def forward(self,x):return self.quality_head(self.embedding(x)).squeeze(-1)
def load(name,split):
 records=[];rng=np.random.default_rng(441)
 for im in G[split]['images']:
  z=np.load(C/name/f"{split}_{im['id']}.npz");x=z['x'];b=torch.tensor(z['boxes']);s=z['score'];xy=torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1)
  anns=[a for a in G[split]['annotations'] if a['image_id']==im['id']];g=torch.tensor([a['bbox'] for a in anns]);g[:,2:]+=g[:,:2];g/=640
  ov=torch.nan_to_num(box_iou(g,xy)).numpy();y=ov.max(0);assign=ov.argmax(0)
  valid=np.isfinite(x).all(1)&np.isfinite(z['boxes']).all(1)&(z['boxes'][:,2:]>0).all(1)&(s>=.001)
  pp=torch.nan_to_num(box_iou(xy,xy)).numpy();remaining=list(np.where(valid)[0][np.argsort(-s[valid],kind='stable')]);groups=[];pairs=[]
  # Greedy seed clusters are based solely on predicted boxes and original scores.
  while remaining:
   leader=remaining[0];members=[j for j in remaining if pp[leader,j]>=.7];remaining=[j for j in remaining if j not in members]
   if not members: raise RuntimeError('Invalid cluster')
   members=members[:32]
   if len(members)>=2:
    groups.append(members)
    aa,bb=np.triu_indices(len(members),1);a=np.array(members)[aa];bb=np.array(members)[bb]
    k=(y[a]>=.5)&(y[bb]>=.5)&(abs(y[a]-y[bb])>=.05)&(assign[a]==assign[bb]);pairs.extend(zip(a[k],bb[k]))
  if len(pairs)>64:pairs=[pairs[i] for i in rng.choice(len(pairs),64,replace=False)]
  records.append(dict(x=x,b=z['boxes'],s=s,uq=z['uq'],y=y,ov=ov,valid=valid,groups=groups,pairs=pairs,im=im))
 return records
def tensors(records):
 X=torch.tensor(np.stack([r['x'] for r in records]),device='cuda');Y=torch.tensor(np.stack([r['y'] for r in records]),device='cuda')
 pi=np.zeros((len(records),64,2),int);pm=np.zeros((len(records),64),bool)
 maxg=max(len(r['groups']) for r in records);gi=np.zeros((len(records),maxg,32),int);gm=np.zeros(gi.shape,bool)
 for i,r in enumerate(records):
  for j,p in enumerate(r['pairs']):pi[i,j]=p;pm[i,j]=True
  for j,g in enumerate(r['groups']):gi[i,j,:len(g)]=g;gm[i,j,:len(g)]=True
 return X,Y,*[torch.tensor(a,device='cuda') for a in [pi,pm,gi,gm]]
def evaluate(head,records,base,seed,mode,out):
 predictions=[];diagnostics=[]
 for r in records:
  with torch.no_grad():q=head(torch.tensor(r['x'],device='cuda')).sigmoid().cpu().numpy()
  scores=np.sqrt(r['s']*q);im=r['im'];t=trans[im['file_name']];orig=next(i for i in original['images'] if i['id']==im['id']);b=r['b']
  for j in np.where(r['valid']&(scores>=.001))[0]:
   x1,y1=(b[j,:2]-b[j,2:]/2)*640;x2,y2=(b[j,:2]+b[j,2:]/2)*640
   xx=np.clip((np.array([x1,x2])-t['pad_left'])/t['scale_x'],0,orig['width']);yy=np.clip((np.array([y1,y2])-t['pad_top'])/t['scale_y'],0,orig['height'])
   if xx[1]>xx[0] and yy[1]>yy[0]:predictions.append(dict(image_id=im['id'],category_id=0,bbox=[float(xx[0]),float(yy[0]),float(xx[1]-xx[0]),float(yy[1]-yy[0])],score=float(scores[j])))
  # Diagnostic oracle grouping never enters inference scores or postprocessing.
  for ov in r['ov']:
   ix=np.where(r['valid']&(ov>=.1))[0]
   if len(ix):diagnostics.append(dict(old=float(ov[ix[r['uq'][ix].argmax()]]),new=float(ov[ix[scores[ix].argmax()]])))
 pp=nms(predictions,.7)
 with contextlib.redirect_stdout(io.StringIO()):ap=coco_metrics(original,pp)
 op=operating(match(pp,ground),.1,66)
 met=dict(base=base,seed=seed,mode=mode,AP=ap[0]*100,AP75=ap[2]*100,TP_FP6=op['TP'],old_precise=sum(d['old']>=.75 for d in diagnostics),new_precise=sum(d['new']>=.75 for d in diagnostics),improved=sum(d['new']>d['old']+1e-6 for d in diagnostics),worsened=sum(d['new']<d['old']-1e-6 for d in diagnostics))
 (out/'predictions.json').write_text(json.dumps(predictions));(out/'metrics.json').write_text(json.dumps(met,indent=2));return met
