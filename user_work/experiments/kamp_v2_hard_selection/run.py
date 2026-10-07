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
start=time.time();results=[];jobs=[]
for base in ['base','mal']:
 for seed in [20260929,20260930,20261001]:
  name=f'dfine_{base}_UQ_seed{seed}';train=load(name,'train');val=load(name,'val');X,Y,PI,PM,GI,GM=tensors(train)
  ck=torch.load(U/'runs_frozen'/name/'last.pth',map_location='cpu',weights_only=False);state={k.split('kamp_selector.',1)[1]:v for k,v in ck['model'].items() if 'kamp_selector.' in k}
  for mode in ['quality','hard_pair','listwise']:
   out=E/f'{base}_{seed}_{mode}';out.mkdir(exist_ok=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);head=Head().cuda();head.load_state_dict(state)
   with torch.no_grad():q=head(torch.tensor(val[0]['x'],device='cuda')).sigmoid().cpu().numpy();np.testing.assert_allclose(np.sqrt(val[0]['s']*q),val[0]['uq'],atol=1e-6,rtol=1e-5)
   opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4);gen=torch.Generator().manual_seed(seed);jobstart=time.time()
   for epoch in range(15):
    losses=[]
    for ids in torch.randperm(len(train),generator=gen).split(16):
     ids=ids.cuda();pred=head(X[ids]);target=Y[ids];lossq=(F.binary_cross_entropy_with_logits(pred,target,reduction='none')*(target-pred.sigmoid()).abs().square()).mean();aux=pred.sum()*0
     if mode=='hard_pair':
      ii=PI[ids];mask=PM[ids];a=pred.gather(1,ii[:,:,0]);b=pred.gather(1,ii[:,:,1]);sign=(target.gather(1,ii[:,:,0])-target.gather(1,ii[:,:,1])).sign();losses_p=F.softplus(-sign*(a-b));counts=mask.sum(1);active=counts>0
      if active.any():aux=((losses_p*mask).sum(1)/counts.clamp_min(1))[active].mean()
     elif mode=='listwise':
      ix=GI[ids];mask=GM[ids];p=pred.gather(1,ix.flatten(1)).reshape(ix.shape);y=target.gather(1,ix.flatten(1)).reshape(ix.shape);active=mask.sum(-1)>=2
      # Only train lists with a credible object; all-background lists use quality loss.
      active=active&y.masked_fill(~mask,-1).amax(-1).ge(.5)
      if active.any():
       p=p[active].masked_fill(~mask[active],-1e4);y=y[active].masked_fill(~mask[active],-1e4);aux=-(F.softmax(y/.1,dim=-1)*F.log_softmax(p,dim=-1)).sum(-1).mean()
     loss=lossq+.1*aux;opt.zero_grad();loss.backward();assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters());opt.step();losses.append([float(loss.detach()),float(lossq.detach()),float(aux.detach())])
    row=dict(epoch=epoch+1,loss=np.mean(losses,axis=0).tolist(),seconds=time.time()-jobstart)
    with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(base,seed,mode,row,flush=True)
    if epoch==0:
     (E/'status.json').write_text(json.dumps(dict(current=out.name,epoch=1,first_epoch_seconds=row['seconds'],estimated_run_seconds=row['seconds']*15,completed_jobs=len(results),total_jobs=18),indent=2))
   torch.save(dict(head=head.state_dict(),epochs=15,base=name,mode=mode,test_used=False),out/'last.pth')
   head.eval();met=evaluate(head,val,base,seed,mode,out);met['seconds']=time.time()-jobstart;results.append(met)
   (E/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',met,flush=True)
   (E/'status.json').write_text(json.dumps(dict(completed_jobs=len(results),total_jobs=18,last=out.name,elapsed_seconds=time.time()-start),indent=2))
  jobs.append(dict(base=base,seed=seed,train_images_with_hard_pairs=sum(bool(r['pairs']) for r in train),train_hard_pairs=sum(len(r['pairs']) for r in train),test_used=False))
  del X,Y,PI,PM,GI,GM;torch.cuda.empty_cache()
(E/'audit.json').write_text(json.dumps(jobs,indent=2));(E/'completion.json').write_text(json.dumps(dict(completed=True,jobs=18,test_used=False,seconds=time.time()-start),indent=2));print('COMPLETE',flush=True)
