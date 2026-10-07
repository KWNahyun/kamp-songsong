"""Train-only within-GT centered ridge diagnostic; GT groups are oracle, not a detector."""
from pathlib import Path
import sys,json,csv
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou
R=Path('/home/viplab/contest');E=Path(__file__).parent;U=R/'experiments/kamp_v2_uq'
sys.path[:0]=[str(U),str(R/'models/D-FINE')]
from selector_patch import install_model
install_model('unary')
from src.core import YAMLConfig
torch.set_num_threads(4)
featuresets={'score':[256+20], 'geometry_score':list(range(276,283)), 'distribution_geometry':list(range(256,283)), 'query_only':list(range(256)), 'all':list(range(283))}
# Actual hidden dimension discovered below, not assumed.
rows=[];details=[];audits=[]
for base in ['base','mal']:
 for seed in [20260929,20260930,20261001]:
  name=f'dfine_{base}_UQ_seed{seed}';out=E/name;out.mkdir(exist_ok=True)
  cfg=YAMLConfig(str(U/'configs/dfine_UQ_seed20260929.yml'));model=cfg.model.cuda().eval()
  ck=torch.load(U/'runs_frozen'/name/'last.pth',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
  capture={}
  def hook(m,args): capture['x']=args[0].detach().cpu().numpy()[0]
  handle=model.decoder.kamp_selector.embedding.register_forward_pre_hook(hook)
  groups={};checks=[]
  for split in ['train','val']:
   groups[split]=[];gt=json.loads((U/f'data640/annotations/{split}.json').read_text())
   for im in gt['images']:
    cache=out/f"{split}_{im['id']}.npz"
    if cache.exists():
     z=np.load(cache);x=z['x'];b=z['boxes'];score=z['score'];uq=z['uq']
    else:
     ar=np.array(Image.open(U/f'data640/images/{split}'/im['file_name']).convert('RGB'))
     with torch.no_grad(): pred=model(torch.from_numpy(ar.copy()).permute(2,0,1)[None].cuda().float()/255)
     x=capture['x'];b=pred['pred_boxes'][0].cpu().numpy();score=pred['pred_logits_base'][0,:,0].sigmoid().cpu().numpy();uq=pred['pred_logits'][0,:,0].sigmoid().cpu().numpy()
     np.savez_compressed(cache,x=x,boxes=b,score=score,uq=uq)
    if split=='val':
     ref=np.load(U/'runs_frozen'/name/'query_audit'/f"image_{im['id']}.npz")
     np.testing.assert_allclose(b,ref['boxes'],atol=1e-6);np.testing.assert_allclose(score,ref['base_scores'],atol=1e-6)
     checks.append(float(abs(b-ref['boxes']).max()))
    anns=[a for a in gt['annotations'] if a['image_id']==im['id']]
    g=torch.tensor([a['bbox'] for a in anns]);g[:,2:]+=g[:,:2];g/=640
    bt=torch.tensor(b);xy=torch.cat((bt[:,:2]-bt[:,2:]/2,bt[:,:2]+bt[:,2:]/2),1)
    ios=torch.nan_to_num(box_iou(g,xy)).numpy();valid=np.isfinite(x).all(1)&np.isfinite(b).all(1)&(b[:,2:]>0).all(1)&(score>=.001)
    # Exclusive GT assignment avoids sharing a query among nearby objects.
    assignment=ios.argmax(0)
    for j,a in enumerate(anns):
     ix=np.where(valid&(ios[j]>=.1)&(assignment==j))[0]
     groups[split].append(dict(x=x[ix],y=ios[j,ix],score=score[ix],uq=uq[ix],id=a['id'],image=im['id']))
   print(name,split,'extracted',len(groups[split]),flush=True)
  hd=x.shape[1]-27
  sets={'score':[hd+20],'geometry_score':list(range(hd+20,hd+27)),'distribution_geometry':list(range(hd,hd+27)),'query_only':list(range(hd)),'all':list(range(hd+27))}
  train=[g for g in groups['train'] if len(g['y'])>=2]
  for feat,cols in sets.items():
   # Each GT carries equal weight; subtract within-group means = relative quality objective.
   X=np.concatenate([g['x'][:,cols]-g['x'][:,cols].mean(0) for g in train]).astype('float64')
   Y=np.concatenate([g['y']-g['y'].mean() for g in train]);weights=np.concatenate([np.full(len(g['y']),1/len(g['y'])) for g in train]);weights/=weights.sum()
   scale=np.sqrt((X**2*weights[:,None]).sum(0)).clip(1e-4);X/=scale
   # Fixed ridge strength; no validation tuning.
   w=np.linalg.solve(X.T@(weights[:,None]*X)+.1*np.eye(len(cols)),X.T@(weights*Y))
   np.savez(out/f'probe_{feat}.npz',cols=cols,scale=scale,w=w)
   for split in ['train','val']:
    rs=[]
    for g in groups[split]:
     y=g['y'];n=len(y)
     if not n: continue
     p=g['x'][:,cols]/scale@w;bi=int(g['score'].argmax());ui=int(g['uq'].argmax());pi=int(p.argmax())
     aa,bb=np.triu_indices(n,1);keep=abs(y[aa]-y[bb])>=.05
     acc=float(np.mean(np.sign(p[aa[keep]]-p[bb[keep]])==np.sign(y[aa[keep]]-y[bb[keep]]))) if keep.any() else float('nan')
     r=dict(run=name,base=base,seed=seed,features=feat,split=split,gt_id=g['id'],candidates=n,oracle=float(y.max()),base_iou=float(y[bi]),uq_iou=float(y[ui]),probe_iou=float(y[pi]),pair_accuracy=acc,changed=pi!=bi)
     rs.append(r);details.append(r)
    rows.append(dict(run=name,base=base,seed=seed,features=feat,split=split,gt_total=len(groups[split]),gt_with_candidate=len(rs),multi_candidate=sum(r['candidates']>=2 for r in rs),pair_accuracy=float(np.nanmean([r['pair_accuracy'] for r in rs])),base_mean_iou=float(np.mean([r['base_iou'] for r in rs])),probe_mean_iou=float(np.mean([r['probe_iou'] for r in rs])),oracle_mean_iou=float(np.mean([r['oracle'] for r in rs])),base75=sum(r['base_iou']>=.75 for r in rs),probe75=sum(r['probe_iou']>=.75 for r in rs),oracle75=sum(r['oracle']>=.75 for r in rs),improved=sum(r['probe_iou']>r['base_iou']+1e-6 for r in rs),worsened=sum(r['probe_iou']<r['base_iou']-1e-6 for r in rs)))
  audits.append(dict(run=name,val_identity_max=max(checks),train_gt=len(groups['train']),val_gt=len(groups['val']),feature_dim=x.shape[1],test_used=False))
  handle.remove();del model;torch.cuda.empty_cache()
  for file,data in [('summary.csv',rows),('instances.csv',details)]:
   with (E/file).open('w') as f:
    wr=csv.DictWriter(f,fieldnames=list(data[0]));wr.writeheader();wr.writerows(data)
  (E/'verification.json').write_text(json.dumps(audits,indent=2))
print('COMPLETE',flush=True)
