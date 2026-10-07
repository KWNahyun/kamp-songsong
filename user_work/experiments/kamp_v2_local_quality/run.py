from pathlib import Path
import argparse,time,json,hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import common as c
E=Path(__file__).parent
class Residual(nn.Module):
 def __init__(self,state):
  super().__init__();self.base=c.Head();self.base.load_state_dict(state)
  for p in self.base.parameters():p.requires_grad_(False)
  self.delta=nn.Sequential(nn.LayerNorm(541),nn.Linear(541,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
  nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
 def forward(self,x):return self.base(x[...,:283])+self.delta(x).squeeze(-1)
def spatial(records,split):
 arrays=[]
 for r in records:
  image=Image.open(c.R/f'kamp_xray_v2/images/{split}'/r['im']['file_name']).convert('L');a=torch.tensor(np.array(image).copy(),device='cuda').float()[None,None]/255;H,W=a.shape[-2:];t=c.trans[r['im']['file_name']]
  b=torch.tensor(r['b'],device='cuda');center=b[:,:2]*640;center[:,0]=(center[:,0]-t['pad_left'])/t['scale_x'];center[:,1]=(center[:,1]-t['pad_top'])/t['scale_y'];size=b[:,2:].clamp_min(1e-6)*640;size[:,0]/=t['scale_x'];size[:,1]/=t['scale_y'];size=size*1.5
  yy,xx=torch.meshgrid((torch.arange(16,device='cuda')+.5)/16-.5,(torch.arange(16,device='cuda')+.5)/16-.5,indexing='ij');offset=torch.stack((xx,yy),-1);coord=center[:,None,None,:]+offset[None]*size[:,None,None,:]
  # Continuous image edge coordinates, align_corners=False; GT never used for crop.
  grid=coord/torch.tensor([W,H],device='cuda')*2-1
  patch=F.grid_sample(a.expand(len(b),-1,-1,-1),grid,align_corners=False,padding_mode='border',mode='bilinear').flatten(1);mu=patch.mean(1,keepdim=True);sd=patch.std(1,keepdim=True,unbiased=False)
  feat=torch.cat(((patch-mu)/sd.clamp_min(.02),mu,sd),1);assert torch.isfinite(feat).all();arrays.append(feat.cpu().numpy())
 return arrays
def main(seed):
 start=time.time();name=f'dfine_mal_UQ_seed{seed}';train=c.load(name,'train');val=c.load(name,'val');orig_train=[r['x'].copy() for r in train];orig_val=[r['x'].copy() for r in val]
 localtrain=spatial(train,'train');localval=spatial(val,'val')
 ck=torch.load(c.U/'runs_frozen'/name/'last.pth',map_location='cpu',weights_only=False);state={k.split('kamp_selector.',1)[1]:v for k,v in ck['model'].items() if 'kamp_selector.' in k}
 results=[]
 for mode in ['query_control','query_local']:
  for records,xs,patches in [(train,orig_train,localtrain),(val,orig_val,localval)]:
   for r,x,patch in zip(records,xs,patches):
    feature=patch if mode=='query_local' else np.concatenate((x[:,:256],x[:,-2:]),1)
    r['x']=np.concatenate((x,feature),1)
  X,Y,PI,PM,GI,GM=c.tensors(train);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);head=Residual(state).cuda();head.eval();out=E/f'mal_{seed}_{mode}';out.mkdir(exist_ok=False)
  with torch.no_grad():q=head(torch.tensor(val[0]['x'],device='cuda')).sigmoid().cpu().numpy();np.testing.assert_allclose(np.sqrt(val[0]['s']*q),val[0]['uq'],atol=1e-6,rtol=1e-5)
  opt=torch.optim.AdamW(head.delta.parameters(),lr=1e-4,weight_decay=1e-4);gen=torch.Generator().manual_seed(seed);jobstart=time.time()
  for epoch in range(30):
   losses=[]
   for ids in torch.randperm(len(train),generator=gen).split(16):
    ids=ids.cuda();pred=head(X[ids]);target=Y[ids];quality=(F.binary_cross_entropy_with_logits(pred,target,reduction='none')*(target-pred.sigmoid()).abs().square()).mean();ix=PI[ids];mask=PM[ids];sign=(target.gather(1,ix[:,:,0])-target.gather(1,ix[:,:,1])).sign();raw=F.softplus(-sign*(pred.gather(1,ix[:,:,0])-pred.gather(1,ix[:,:,1])));counts=mask.sum(1);active=counts>0
    pair=((raw*mask).sum(1)/counts.clamp_min(1))[active].mean() if active.any() else pred.sum()*0;loss=quality+.1*pair;opt.zero_grad();loss.backward();assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.delta.parameters());opt.step();losses.append([float(quality.detach()),float(pair.detach())])
   row=dict(epoch=epoch+1,quality_pair=np.mean(losses,axis=0).tolist(),seconds=time.time()-jobstart)
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(seed,mode,row,flush=True)
  assert all(torch.equal(v.cpu(),head.base.state_dict()[k].cpu()) for k,v in state.items())
  torch.save(dict(model=head.state_dict(),seed=seed,mode=mode,epochs=30),out/'last.pth');met=c.evaluate(head,val,'mal',seed,mode,out);met.update(trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),base_unchanged=True,seconds=time.time()-jobstart);results.append(met)
  if mode=='query_local':
   # Inference-only removal diagnostic, no refitting or model selection.
   backup=[r['x'].copy() for r in val]
   for r in val:r['x'][:,283:]=0
   zero=out/'zero_local_diagnostic';zero.mkdir();zmet=c.evaluate(head,val,'mal',seed,'zero_local_diagnostic',zero);met['zero_local_diagnostic']=zmet
   for r,x in zip(val,backup):r['x']=x
  (out/'metrics.json').write_text(json.dumps(met,indent=2));print('RESULT',met,flush=True)
  del X,Y,PI,PM,GI,GM;torch.cuda.empty_cache()
 (E/f'results_seed{seed}.json').write_text(json.dumps(results,indent=2));(E/f'complete_seed{seed}.json').write_text(json.dumps(dict(completed=True,seed=seed,seconds=time.time()-start,test_used=False,detector_unchanged=True),indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20260929);main(p.parse_args().seed)
