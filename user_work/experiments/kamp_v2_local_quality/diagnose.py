from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import run as r
import common as c
E=Path(__file__).parent;seed=20260929
records=c.load(f'dfine_mal_UQ_seed{seed}','val');xs=[z['x'].copy() for z in records];patches=r.spatial(records,'val')
ck=torch.load(c.U/f'runs_frozen/dfine_mal_UQ_seed{seed}/last.pth',map_location='cpu',weights_only=False);state={k.split('kamp_selector.',1)[1]:v for k,v in ck['model'].items() if 'kamp_selector.' in k};results=[]
for mode in ['query_control','query_local']:
 head=r.Residual(state).cuda();head.load_state_dict(torch.load(E/f'mal_{seed}_{mode}/last.pth',map_location='cpu',weights_only=False)['model']);head.eval();accuracy=[];quality_precise=improved=worsened=0;residual=[]
 for z,x,patch in zip(records,xs,patches):
  branch=patch if mode=='query_local' else np.concatenate((x[:,:256],x[:,-2:]),1);xx=torch.tensor(np.concatenate((x,branch),1),device='cuda')
  with torch.no_grad():quality=head(xx).sigmoid().cpu().numpy();residual.extend(head.delta(xx).cpu().numpy().flatten().tolist())
  if z['pairs']:
   p=np.array(z['pairs']);accuracy.append(float(np.mean(np.sign(quality[p[:,0]]-quality[p[:,1]])==np.sign(z['y'][p[:,0]]-z['y'][p[:,1]]))))
  for ov in z['ov']:
   ix=np.where(z['valid']&(ov>=.1))[0]
   if len(ix):
    old=ix[z['uq'][ix].argmax()];new=ix[quality[ix].argmax()];quality_precise+=int(ov[new]>=.75);improved+=int(ov[new]>ov[old]+1e-6);worsened+=int(ov[new]<ov[old]-1e-6)
 results.append(dict(mode=mode,hard_pair_accuracy=float(np.mean(accuracy)),quality_only_precise=quality_precise,quality_only_improved=improved,quality_only_worsened=worsened,residual_logit_std=float(np.std(residual))))
# Sampling-coordinate check: manually sample one predicted patch with CPU arithmetic.
z=records[0];j=int(z['s'].argmax());im=Image.open(c.R/'kamp_xray_v2/images/val'/z['im']['file_name']).convert('L');a=np.array(im,dtype=float)/255;t=c.trans[z['im']['file_name']];b=z['b'][j];cx=(b[0]*640-t['pad_left'])/t['scale_x'];cy=(b[1]*640-t['pad_top'])/t['scale_y'];w=max(float(b[2]),1e-6)*640/t['scale_x']*1.5;h=max(float(b[3]),1e-6)*640/t['scale_y']*1.5
vals=[]
for yy in range(16):
 for xx in range(16):
  px=np.clip(cx+((xx+.5)/16-.5)*w-.5,0,a.shape[1]-1);py=np.clip(cy+((yy+.5)/16-.5)*h-.5,0,a.shape[0]-1);x0=int(np.floor(px));y0=int(np.floor(py));x1=min(x0+1,a.shape[1]-1);y1=min(y0+1,a.shape[0]-1);fx=px-x0;fy=py-y0;vals.append(a[y0,x0]*(1-fx)*(1-fy)+a[y0,x1]*fx*(1-fy)+a[y1,x0]*(1-fx)*fy+a[y1,x1]*fx*fy)
vals=np.array(vals);expected=np.r_[(vals-vals.mean())/max(vals.std(),.02),vals.mean(),vals.std()];diff=float(np.max(abs(expected-patches[0][j])));assert diff<.001,diff
for mode in ['query_control','query_local']:
 rows=[json.loads(line) for line in (E/f'mal_{seed}_{mode}/train.jsonl').read_text().splitlines()];assert len(rows)==30 and rows[-1]['epoch']==30
(E/'diagnostics.json').write_text(json.dumps(dict(results=results,cpu_bilinear_feature_max_error=diff,epochs_verified=30,test_used=False),indent=2));print(json.dumps(results,indent=2));print('ROI validation',diff)
