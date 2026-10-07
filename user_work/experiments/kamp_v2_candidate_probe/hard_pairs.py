from pathlib import Path
import json,csv
import numpy as np
import torch
from torchvision.ops import box_iou
E=Path(__file__).parent;U=E.parent/'kamp_v2_uq';rows=[]
gt=json.loads((U/'data640/annotations/val.json').read_text())
for folder in sorted(E.glob('dfine_*')):
 p=np.load(folder/'probe_all.npz');cols=p['cols'];scale=p['scale'];w=p['w'];acc={k:[] for k in ['all','both_iou_05','both_iou_065']}
 for im in gt['images']:
  z=np.load(folder/f"val_{im['id']}.npz");b=torch.tensor(z['boxes']);x=z['x'];s=z['score'];xy=torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1)
  anns=[a for a in gt['annotations'] if a['image_id']==im['id']];g=torch.tensor([a['bbox'] for a in anns]);g[:,2:]+=g[:,:2];g/=640;io=torch.nan_to_num(box_iou(g,xy)).numpy();assignment=io.argmax(0)
  valid=np.isfinite(x).all(1)&np.isfinite(z['boxes']).all(1)&(z['boxes'][:,2:]>0).all(1)&(s>=.001)
  for j,a in enumerate(anns):
   ix=np.where(valid&(io[j]>=.1)&(assignment==j))[0];y=io[j,ix];q=x[ix][:,cols]/scale@w;aa,bb=np.triu_indices(len(ix),1)
   for key,cut in [('all',.1),('both_iou_05',.5),('both_iou_065',.65)]:
    k=(abs(y[aa]-y[bb])>=.05)&(y[aa]>=cut)&(y[bb]>=cut)
    if k.any():acc[key].append(float(np.mean(np.sign(q[aa[k]]-q[bb[k]])==np.sign(y[aa[k]]-y[bb[k]]))))
 for key,v in acc.items():rows.append(dict(run=folder.name,subset=key,gt_with_pairs=len(v),accuracy=float(np.mean(v))))
with (E/'hard_pairs.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print(json.dumps(rows,indent=2))
