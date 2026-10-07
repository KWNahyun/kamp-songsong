from pathlib import Path
import csv,json,random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=Path('/home/viplab/contest/kamp_xray_v2');O=Path(__file__).parent
rows=list(csv.DictReader((R/'split_manifest.csv').open()));rng=random.Random(20261002);selected=[]
for machine in ['M1','M2','M3']:
 pool=[r for r in rows if r['split']=='train' and r['machine']==machine]; groups=sorted({r['group'] for r in pool});rng.shuffle(groups);chosen=[]
 for group in groups:chosen.append(rng.choice([r for r in pool if r['group']==group]))
 extra=[r for r in pool if r not in chosen];rng.shuffle(extra);chosen=(chosen+extra)[:6]
 fig,axes=plt.subplots(2,6,figsize=(15,6.8))
 for i,row in enumerate(chosen):
  arr=np.asarray(Image.open(R/row['image_path']).convert('L'));h,w=arr.shape
  lines=(R/row['label_path']).read_text().splitlines();li=rng.randrange(len(lines));_,cx,cy,bw,bh=map(float,lines[li].split());cx*=w;cy*=h;bw*=w;bh*=h
  x=max(0,min(w-32,int(np.floor(cx))-16));y=max(0,min(h-32,int(np.floor(cy))-16)); crop=arr[y:y+32,x:x+32];a,b=axes[i//3,(i%3)*2:(i%3)*2+2]
  for ax in [a,b]:ax.imshow(crop,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(x,x+32,y+32,y));ax.set_xticks([]);ax.set_yticks([])
  b.add_patch(Rectangle((cx-bw/2,cy-bh/2),bw,bh,fill=False,edgecolor='#00dd88',lw=1,linestyle='--'))
  a.set_title(f'{machine}-{i+1} image',fontsize=10);b.set_title(f'GT {bw:.1f} x {bh:.1f} px',fontsize=10)
  selected.append(dict(case=f'{machine}-{i+1}',**row,label_line=li+1,bbox_xywh=[cx-bw/2,cy-bh/2,bw,bh],crop_xywh=[x,y,32,32]))
 fig.suptitle(f'{machine}: train only | 32 x 32 original pixels | nearest-neighbor | no contrast adjustment',fontsize=12);fig.tight_layout(h_pad=2.0);fig.savefig(O/f'{machine}_train_gt.png',dpi=140);plt.close(fig)
(O/'selection.json').write_text(json.dumps(dict(seed=20261002,selection='one random image per shuffled training group, then additional random images up to six per machine; random object per image; no prediction-based selection',cases=selected),indent=2))
print('Saved 18 train crops, 3 machine sheets; no test or prediction read.')
