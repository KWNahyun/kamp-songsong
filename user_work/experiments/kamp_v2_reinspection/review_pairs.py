from pathlib import Path
import json,csv,datetime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
R=Path('/home/viplab/contest');E=Path(__file__).parent;U=E.parent/'kamp_v2_uq';G=json.loads((U/'original/annotations/val.json').read_text());ims={i['id']:i for i in G['images']};conds={int(r['gt_id']):r for r in csv.DictReader((U/'analysis/gt_conditions.csv').open(encoding='utf-8-sig'))};rows=list(csv.DictReader((E/'gt_failures.csv').open(encoding='utf-8-sig')));anns={a['id']:a for a in G['annotations']}
def succeeded(gid):return all(r['matched']=='True' for r in rows if int(r['gt_id'])==gid and r['model']=='dfine_mal_UQ')
def stamp(a):return datetime.datetime.strptime(ims[a['image_id']]['file_name'].split('(')[0],'%03d_%%Y%%m%%d_%%H%%M%%S'%int(ims[a['image_id']]['file_name'].split('_')[0]))
pairs=[]
for gid in [110,136,143]:
 a=anns[gid];cx,cy=np.array(a['bbox'][:2])+np.array(a['bbox'][2:])/2;pool=[]
 for b in G['annotations']:
  if b['image_id']==a['image_id'] or conds[b['id']]['group']!=conds[gid]['group'] or not succeeded(b['id']):continue
  bx,by=np.array(b['bbox'][:2])+np.array(b['bbox'][2:])/2;distance=float(np.hypot(bx-cx,by-cy))
  if distance>40:continue
  delta=abs((stamp(b)-stamp(a)).total_seconds());pool.append((delta,distance,b['id']))
 assert pool
 _,dist,other=min(pool);pairs.append(dict(failure_gt=gid,success_gt=other,center_distance=dist,time_seconds=min(pool)[0],group=conds[gid]['group']))
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();rng=np.random.default_rng(771);fig,axs=plt.subplots(3,4,figsize=(12,9));key=[]
for row,pair in enumerate(pairs):
 ids=[pair['failure_gt'],pair['success_gt']];rng.shuffle(ids)
 for side,gid in enumerate(ids):
  a=anns[gid];im=ims[a['image_id']];ar=np.array(Image.open(R/'kamp_xray_v2/images/val'/im['file_name']));x,y,w,h=a['bbox'];cx=x+w/2;cy=y+h/2
  for over in [0,1]:
   ax=axs[row,side*2+over];ax.imshow(ar,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(0,ar.shape[1],ar.shape[0],0));ax.set_xlim(cx-20,cx+20);ax.set_ylim(cy+20,cy-20);ax.set_xticks([]);ax.set_yticks([])
   if over:ax.add_patch(Rectangle((x,y),w,h,fill=False,ec='#19b86a',ls='--',lw=1.5))
   ax.set_title(f"쌍{row+1} {'AB'[side]} · "+(f'TXT {w:.1f}×{h:.1f}px' if over else '영상만'),fontsize=10)
  key.append(dict(pair=row+1,side='AB'[side],gt_id=gid,status='repeat_failure' if gid==pair['failure_gt'] else 'all3_success',stem=Path(im['file_name']).stem))
fig.suptitle('모델 예측·성공 여부를 가린 경계 검토 자료\n같은 장비·날짜의 인접 촬영, 원본40×40px 최근접 확대',fontsize=13);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(E/'blind_boundary_pairs.png',dpi=170);plt.close(fig);(E/'boundary_pair_key.json').write_text(json.dumps(dict(pairs=pairs,key=key,scope='review material only; no independent human adjudication performed'),ensure_ascii=False,indent=2));print(pairs)
