from pathlib import Path
import json,csv,sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
R=Path('/home/viplab/contest');E=Path(__file__).parent;U=E.parent/'kamp_v2_uq';sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import iou,nms,dumpcsv
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
curves=list(csv.DictReader((E/'review_curves.csv').open(encoding='utf-8-sig')));fig,axs=plt.subplots(1,3,figsize=(12,3.6))
for ax,seed in zip(axs,[20260929,20260930,20261001]):
 for signal,label in [('low_confidence','낮은 신뢰도'),('cross_model_disagreement','모델 불일치')]:
  rr=[r for r in curves if r['seed']==str(seed) and r['signal']==signal];x=[int(r['budget_images']) for r in rr];ax.plot(x,[int(r['miss_gt_in_selected']) for r in rr],'o-',label=label)
 rr=[r for r in curves if r['seed']==str(seed) and r['signal']=='low_confidence'];total=int(rr[0]['miss_gt_total']);ax.plot(x,[total*v/66 for v in x],'--',color='gray',label='무작위 기대값');ax.fill_between(x,[float(r['random_p025']) for r in rr],[float(r['random_p975']) for r in rr],color='gray',alpha=.12,label='무작위 95% 범위');ax.set_ylim(-.1,3.2);ax.set_yticks(range(4));ax.set_xticks(x);ax.set_xlabel('검토할 영상 수 / 66');ax.set_title(f'seed {seed} · 미탐 {total}개');ax.grid(alpha=.2)
axs[0].set_ylabel('검토 목록에 포함된 미탐 GT 수');axs[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(E/'review_curves.png',dpi=170);plt.close(fig)
fail=[r for r in json.loads((E/'repeat_failures.json').read_text()) if r['mal_uq_misses_out_of3']];checks=json.loads((E/'verification.json').read_text())['operating_points'];pred={s:nms(json.loads((U/f'runs_frozen/dfine_mal_UQ_seed{s}/common_eval/predictions_original.json').read_text()),.7) for s in [20260929,20260930,20261001]}
fig,axs=plt.subplots(3,4,figsize=(12,9))
for row,r in enumerate(fail):
 a=np.array(Image.open(R/f"kamp_xray_v2/images/val/{r['stem']}.png"));g=r['bbox'];cx=g[0]+g[2]/2;cy=g[1]+g[3]/2;side=max(30,g[2]+14,g[3]+14)
 for col,ax in enumerate(axs[row]):
  ax.imshow(a,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(0,a.shape[1],a.shape[0],0));ax.set_xlim(cx-side/2,cx+side/2);ax.set_ylim(cy+side/2,cy-side/2);ax.set_xticks([]);ax.set_yticks([])
  if col==0:ax.set_title(f"GT {r['gt_id']} · 원본 확대\n{r['group']}",fontsize=10)
  else:
   seed=[20260929,20260930,20261001][col-1];th=next(x['threshold'] for x in checks if x['seed']==seed);ps=[p for p in pred[seed] if p['image_id']==r['image_id'] and p['score']>=th and iou(g,p['bbox'])>=.1];p=max(ps,key=lambda p:p['score'],default=None);ax.add_patch(Rectangle(g[:2],g[2],g[3],fill=False,ec='#16b46a',ls='--',lw=1.5))
   if p:
    b=p['bbox'];ax.add_patch(Rectangle(b[:2],b[2],b[3],fill=False,ec='#ed593d',lw=1.5));ax.set_title(f"{seed} · IoU {iou(g,b):.3f}\nscore {p['score']:.3f} / 기준 {th:.3f}",fontsize=10)
   else:ax.set_title(f'{seed} · 최종 근접 출력 없음',fontsize=10)
fig.suptitle('반복 M3 실패: 초록=공식 TXT, 주황=최종 근접 출력\nGT110·136은 세 seed 모두 미탐, GT143은 첫 seed에서만 미탐',fontsize=13);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(E/'m3_failure_cases.png',dpi=170);plt.close(fig)
rows=list(csv.DictReader((E/'image_signals.csv').open(encoding='utf-8-sig')));ranks=[]
for seed in [20260929,20260930,20261001]:
 rr=[r for r in rows if r['seed']==str(seed)]
 for signal in ['low_confidence','cross_model_disagreement','count_difference']:
  ordered=sorted(rr,key=lambda r:(-float(r[signal]),int(r['image_id'])))
  for rank,r in enumerate(ordered,1):
   if int(r['missed_gt']):ranks.append(dict(seed=seed,signal=signal,rank=rank,image_id=r['image_id'],stem=r['stem'],missed_gt=r['missed_gt'],signal_value=r[signal]))
dumpcsv(E/'missed_image_ranks.csv',ranks)
print(ranks)
