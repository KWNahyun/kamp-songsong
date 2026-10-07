from pathlib import Path
import sys,json,csv,collections,copy
import numpy as np
import torch
from torchvision.ops import nms as tnms
E=Path(__file__).parent;P=E.parent/'kamp_pilot_v1';O=E/'analysis';O.mkdir(exist_ok=True)
sys.path.insert(0,str(P))
from analyze_failures import match,operating,iou,coco_metrics,dumpcsv
GT=json.loads((Path('/home/viplab/contest/data/processed/kamp500_telea_v1/annotations/val.json')).read_text())
ground={i['id']:[a for a in GT['annotations'] if a['image_id']==i['id']] for i in GT['images']}
anns={a['id']:a for a in GT['annotations']}
descriptors=list(csv.DictReader((P/'failure_analysis/gt_conditions.csv').open(encoding='utf-8-sig')))
jobs=json.loads((E/'status.json').read_text())['jobs']
runs=[dict(j,path=E/'runs'/j['name']) for j in jobs if j['status']=='trained']
V=E.parent/'kamp_ablation_v3'
previous=json.loads((V/'queue_status.json').read_text())['jobs']
runs += [dict(j,path=V/'runs'/j['name']) for j in previous if j['variant'] in ['dfine_s','M'] and j['status']=='trained']
runs.append(dict(name='dfine_s',variant='dfine_s',seed=20260929,path=P/'dfine_s'))
rows=[];conditions=[];locs=[];post=[];rankrows=[]
for j in runs:
 path=j['path'];m=json.loads((path/'common_eval/metrics.json').read_text());pred=json.loads((path/'common_eval/predictions_original.json').read_text())
 events=match(pred,ground);op=operating(events,.1,64)
 row={k:j[k] for k in ['name','variant','seed']}
 row.update(AP=100*m['AP50_95'],AP50=100*m['AP50'],AP75=100*m['AP75'])
 for budget in [.05,.1,.2,.5]:row[f'TP_{budget}']=operating(events,budget,64)['TP']
 for label in ['lt8','8to12','12to16','ge16']:
  ids={int(d['gt_id']) for d in descriptors if d['size_bin']==label};row['TP_'+label]=len(ids & op['matched'])
 for category in ['localization','duplicate','background_or_unlabeled']:
  row['FP_'+category]=sum(e['category']==category for e in op['events'])
 row['image_alarm']=len({p['image_id'] for p in pred if p['score']>=op['threshold']})
 for axis in ['machine','date','contrast_bin']:
  for value in sorted({d[axis] for d in descriptors}):
   ids={int(d['gt_id']) for d in descriptors if d[axis]==value}
   conditions.append(dict(name=j['name'],variant=j['variant'],seed=j['seed'],axis=axis,value=value,GT=len(ids),TP=len(ids & op['matched'])))
 local=[]
 for e in match([p for p in pred if p['score']>=.05],ground,.1):
  if not e['tp']:continue
  a=anns[e['gt_id']];x,y,w,h=e['bbox'];gx,gy,gw,gh=a['bbox']
  rr=dict(name=j['name'],gt_id=a['id'],center=float(np.hypot(x+w/2-gx-gw/2,y+h/2-gy-gh/2)),wlog=float(np.log(w/gw)),hlog=float(np.log(h/gh)),iou=iou(e['bbox'],a['bbox']))
  local.append(rr);locs.append(rr)
 row.update(matched_pairs=len(local),center_median=float(np.median([r['center'] for r in local])),size_abslog=float(np.mean([(abs(r['wlog'])+abs(r['hlog']))/2 for r in local])),width_bias=float(np.median([r['wlog'] for r in local])),height_bias=float(np.median([r['hlog'] for r in local])))
 # Candidate ranking diagnostic; candidate set has oracle GT overlap >= .1, not runtime.
 ranks=[]
 byim=collections.defaultdict(list)
 for p in pred:byim[p['image_id']].append(p)
 for a in GT['annotations']:
  candidates=[(p['score'],iou(a['bbox'],p['bbox'])) for p in byim[a['image_id']] if p['score']>=.05 and iou(a['bbox'],p['bbox'])>=.1]
  if candidates:
   selected=max(candidates,key=lambda x:x[0])[1];best=max(x[1] for x in candidates)
   ranks.append((best-selected,best,selected))
 row.update(rank_gap=float(np.mean([r[0] for r in ranks])),rank_recoverable75=sum(r[1]>=.75 and r[2]<.75 for r in ranks),candidate75=sum(r[1]>=.75 for r in ranks))
 rows.append(row)
 # Same fixed postprocessing for all: original, NMS .7 (DETR only), scale .95, both.
 nms_pred=pred
 if j['variant'].startswith('dfine') or j['variant'] in ['G01','G03','M','GM01','GM03','C0']:
  nms_pred=[]
  for iid,ps in byim.items():
   b=torch.tensor([p['bbox'] for p in ps]);b[:,2:]+=b[:,:2]
   keep=tnms(b,torch.tensor([p['score'] for p in ps]),.7).tolist()
   nms_pred.extend(ps[k] for k in keep)
 variants=[('nms07',nms_pred),('scale095',pred),('nms07_scale095',nms_pred)]
 for label,ps in variants:
  if 'scale' in label:
   ps=copy.deepcopy(ps)
   for p in ps:
    x,y,w,h=p['bbox'];p['bbox']=[x+.025*w,y+.025*h,.95*w,.95*h]
  ap=coco_metrics(GT,ps);o=operating(match(ps,ground),.1,64)
  post.append(dict(name=j['name'],variant=j['variant'],seed=j['seed'],setting=label,AP=100*ap[0],AP75=100*ap[2],TP=o['TP']))
 print(j['name'],round(row['AP'],2),flush=True)
dumpcsv(O/'per_run.csv',rows);dumpcsv(O/'conditions.csv',conditions);dumpcsv(O/'localization.csv',locs);dumpcsv(O/'postprocess.csv',post)
summary={}
metrics=[k for k in rows[0] if k not in ['name','variant','seed']]
for v in dict.fromkeys(r['variant'] for r in rows):
 rr=[r for r in rows if r['variant']==v];summary[v]={'n':len(rr),'seeds':[r['seed'] for r in rr]}
 for key in metrics:
  vals=[r[key] for r in rr];summary[v][key]={'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0,'min':float(min(vals)),'max':float(max(vals))}
paired=[]
for r in rows:
 bv='yolov8s' if r['variant'].startswith('yolo') else 'dfine_s'
 b=next(x for x in rows if x['variant']==bv and x['seed']==r['seed'])
 paired.append(dict(name=r['name'],variant=r['variant'],seed=r['seed'],**{k:r[k]-b[k] for k in metrics}))
dumpcsv(O/'paired_deltas.csv',paired)
(O/'summary.json').write_text(json.dumps(summary,indent=2))
# Interaction estimates available only where all four cells completed.
interactions=[]
for strength in ['01','03']:
 for seed in [20260929,20260930,20261001]:
  cell={v:next((r for r in rows if r['variant']==v and r['seed']==seed),None) for v in ['dfine_s','M','G'+strength,'GM'+strength]}
  if all(cell.values()):
   interactions.append(dict(strength=strength,seed=seed,**{k:cell['GM'+strength][k]-cell['G'+strength][k]-cell['M'][k]+cell['dfine_s'][k] for k in ['AP','AP75','TP_0.1','size_abslog']}))
dumpcsv(O/'interactions.csv',interactions)
# Plot observed seeds, never imply missing-seed results.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
order=['dfine_s','M','G01','GM01']
fig,axes=plt.subplots(1,3,figsize=(15,4.7))
for ax,key,title in zip(axes,['AP','TP_0.1','size_abslog'],['Common validation AP50:95','TP at <=6 FP / 64 images','Mean absolute log-size error']):
 for i,v in enumerate(order):
  rr=[r for r in rows if r['variant']==v];ys=[r[key] for r in rr]
  ax.scatter([i+(r['seed']-20260929)*0 if False else i+[-.12,0,.12][[20260929,20260930,20261001].index(r['seed'])] for r in rr],ys,s=28)
  ax.plot([i-.2,i+.2],[np.mean(ys)]*2,color='black')
 ax.set_xticks(range(len(order)),order,rotation=55,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.25)
fig.suptitle('v3.1 relative-size loss: completed runs only; counts in summary.json',fontsize=11)
fig.tight_layout();fig.savefig(O/'ablation_overview.png',dpi=170)
print(json.dumps({k:{m:round(v[m]['mean'],3) for m in ['AP','AP75','TP_0.1','TP_lt8','size_abslog','rank_gap']} for k,v in summary.items()},indent=2))
