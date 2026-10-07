from pathlib import Path
import sys,json,csv,collections
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src')]
from metrics import match,nms,iou,dumpcsv,coco_metrics
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
G=json.loads((R/'reference_results/val_annotations.json').read_text());raw=json.loads((R/'reference_results/val_raw_predictions.json').read_text());pp=nms(raw,.7);t=json.loads((R/'manifests/model.json').read_text())['val_threshold'];ground={im['id']:[a for a in G['annotations'] if a['image_id']==im['id']] for im in G['images']};events=match(pp,ground);ev=[e for e in events if e['score']>=t];matched={e['gt_id'] for e in ev if e['tp']};ims={im['id']:im for im in G['images']};M={r['image_id']:r for r in csv.DictReader((R/'manifests/split_manifest.csv').open())};rows=[]
for a in G['annotations']:
 im=ims[a['image_id']];meta=M[Path(im['file_name']).stem];ps=[p for p in raw if p['image_id']==a['image_id']];pn=[p for p in pp if p['image_id']==a['image_id']];pa=[p for p in pn if p['score']>=t];best=lambda ps:max((iou(a['bbox'],p['bbox']) for p in ps),default=0);br,bn,ba=best(ps),best(pn),best(pa);near=[p for p in ps if iou(a['bbox'],p['bbox'])>=.3];lead=max(near,key=lambda p:p['score'],default=None)
 stage='TP' if a['id'] in matched else ('no_near_candidate' if br<.3 else ('localization' if br<.5 else ('NMS' if bn<.5 else ('threshold' if ba<.5 else 'matching_conflict'))))
 x,y,w,h=a['bbox'];short=min(w,h);rows.append(dict(gt_id=a['id'],image_id=a['image_id'],file_name=im['file_name'],machine=meta['machine'],group=meta['group'],size_bin='<8' if short<8 else ('8-12' if short<12 else ('12-16' if short<16 else '>=16')),width=w,height=h,x_norm=(x+w/2)/im['width'],y_norm=(y+h/2)/im['height'],TP=a['id'] in matched,stage=stage,best_raw_iou=br,best_nms_iou=bn,best_alarm_iou=ba,leader_iou=iou(a['bbox'],lead['bbox']) if lead else 0,ranking_gap=br-(iou(a['bbox'],lead['bbox']) if lead else 0),product_alarm=bool(pa)))
dumpcsv(R/'analysis/gt_failures.csv',rows);dumpcsv(R/'analysis/FP.csv',[e for e in ev if not e['tp']]);groups=sorted({r['group'] for r in rows});rng=np.random.default_rng(20261003);boots=rng.choice(groups,(3000,len(groups)),replace=True);table=[]
for axis in ['machine','size_bin']:
 for val in sorted({r[axis] for r in rows}):
  subset=[r for r in rows if r[axis]==val];tp=sum(r['TP'] for r in subset);by={g:[r for r in subset if r['group']==g] for g in groups};ratios=[]
  for draw in boots:
   n=sum(len(by[g]) for g in draw)
   if n:ratios.append(sum(sum(r['TP'] for r in by[g]) for g in draw)/n)
  ci=np.quantile(ratios,[.025,.975]);table.append(dict(axis=axis,condition=val,images=len({r['image_id'] for r in subset}),groups=len({r['group'] for r in subset}),GT=len(subset),TP=tp,FN=len(subset)-tp,recall=tp/len(subset),cluster_bootstrap_low=ci[0],cluster_bootstrap_high=ci[1],warning='Exploratory; 11 validation groups; zero observed failures does not establish zero risk; sparse bins not generalizable'))
dumpcsv(R/'analysis/conditions.csv',table)
curve=[];tp=fp=0
for score in sorted({e['score'] for e in events},reverse=True):
 block=[e for e in events if e['score']==score];tp+=sum(e['tp'] for e in block);fp+=sum(not e['tp'] for e in block);curve.append(dict(threshold=score,TP=tp,FP=fp,recall=tp/144,FP_per_image=fp/66))
dumpcsv(R/'analysis/FROC.csv',curve)
fig,ax=plt.subplots(figsize=(6,4));ax.step([r['FP_per_image'] for r in curve],[r['recall'] for r in curve],where='post');ax.scatter([6/66],[142/144],color='red',label='Frozen validation threshold');ax.set(xlim=(0,1),ylim=(.8,1.005),xlabel='False positive boxes / image',ylabel='Object recall at IoU 0.5',title='Validation FROC: MAL+UQ seed 20260930');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(R/'analysis/FROC.png',dpi=160)
summary=dict(FN=[r for r in rows if not r['TP']],FP_categories=dict(collections.Counter(e['category'] for e in ev if not e['tp'])),conditions=table,threshold_frozen=t,test_used=False);(R/'analysis/summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
