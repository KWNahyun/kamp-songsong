"""Exploratory val-only review prioritization, never an automatic-pass policy."""
from pathlib import Path
import json,csv,sys
import numpy as np
from scipy.optimize import linear_sum_assignment
R=Path('/home/viplab/contest');E=Path(__file__).parent;B=E.parent/'kamp_v2_baselines';U=E.parent/'kamp_v2_uq';sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import iou,nms,match,operating,dumpcsv
G=json.loads((U/'original/annotations/val.json').read_text());ims={i['id']:i for i in G['images']};ground={i:[a for a in G['annotations'] if a['image_id']==i] for i in ims};conditions={int(r['gt_id']):r for r in csv.DictReader((U/'analysis/gt_conditions.csv').open(encoding='utf-8-sig'))}
seeds=[20260929,20260930,20261001];failures=[];allrows=[];curves=[];checks=[]
# Audit across existing 4 architecture settings, plus MAL+UQ.
for seed in seeds:
 outputs={}
 for model in ['yolov8s','yolov8s_p2','dfine_s','dfine_s_p2','dfine_mal_UQ']:
  parent=U/'runs_frozen' if model=='dfine_mal_UQ' else B/'runs';name=f'{model}_seed{seed}';pred=json.loads((parent/name/'common_eval/predictions_original.json').read_text());pp=nms(pred,.7);op=operating(match(pp,ground),.1,66);outputs[model]=pp
  for a in G['annotations']:
   ps=[p for p in pp if p['image_id']==a['image_id']];near=[p for p in ps if iou(a['bbox'],p['bbox'])>=.1];leader=max(near,key=lambda p:p['score'],default=None);best=max([iou(a['bbox'],p['bbox']) for p in ps],default=0)
   failures.append(dict(seed=seed,model=model,gt_id=a['id'],image_id=a['image_id'],machine=conditions[a['id']]['machine'],matched=a['id'] in op['matched'],best_iou_after_nms=best,leader_iou=iou(a['bbox'],leader['bbox']) if leader else 0,leader_score=leader['score'] if leader else None,threshold=op['threshold']))
  if model=='dfine_mal_UQ': reference=op
 checks.append(dict(seed=seed,TP=reference['TP'],FP=reference['FP'],threshold=reference['threshold']))
 rows=[]
 for iid,im in ims.items():
  # Fixed signal cutoff, not a GT-tuned operating threshold.
  d=[p for p in outputs['dfine_mal_UQ'] if p['image_id']==iid and p['score']>=.25];y=[p for p in outputs['yolov8s'] if p['image_id']==iid and p['score']>=.25]
  if d and y:
   matrix=np.array([[iou(a['bbox'],b['bbox']) for b in y] for a in d]);ii,jj=linear_sum_assignment(-matrix);agreement=float(matrix[ii,jj].sum()/max(len(d),len(y)))
  else:agreement=1. if not d and not y else 0.
  # Both empty gets high separate empty signal; low-confidence catches that.
  row=dict(seed=seed,image_id=iid,stem=Path(im['file_name']).stem,machine=conditions[ground[iid][0]['id']]['machine'],group=conditions[ground[iid][0]['id']]['group'],low_confidence=1-min([p['score'] for p in d],default=0),cross_model_disagreement=1-agreement,count_difference=abs(len(d)-len(y)),dfine_count=len(d),yolo_count=len(y),missed_gt=sum(a['id'] not in reference['matched'] for a in ground[iid]),gt_count=len(ground[iid]),final_alarm_boxes=sum(p['image_id']==iid and p['score']>=reference['threshold'] for p in outputs['dfine_mal_UQ']))
  rows.append(row)
 allrows.extend(rows);rng=np.random.default_rng(27001);counts=np.array([r['missed_gt'] for r in rows]);total=int(counts.sum());order_random=np.array([rng.permutation(66) for _ in range(10000)])
 for signal in ['low_confidence','cross_model_disagreement','count_difference']:
  # Ties broken by stable ID for concrete list; report boundary tie range too.
  order=sorted(range(66),key=lambda i:(-rows[i][signal],rows[i]['image_id']))
  for budget in [4,7,14,20]:
   chosen=order[:budget];cut=rows[order[budget-1]][signal];above=[i for i in order if rows[i][signal]>cut];tied=[i for i in order if rows[i][signal]==cut];slots=budget-len(above);tiecounts=sorted(counts[tied]);lower=int(counts[above].sum()+sum(tiecounts[:slots]));upper=int(counts[above].sum()+sum(tiecounts[-slots:]));random_found=counts[order_random[:,:budget]].sum(1)
   curves.append(dict(seed=seed,signal=signal,budget_images=budget,review_fraction=budget/66,miss_gt_total=total,miss_gt_in_selected=int(counts[chosen].sum()),miss_images_in_selected=int((counts[chosen]>0).sum()),fraction_missed_gt_covered=float(counts[chosen].sum()/total),random_expected=float(random_found.mean()),random_p025=float(np.quantile(random_found,.025)),random_p975=float(np.quantile(random_found,.975)),tie_min=lower,tie_max=upper,selected_image_ids=','.join(str(rows[i]['image_id']) for i in chosen)))
dumpcsv(E/'gt_failures.csv',failures);dumpcsv(E/'image_signals.csv',allrows);dumpcsv(E/'review_curves.csv',curves)
consensus=[]
for a in G['annotations']:
 fs=[r for r in failures if r['gt_id']==a['id']];base=[r for r in fs if r['model']!='dfine_mal_UQ'];uq=[r for r in fs if r['model']=='dfine_mal_UQ'];nbase=sum(not r['matched'] for r in base);nuq=sum(not r['matched'] for r in uq)
 if nbase or nuq:consensus.append(dict(gt_id=a['id'],image_id=a['image_id'],stem=Path(ims[a['image_id']]['file_name']).stem,machine=conditions[a['id']]['machine'],group=conditions[a['id']]['group'],bbox=a['bbox'],baseline_misses_out_of12=nbase,mal_uq_misses_out_of3=nuq,mal_uq_best_iou_after_nms=[r['best_iou_after_nms'] for r in uq],mal_uq_leader_iou=[r['leader_iou'] for r in uq]))
(E/'repeat_failures.json').write_text(json.dumps(consensus,ensure_ascii=False,indent=2));(E/'verification.json').write_text(json.dumps(dict(operating_points=checks,test_used=False,signal_cutoff=.25,policy_fitted=False,random_draws=10000,scope='validation exploratory retrieval of missed-GT images, not actual reinspection recovery'),indent=2));print(json.dumps({'recurrent':consensus,'budget7':[x for x in curves if x['budget_images']==7]},ensure_ascii=False,indent=2))
