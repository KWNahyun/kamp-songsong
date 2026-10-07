from pathlib import Path
import sys,json,csv,hashlib,collections
import numpy as np
import torch
E=Path(__file__).parent;P=E.parent/'kamp_pilot_v1';O=E/'analysis'
sys.path.insert(0,str(P));sys.path.insert(0,'/home/viplab/contest/models/D-FINE')
from analyze_failures import match,operating,dumpcsv
from torchvision.ops import nms
from loss_patch import size_loss
# Negative finite widths are outside log-size's domain. Reproduce the missing guard,
# not claim this synthetic input was captured in the original failed training.
x=torch.tensor([[.5,.5,-.01,.02]],requires_grad=True);g=torch.tensor([[.5,.5,.02,.02]])
unit={'synthetic_negative_width_loss_is_nonfinite':not torch.isfinite(size_loss(x,g)).item(),'actual_failed_tensor_captured':False}
(O/'numeric_domain_audit.json').write_text(json.dumps(unit,indent=2))
gt=json.loads(Path('/home/viplab/contest/data/processed/kamp500_telea_v1/annotations/val.json').read_text());ground={i['id']:[a for a in gt['annotations'] if a['image_id']==i['id']] for i in gt['images']}
desc={int(r['gt_id']):r for r in csv.DictReader((P/'failure_analysis/gt_conditions.csv').open(encoding='utf-8-sig'))}
rows=list(csv.DictReader((O/'per_run.csv').open(encoding='utf-8-sig')));fail=[];summ=[]
for r in rows:
 if r['variant'] not in ['dfine_s','M']:continue
 run=P/'dfine_s' if r['name']=='dfine_s' else E/'runs'/r['name']
 preds=json.loads((run/'common_eval/predictions_original.json').read_text());by=collections.defaultdict(list)
 for p in preds:by[p['image_id']].append(p)
 keep=[]
 for ps in by.values():
  b=torch.tensor([p['bbox'] for p in ps]);b[:,2:]+=b[:,:2]
  for k in nms(b,torch.tensor([p['score'] for p in ps]),.7).tolist():
   p=ps[k].copy();x,y,w,h=p['bbox'];p['bbox']=[x+.025*w,y+.025*h,.95*w,.95*h];keep.append(p)
 op=operating(match(keep,ground),.1,64)
 for aid,d in desc.items():
  if aid not in op['matched']:fail.append(dict(name=r['name'],**d))
 summ.append(dict(name=r['name'],variant=r['variant'],seed=r['seed'],TP=op['TP'],TP_lt8=sum(k in op['matched'] and d['size_bin']=='lt8' for k,d in desc.items()),image_alarm=len({p['image_id'] for p in keep if p['score']>=op['threshold']})))
dumpcsv(O/'postprocess_remaining_FN.csv',fail);dumpcsv(O/'postprocess_conditions.csv',summ)
s=json.loads((O/'summary.json').read_text())
for k,v in s.items():v['effective_distinct_prediction_repeats']=1 if k.startswith('yolo') else v['n']
(O/'summary.json').write_text(json.dumps(s,indent=2))
inputs=list((E/'runs').glob('*/common_eval/predictions_original.json'))+[P/v/'common_eval/predictions_original.json' for v in ['dfine_s','dfine_s_p2','yolov8s','yolov8s_p2']]
(O/'manifest.json').write_text(json.dumps({'validation_images':64,'GT':142,'test_used':False,'new_completed_runs':23,'new_failed_runs':3,'evaluated_records_including_pilot':27,'distinct_prediction_files':len(set(hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs)),'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'postprocess':'NMS .7 then width/height .95; fixed from prior exploratory work, not independent validation','failed_runs_excluded_from_performance_not_hidden':True,'diagnostic_reproduction_excluded':True},indent=2))
# Revised plot shows YOLO once, rather than implying independent repeats.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
order=['dfine_s','C0','G01','G03','M','GM01','GM03','dfine_s_p2','yolov8s','yolov8s_p2']
fig,axs=plt.subplots(1,3,figsize=(15,4.7))
for ax,key,title in zip(axs,['AP','TP_0.1','size_abslog'],['Common validation AP50:95','TP at <=6 FP / 64 images','Mean absolute log-size error']):
 for i,v in enumerate(order):
  rr=[r for r in rows if r['variant']==v and (not v.startswith('yolo') or r['seed']=='20260929')];ys=[float(r[key]) for r in rr]
  for r,y in zip(rr,ys):
   ix=[20260929,20260930,20261001].index(int(r['seed']));ax.scatter(i+[-.12,0,.12][ix],y,s=28,color=['#3178b5','#e07c24','#389b48'][ix])
  ax.plot([i-.2,i+.2],[np.mean(ys)]*2,color='black')
 ax.set_xticks(range(len(order)),order,rotation=55,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.25)
fig.suptitle('Completed runs: G01/G03/GM01 n=2; other D-FINE n=3; YOLO shown once (identical reruns)',fontsize=10)
fig.tight_layout();fig.savefig(O/'ablation_overview.png',dpi=170)
print(summ)
