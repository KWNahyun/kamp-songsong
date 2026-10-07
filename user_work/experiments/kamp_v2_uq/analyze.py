"""Validation only: frozen predictions, condition diagnosis and common postprocessing."""
from pathlib import Path
import sys,json,csv,collections,hashlib,copy
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=Path(__file__).parent; R=E.parents[1]; O=E/'analysis';O.mkdir(exist_ok=True)
sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import iou,match,operating,coco_metrics,nms,dumpcsv
gt=json.loads((E/'original/annotations/val.json').read_text()); ims={i['id']:i for i in gt['images']}
ground={i:[a for a in gt['annotations'] if a['image_id']==i] for i in ims}
meta={r['image_id']:r for r in csv.DictReader((R/'kamp_xray_v2/split_manifest.csv').open())}
conditions={}; train_contrast=[]
for split in ['train','val']:
 data=json.loads((E/'original/annotations'/f'{split}.json').read_text())
 for im in data['images']:
  arr=np.array(Image.open(R/'kamp_xray_v2/images'/split/im['file_name']),dtype=float)
  for a in [a for a in data['annotations'] if a['image_id']==im['id']]:
   x,y,w,h=a['bbox'];x1,y1=max(0,int(x)),max(0,int(y));x2,y2=min(arr.shape[1],int(np.ceil(x+w))),min(arr.shape[0],int(np.ceil(y+h)))
   m=max(3,round(max(w,h)/2)); rx,ry=max(0,x1-m),max(0,y1-m); patch=arr[ry:min(arr.shape[0],y2+m),rx:min(arr.shape[1],x2+m)]
   mask=np.ones(patch.shape,bool);mask[y1-ry:y2-ry,x1-rx:x2-rx]=False;ring=patch[mask];inside=arr[y1:y2,x1:x2]
   med=np.median(ring);v=float(abs(np.median(inside)-med)/(1.4826*np.median(abs(ring-med))+1))
   if split=='train':train_contrast.append(v)
   else:
    r=meta[Path(im['file_name']).stem];short=min(w,h)
    conditions[a['id']]={'gt_id':a['id'],'image_id':a['image_id'],'stem':r['image_id'],'machine':r['machine'],'date':r['date'],'group':r['group'],'bbox_short_px':short,'size_bin':'lt8' if short<8 else '8to12' if short<12 else '12to16' if short<16 else 'ge16','contrast_proxy':v}
cuts=np.quantile(train_contrast,[1/3,2/3]).tolist()
for d in conditions.values(): d['contrast_bin']=['low','mid','high'][int(np.searchsorted(cuts,d['contrast_proxy']))]
dumpcsv(O/'gt_conditions.csv',list(conditions.values()))
metrics=[]; cond=[]; diagnoses=[]; geometry=[]; fps=[]; curves={}; manifest=[]
for j in json.loads((E/'comparison_manifest.json').read_text())['jobs']:
 name=j['name']; variant=name.split('_seed')[0]; run=E/'runs'/name
 pfile=run/'common_eval/predictions_original.json';pred=json.loads(pfile.read_text());manifest.append({'run':name,'predictions_sha256':hashlib.sha256(pfile.read_bytes()).hexdigest()})
 if j['family']=='uq':
  quality_rows=[json.loads(x) for x in (run/'log.jsonl').read_text().splitlines()]
  curves[name]={'quality_loss':[x['train_selector_quality'] for x in quality_rows],'validation_epoch_selection':False}
 else:
  if j['family']=='yolo':
   history=list(csv.DictReader((run/'results.csv').open()));aps=[float(x['metrics/mAP50-95(B)']) for x in history];loss=[float(x['train/box_loss']) for x in history]
  else:
   history=[json.loads(x) for x in (run/'log.txt').read_text().splitlines()];aps=[x['test_coco_eval_bbox'][0] for x in history];loss=[x['train_loss_bbox'] for x in history]
  curves[name]={'AP':aps,'bbox_loss':loss,'best_epoch':int(np.argmax(aps))+1,'last_AP':aps[-1],'best_AP':max(aps),'last5_mean':float(np.mean(aps[-5:])),'previous5_mean':float(np.mean(aps[-10:-5]))}
 for setting in ['native','nms07','nms07_scale095']:
  pp=pred if setting=='native' else nms(pred,.7)
  if setting.endswith('095'):
   pp=copy.deepcopy(pp)
   for p in pp:
    x,y,w,h=p['bbox'];p['bbox']=[x+w*.025,y+h*.025,w*.95,h*.95]
  ap=coco_metrics(gt,pp);events=match(pp,ground);op=operating(events,.1,len(ims));op5=operating(events,.5,len(ims))
  met={'run':name,'variant':variant,'seed':j['seed'],'setting':setting,'AP':ap[0]*100,'AP50':ap[1]*100,'AP75':ap[2]*100,'TP_FP6':op['TP'],'FP':op['FP'],'threshold':op['threshold'],'TP_FP33':op5['TP'],'TP_conf025':sum(x['tp'] for x in events if x['score']>=.25),'FP_conf025':sum(1-x['tp'] for x in events if x['score']>=.25)}
  for cut in [.3,.75]:met['TP_FP6_iou'+str(cut)]=operating(match(pp,ground,cut),.1,len(ims))['TP']
  metrics.append(met)
  for axis in ['machine','size_bin','contrast_bin','group']:
   for value in sorted({d[axis] for d in conditions.values()}):
    gg=[d for d in conditions.values() if d[axis]==value];tp=sum(d['gt_id'] in op['matched'] for d in gg)
    cond.append({'run':name,'variant':variant,'seed':j['seed'],'setting':setting,'axis':axis,'value':value,'GT':len(gg),'TP':tp,'FN':len(gg)-tp,'recall':tp/len(gg)})
  if setting=='nms07':
   for rank,ev in enumerate([ev for ev in events if not ev['tp']][:12],1):fps.append({'run':name,'rank':rank,**ev})
  byim={i:[p for p in pp if p['image_id']==i] for i in ims}
  for a in gt['annotations']:
   ps=byim[a['image_id']];ious=np.array([iou(a['bbox'],p['bbox']) for p in ps]);near=[k for k,v in enumerate(ious) if v>=.1]; leader=max(near,key=lambda k:ps[k]['score'],default=None);best=max(ious,default=0)
   reason='matched' if a['id'] in op['matched'] else 'no_final_near_candidate' if best<.1 else 'localization_below_05' if best<.5 else 'score_or_competing_match'
   diagnoses.append({'run':name,'variant':variant,'seed':j['seed'],'setting':setting,'gt_id':a['id'],'best_iou':float(best),'leader_iou':float(ious[leader]) if leader is not None else 0,'n_iou50':int((ious>=.5).sum()),'matched_FP6':a['id'] in op['matched'],'reason':reason})
   if setting=='native' and leader is not None:
    p=ps[leader];x,y,w,h=p['bbox'];gx,gy,gw,gh=a['bbox'];cx,cy=x+w/2,y+h/2;gcx,gcy=gx+gw/2,gy+gh/2
    geometry.append({'run':name,'gt_id':a['id'],'score':p['score'],'IoU':iou(p['bbox'],a['bbox']),'center_error_px':float(np.hypot(cx-gcx,cy-gcy)),'width_ratio':w/gw,'height_ratio':h/gh,'center_oracle':iou([gcx-w/2,gcy-h/2,w,h],a['bbox']),'size_oracle':iou([cx-gw/2,cy-gh/2,gw,gh],a['bbox'])})
 print('analyzed',name,flush=True)
dumpcsv(O/'metrics.csv',metrics);dumpcsv(O/'conditions.csv',cond);dumpcsv(O/'gt_diagnostics.csv',diagnoses);dumpcsv(O/'geometry.csv',geometry);dumpcsv(O/'top_fp.csv',fps)
(O/'curves.json').write_text(json.dumps(curves,indent=2));(O/'prediction_hashes.json').write_text(json.dumps(manifest,indent=2))
aggregate=[]
for variant in sorted({m['variant'] for m in metrics}):
 for setting in ['native','nms07','nms07_scale095']:
  rr=[m for m in metrics if m['variant']==variant and m['setting']==setting];a={'variant':variant,'setting':setting}
  for k in ['AP','AP50','AP75','TP_FP6','TP_FP33','TP_FP6_iou0.3','TP_FP6_iou0.75']:
   a[k]=float(np.mean([x[k] for x in rr]));a[k+'_std']=float(np.std([x[k] for x in rr],ddof=1))
  aggregate.append(a)
dumpcsv(O/'aggregate.csv',aggregate)
paired=[]
for control,treatment in [('dfine_s','dfine_base_UQ'),('dfine_M','dfine_mal_UQ'),('dfine_base_UQ','dfine_mal_UQ')]:
 for setting in ['native','nms07','nms07_scale095']:
  for seed in [20260929,20260930,20261001]:
   b=next(m for m in metrics if m['variant']==control and m['setting']==setting and m['seed']==seed);p=next(m for m in metrics if m['variant']==treatment and m['setting']==setting and m['seed']==seed)
   paired.append({'control':control,'treatment':treatment,'setting':setting,'seed':seed,**{'delta_'+k:p[k]-b[k] for k in ['AP','AP75','TP_FP6']}})
dumpcsv(O/'uq_paired.csv',paired)
fig,axs=plt.subplots(1,3,figsize=(15,4))
for v in sorted({m['variant'] for m in metrics}):
 rr=[m for m in metrics if m['variant']==v and m['setting']=='nms07'];xs=np.arange(3)
 axs[0].plot(xs,[m['AP'] for m in rr],'o-',label=v);axs[1].plot(xs,[m['AP75'] for m in rr],'o-',label=v)
 hs=[d['AP'] for k,d in curves.items() if k.split('_seed')[0]==v and 'AP' in d]
 if hs:axs[2].plot(np.arange(1,31),np.mean(hs,axis=0)*100,label=v)
for ax in axs[:2]:ax.set_xticks(range(3),['seed 1','seed 2','seed 3'])
axs[0].set_title('Common NMS 0.7: AP');axs[1].set_title('Common NMS 0.7: AP75');axs[2].set_title('Native validation AP learning curves');axs[2].set_xlabel('Epoch');axs[2].legend(fontsize=7)
fig.tight_layout();fig.savefig(O/'comparison.png',dpi=170);plt.close(fig)
(O/'summary.json').write_text(json.dumps({'aggregate':aggregate,'contrast_train_tertiles':cuts,'test_used':False,'prediction_hashes':manifest,'cautions':['validation-selected checkpoints and thresholds','contrast is GT-box median vs surrounding ring robust contrast, not true defect CNR','YOLO native predictions already include NMS; pre-NMS candidates not measured','encoder full-grid tracing is counterfactual, not actual runtime candidates']},indent=2))
