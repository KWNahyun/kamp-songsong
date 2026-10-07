from pathlib import Path
import csv,json,sys,copy,collections,hashlib,datetime
import torch
from torchvision.ops import nms
E=Path(__file__).parent;P=E.parent/'kamp_pilot_v1';V=E.parent/'kamp_ablation_v3';R=E.parents[1]
sys.path.insert(0,str(P));from analyze_failures import match,operating,dumpcsv
read=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig')))
gt=json.loads((R/'data/processed/kamp500_telea_v1/annotations/val.json').read_text());ground={im['id']:[a for a in gt['annotations'] if a['image_id']==im['id']] for im in gt['images']}
descriptors=read(P/'failure_analysis/gt_conditions.csv');axes=['machine','date','contrast_bin','size_bin']
rows=read(E/'analysis/per_run.csv');conditions=[];remaining=[];image_stats=[]
for r in rows:
 name=r['name'];path=E/'runs'/name if '_v31_' in name else (P/name if name=='dfine_s' else V/'runs'/name)
 pred=json.loads((path/'common_eval/predictions_original.json').read_text());byim=collections.defaultdict(list)
 for x in pred:byim[x['image_id']].append(x)
 nm=[]
 for ps in byim.values():
  boxes=torch.tensor([x['bbox'] for x in ps]);boxes[:,2:]+=boxes[:,:2]
  ix=nms(boxes,torch.tensor([x['score'] for x in ps]),.7).tolist();nm.extend(ps[i] for i in ix)
 for setting,ps in [('raw',pred),('nms07',nm),('scale095',pred),('nms07_scale095',nm)]:
  ps=copy.deepcopy(ps)
  if 'scale' in setting:
   for x in ps:
    a,b,w,h=x['bbox'];x['bbox']=[a+.025*w,b+.025*h,.95*w,.95*h]
  op=operating(match(ps,ground),.1,64);meta={k:r[k] for k in ['name','variant','seed']};meta['setting']=setting
  for axis in axes:
   for val in sorted({d[axis] for d in descriptors}):
    ids={int(d['gt_id']) for d in descriptors if d[axis]==val};conditions.append(dict(meta,axis=axis,value=val,GT=len(ids),TP=len(ids&op['matched'])))
  for d in descriptors:
   if int(d['gt_id']) not in op['matched']:remaining.append(dict(meta,**d))
  image_stats.append(dict(meta,TP=op['TP'],FP=op['FP'],threshold=op['threshold'],image_alarm=len({x['image_id'] for x in ps if x['score']>=op['threshold']})))
dumpcsv(E/'analysis/postprocess_conditions.csv',conditions);dumpcsv(E/'analysis/postprocess_remaining_FN.csv',remaining);dumpcsv(E/'analysis/postprocess_image_stats.csv',image_stats)
tr=[]
for f in sorted((E/'trace').glob('*/verification.json')):
 m=json.loads(f.read_text());rs=read(f.parent/'gt_stages.csv');ts=read(f.parent/'query_transitions.csv')
 stages={st:{int(x['gt_id']):x for x in rs if x['stage']==st} for st in {x['stage'] for x in rs}}
 final=stages['decoder_3'];out={k:m[k] for k in ['name','variant','seed']}
 out.update(no50=0,between50and75=0,good75_bad_leader=0,good75_good_leader=0,multiple50_score05=0,nms_lost75_at001=0,topk_missing50_recovered_final=0,layer2_good75_same_query_lost=0,layer2_good75_all_final_lost=0)
 for gid,x in final.items():
  best=float(x['best_iou_score_ge_0.05']);lead=float(x['leader_iou'])
  key='no50' if best<.5 else ('between50and75' if best<.75 else ('good75_bad_leader' if lead<.75 else 'good75_good_leader'))
  out[key]+=1;out['multiple50_score05']+=int(x['count_iou50_score_ge_0.05'])>=2
  out['nms_lost75_at001']+=float(x['best_iou_score_ge_0.001'])>=.75 and float(stages['final_nms07'][gid]['best_iou'])<.75
  out['topk_missing50_recovered_final']+=float(stages['topk'][gid]['best_iou'])<.5 and float(x['best_iou'])>=.5
 for x in ts:
  if x['from_stage']=='decoder_2':
   out['layer2_good75_same_query_lost']+=float(x['from_iou'])>=.75 and float(x['same_query_final_iou'])<.75
   out['layer2_good75_all_final_lost']+=float(x['from_iou'])>=.75 and float(x['any_query_final_best_iou'])<.75
 assert sum(out[x] for x in ['no50','between50and75','good75_bad_leader','good75_good_leader'])==142
 tr.append(out)
dumpcsv(E/'analysis/trace_failure_partition.csv',tr)
manifest={'test_used':False,'trained_successfully':6,'evaluated_successfully':6,'trace_forward_equivalence_checks':384,'trace_output_max_difference':0,'trace_score_floor':.05,'trace_notes':'Per-GT oracle, not one-to-one recall. NMS loss uses same .001 floor. Encoder expanded-grid outputs are diagnostic counterfactuals.','source_hashes':{str(f.relative_to(E)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [E/'status.json',E/'analysis/per_run.csv',E/'analysis/postprocess.csv',E/'trace/stage_summary.csv']}}
(E/'analysis/final_audit_manifest.json').write_text(json.dumps(manifest,indent=2))
print('TRACE',json.dumps(tr,ensure_ascii=False))
print('FINAL CONDITIONS')
for v in ['dfine_s','M','G01','GM01']:
 for axis in ['machine','contrast_bin','size_bin']:
  for val in sorted({x['value'] for x in conditions if x['axis']==axis}):
   rr=[x for x in conditions if x['variant']==v and x['setting']=='nms07_scale095' and x['axis']==axis and x['value']==val]
   print(v,axis,val,rr[0]['GT'],sum(x['TP'] for x in rr)/len(rr))
