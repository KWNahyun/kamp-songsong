"""Compare official-GT ranking and stage losses independently of annotation claims."""
from pathlib import Path
import csv,json,sys
import numpy as np
from scipy.stats import spearmanr
E=Path(__file__).parent;B=E.parent/'kamp_v2_baselines';O=E/'analysis';sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import iou,dumpcsv,nms
gt=json.loads((E/'original/annotations/val.json').read_text());rows=[];stage_rows=[];summaries=[]
for j in json.loads((E/'comparison_manifest.json').read_text())['jobs']:
 name=j['name'];variant=name.split('_seed')[0];pred=json.loads((E/'runs'/name/'common_eval/predictions_original.json').read_text());scores=[];qualities=[];local=[]
 for a in gt['annotations']:
  ps=[p for p in pred if p['image_id']==a['image_id']];near=[p for p in ps if iou(a['bbox'],p['bbox'])>=.1]
  best=max(ps,key=lambda p:iou(a['bbox'],p['bbox']));leader=max(near,key=lambda p:p['score'],default=None)
  rr={'run':name,'variant':variant,'seed':j['seed'],'gt_id':a['id'],'best_iou':iou(a['bbox'],best['bbox']),'best_score':best['score'],'leader_iou':iou(a['bbox'],leader['bbox']) if leader else 0,'leader_score':leader['score'] if leader else 0}
  rr['gap']=rr['best_iou']-rr['leader_iou'];rr['precise_wrong_leader']=rr['best_iou']>=.75 and rr['leader_iou']<.75;rows.append(rr);local.append(rr)
 for p in pred:
  quality=max(iou(p['bbox'],a['bbox']) for a in gt['annotations'] if a['image_id']==p['image_id']);scores.append(p['score']);qualities.append(quality)
 trace=(B if variant=='dfine_s' else E)/'trace'/name
 rr=list(csv.DictReader((trace/'gt_stages.csv').open()));by={}
 for r in rr:by.setdefault(r['gt_id'],{})[r['stage']]=r
 summary={'run':name,'variant':variant,'seed':j['seed'],'precise_wrong_leader':sum(r['precise_wrong_leader'] for r in local),'mean_IoU_gap':float(np.mean([r['gap'] for r in local])),'score_IoU_spearman':float(spearmanr(scores,qualities).statistic)}
 for floor in ['0.001','0.05']:
  key='best_iou_score_ge_'+floor
  summary['final75_floor'+floor]=sum(float(d['decoder_3'][key])>=.75 for d in by.values())
  summary['nms75_floor'+floor]=sum(float(d['final_nms07'][key])>=.75 for d in by.values())
  summary['nms_lost75_floor'+floor]=sum(float(d['decoder_3'][key])>=.75 and float(d['final_nms07'][key])<.75 for d in by.values())
  summary['middle_lost75_floor'+floor]=sum(any(float(d[s][key])>=.75 for s in ['pre_bbox','decoder_1','decoder_2']) and float(d['decoder_3'][key])<.75 for d in by.values())
 summaries.append(summary)
dumpcsv(O/'ranking_instances.csv',rows);dumpcsv(O/'selection_summary.csv',summaries)
print(json.dumps(summaries,indent=2))
