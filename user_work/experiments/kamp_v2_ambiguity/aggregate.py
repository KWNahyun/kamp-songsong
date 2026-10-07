from pathlib import Path
import json,csv,statistics
E=Path(__file__).parent;manifest=json.loads((E/'manifest.json').read_text());rows=[]
for j in manifest['jobs']:
 p=E/'runs'/j['name']/'common_eval/metrics.json'
 if p.exists():rows.append(json.loads(p.read_text()))
keys=['AP','AP50','AP75','raw_AP','raw_AP75','TP','FP','recall','precise_candidate','precise_leader','precise_after_nms']
flat=[{k:r[k] for k in ['name','variant','seed']+keys} for r in rows]
def dump(p,rs):
 if rs:
  with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
dump(E/'analysis/per_run.csv',flat);summary={}
for v in manifest['variants']:
 rr=[r for r in rows if r['variant']==v]
 if rr:summary[v]={'n':len(rr),**{k:{'mean':statistics.mean(r[k] for r in rr),'std':statistics.stdev(r[k] for r in rr) if len(rr)>1 else None} for k in keys}}
lookup={(r['variant'],r['seed']):r for r in rows};pairs=[]
for v,control in [('cx_l1half','mal'),('edge','mal'),('edge_half','edge'),('soft05','edge'),('soft10','edge'),('soft20','edge'),('hard10','edge'),('fgl05','mal'),('fgl10','mal'),('jitter10','mal'),('soft_fgl10','soft10'),('soft10','edge_half'),('fgl10','jitter10')]:
 for seed in [20260929,20260930,20261001]:
  if (v,seed) in lookup and (control,seed) in lookup:pairs.append(dict(variant=v,control=control,seed=seed,**{f'delta_{k}':lookup[v,seed][k]-lookup[control,seed][k] for k in keys}))
dump(E/'analysis/paired_deltas.csv',pairs);(E/'analysis/summary.json').write_text(json.dumps(dict(evaluated=len(rows),expected=len(manifest['jobs']),summary=summary,validation_exploration_only=True,test_used=False),indent=2));print('Aggregated',len(rows),'runs')
