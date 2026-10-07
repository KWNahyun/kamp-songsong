"""Score exported human clicks against official TXT boxes, not physical defect truth."""
from pathlib import Path
import argparse,json,math
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
F=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--software-test',action='store_true');a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
key=json.loads((F/'analysis/review_study_admin/answer_key.json').read_text());rows=[];participants=set()
for file in a.results:
 data=json.loads(file.read_text());assert bool(data.get('software_test',False))==a.software_test,'Software-test export needs explicit --software-test';assert data['version']=='kamp-review-v1-20261003' and not data['preview'],'Preview exports cannot be human evidence'
 pid=data['participant'];assert pid not in participants,'One cumulative export per participant required';participants.add(pid);seen=set()
 for r in data['results']:
  cid=r['case_id'];assert cid in key and cid not in seen;seen.add(cid);k=key[cid];condition=['raw','boxes','regions'][(k['assignment']+data['armIndex'])%3];assert r['condition']==condition
  clicks=r['clicks'];assert all(math.isfinite(c[v]) for c in clicks for v in ['x','y']);assert all(0<=c['x']<=k['width'] and 0<=c['y']<=k['height'] for c in clicks);secs=r['active_seconds'];assert math.isfinite(secs) and secs>=0
  hit=np.array([[x<=c['x']<=x+w and y<=c['y']<=y+h for x,y,w,h in k['GT']] for c in clicks],dtype=int)
  if clicks:
   ii,jj=linear_sum_assignment(-hit);tp=int(hit[ii,jj].sum())
  else:tp=0
  rows.append(dict(participant=pid,arm=data['armIndex'],previous_exposure=data['familiar'],case_id=cid,group=k['group'],condition=condition,GT=len(k['GT']),TP_click=tp,unmatched_clicks=len(clicks)-tp,FN_click=len(k['GT'])-tp,active_seconds=secs,unmarked=r['unmarked']))
assert rows,'No completed cases'
a.output.mkdir(parents=True);d=pd.DataFrame(rows);d.to_csv(a.output/'per_case.csv',index=False)
z=d.groupby(['participant','condition']).agg(images=('case_id','size'),GT=('GT','sum'),TP_click=('TP_click','sum'),FN_click=('FN_click','sum'),unmatched_clicks=('unmatched_clicks','sum'),median_active_seconds=('active_seconds','median'),total_active_seconds=('active_seconds','sum')).reset_index();z['click_recall']=z.TP_click/z.GT;z.to_csv(a.output/'summary.csv',index=False)
(a.output/'interpretation.json').write_text(json.dumps({'metric':'one-to-one click containment in official TXT bbox','not_detector_AP':True,'not_physical_detection_ground_truth':True,'not_factory_trial':True,'participants':len(participants),'rows':len(d),'uncertainty':'no independent image/binomial confidence intervals; repeated captures and shared participants require group-aware analysis','human_results_provided_by_user':not a.software_test,'software_test_only':a.software_test},indent=2));print(z.to_string(index=False))
