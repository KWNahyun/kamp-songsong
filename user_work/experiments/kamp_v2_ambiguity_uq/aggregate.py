import json,csv
from pathlib import Path
import pandas as pd
E=Path(__file__).parent;A=E.parent/'kamp_v2_ambiguity';jobs=json.loads((E/'manifest.json').read_text())['jobs'];rows=[];paired=[]
keys=['AP','AP75','TP','precise_candidate','precise_leader','precise_after_nms']
for j in jobs:
 r=json.loads((E/'runs_frozen'/j['name']/'common_eval/metrics.json').read_text()); rows.append(r)
 b=json.loads((A/'runs'/f"{j['variant']}_seed{j['seed']}"/'common_eval/metrics.json').read_text())
 paired.append(dict(variant=j['variant'],seed=j['seed'],**{k:r[k]-b[k] for k in keys}))
D=pd.DataFrame(rows);(E/'analysis').mkdir(exist_ok=True);D.to_csv(E/'analysis/per_run.csv',index=False);pd.DataFrame(paired).to_csv(E/'analysis/UQ_gain.csv',index=False)
means=D.groupby('variant')[keys].agg(['mean','std']);means.to_csv(E/'analysis/summary.csv');comparisons=[]
for control in ['mal','edge']:
 s=D[D.variant=='soft10'].set_index('seed');b=D[D.variant==control].set_index('seed');delta=s[keys]-b[keys]
 for seed,r in delta.iterrows():comparisons.append(dict(control=control,seed=int(seed),**r.to_dict()))
 outcome=dict(control=control,mean_deltas=delta.mean().to_dict(),positive_seeds=(delta>0).sum().to_dict(),passes=bool(delta.AP.mean()>0 and delta.AP75.mean()>0 and (delta.AP>0).sum()>=2 and (delta.AP75>0).sum()>=2 and delta.TP.mean()>=0))
 (E/'analysis'/f'decision_vs_{control}.json').write_text(json.dumps(outcome,indent=2))
pd.DataFrame(comparisons).to_csv(E/'analysis/paired_UQ.csv',index=False)
(E/'analysis/completion.json').write_text(json.dumps(dict(runs=len(rows),test_used=False,report_update_pending=True),indent=2))
