"""Cross-check saved evidence and report links; no training or inference."""
from pathlib import Path
import json,re,sys,hashlib
import pandas as pd
F=Path(__file__).resolve().parents[1];R=F.parent;sys.path.insert(0,str(F/'src'))
from metrics import nms
checks=[]
def check(name,ok):
 assert ok,name
 checks.append({'check':name,'passed':bool(ok)})
s=(R/'제출용_결과보고서_사용자담당.md').read_text()
for kind in ['표','그림']:
 nums=[int(x) for x in re.findall(r'\*\*'+kind+r' (\d+)\.',s)];check(kind+' captions sequential',nums==list(range(1,len(nums)+1)))
imgs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',s);check('13 figures exist',len(imgs)==13 and all(Path(x).exists() and Path(x).with_suffix('.pdf').exists() for x in imgs))
u=pd.read_csv(R/'experiments/kamp_v2_uq/analysis/metrics.csv');u=u[u.setting=='nms07']
for variant,expected in [('dfine_M',38.36),('dfine_mal_UQ',39.38)]:
 d=u[u.variant==variant];check(variant+' AP mean',round(d.AP.mean(),2)==expected)
 for _,row in d.iterrows():
  base='kamp_v2_mal/runs' if variant=='dfine_M' else 'kamp_v2_uq/runs_frozen'
  preds=json.loads((R/'experiments'/base/row.run/'common_eval/predictions_original.json').read_text())
  alarms={p['image_id'] for p in nms(preds,.7) if p['score']>=row.threshold}
  check(row.run+' all validation images alarm',len(alarms)==66)
a=json.loads((F/'reference_results/frozen_test_summary.json').read_text())['aggregate'];a={x['variant']:x for x in a}
check('saved test UQ AP',round(a['dfine_mal_UQ']['AP'],2)==37.46)
check('saved test FP increases',a['dfine_mal_UQ']['FP']>a['dfine_M']['FP'])
g=pd.read_csv(F/'analysis/gt_failures.csv');check('representative FN IDs',set(g.loc[~g.TP,'gt_id'])=={110,136});check('representative FN localization',set(g.loc[~g.TP,'stage'])=={'localization'})
c=pd.read_csv(F/'analysis/condition_extended.csv')
check('each condition denominator 144',all(c.groupby('variable').GT.sum()==144));check('each condition FN 2',all(c.groupby('variable').FN.sum()==2))
p=pd.read_csv(R/'experiments/kamp_v2_ambiguity_uq/analysis/paired_UQ.csv');check('Soft1 vs edge AP75 improves only one seed',sum(p.loc[p.control=='edge','AP75']>0)==1)
report={'checks':checks,'test_inference_executed':False,'saved_test_results_read':True,'training_executed':False,'report_sha256':hashlib.sha256(s.encode()).hexdigest()}
(F/'analysis/report_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('Passed',len(checks),'checks')
