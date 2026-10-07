from pathlib import Path
import json,hashlib
import torch,pandas as pd
E=Path(__file__).parent;A=E.parent/'kamp_v2_ambiguity'; jobs=json.loads((E/'manifest.json').read_text())['jobs'];checks=[];fail=[];cond=[];geometry=[]
for j in jobs:
 d=E/'runs_frozen'/j['name'];b=A/'runs'/f"{j['variant']}_seed{j['seed']}";m=json.loads((d/'common_eval/metrics.json').read_text());meta=json.loads((d/'metadata.json').read_text())
 cp=Path(j['base_checkpoint']);c=torch.load(cp,map_location='cpu',weights_only=False);bs=c['ema']['module'] if 'ema' in c else c['model'];u=torch.load(d/'last.pth',map_location='cpu',weights_only=False)['model']
 identical=all(torch.equal(v,u[k]) for k,v in bs.items());assert identical
 assert hashlib.sha256(cp.read_bytes()).hexdigest()==j['base_sha256'];assert hashlib.sha256((d/'last.pth').read_bytes()).hexdigest()==m['checkpoint_sha256']
 assert meta['epochs']==30 and meta['train_images']==350 and not meta['test_used']
 checks.append(dict(name=j['name'],base_identity=identical,hashes_valid=True,epochs=30));del c,bs,u
 g=pd.read_csv(d/'common_eval/gt_diagnostics.csv',float_precision='round_trip');old=pd.read_csv(b/'common_eval/gt_diagnostics.csv',float_precision='round_trip')
 assert (g.best_iou==old.best_iou).all()
 for _,r in g[g.gt_id.isin([110,136])].iterrows(): fail.append(dict(variant=j['variant'],seed=j['seed'],**r.to_dict()))
 for machine,v in m['conditions'].items():cond.append(dict(variant=j['variant'],seed=j['seed'],machine=machine,**v))
 geometry.append(dict(variant=j['variant'],seed=j['seed'],leader_edge_MAE=g[[f'leader_{x}' for x in ['left','top','right','bottom']]].abs().to_numpy().mean(),precise_candidate_but_not_leader=int(((g.best_iou>=.75)&(g.leader_iou<.75)).sum())))
pd.DataFrame(fail).to_csv(E/'analysis/repeated_misses.csv',index=False);pd.DataFrame(cond).to_csv(E/'analysis/conditions.csv',index=False);pd.DataFrame(geometry).to_csv(E/'analysis/geometry.csv',index=False)
(E/'analysis/verification.json').write_text(json.dumps(dict(runs=checks,raw_best_iou_unchanged_all_GT=True,test_used=False),indent=2))
print(pd.DataFrame(fail)[['variant','seed','gt_id','TP','best_iou','leader_iou']].to_string(index=False));print(pd.DataFrame(cond).groupby(['variant','machine']).TP.mean());print(pd.DataFrame(geometry).groupby('variant').mean(numeric_only=True).to_string())
