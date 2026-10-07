from pathlib import Path
import json,hashlib,sys,yaml
R=Path('/home/viplab/contest');E=Path(__file__).parent
sys.path[:0]=[str(R/'models/D-FINE')]
from ambiguity_patch import VARIANTS,OFFSETS,PROBS
for d in ['configs','runs','logs','preflight','analysis']:(E/d).mkdir(exist_ok=True)
base=yaml.safe_load((R/'experiments/kamp_v2_mal/configs/dfine_M_seed20260929.yml').read_text())
jobs=[]
for seed in [20260929,20260930,20261001]:
 for variant,spec in VARIANTS.items():
  name=f'{variant}_seed{seed}';cfg=json.loads(json.dumps(base));cfg['output_dir']=str(E/'runs'/name)
  cp=E/'configs'/f'{name}.yml';cp.write_text(yaml.safe_dump(cfg))
  jobs.append(dict(name=name,variant=variant,seed=seed,config=str(cp),status='pending',command=[str(R/'.detector-venv/bin/python'),str(E/'train.py'),'--variant',variant,'--audit',str(E/'runs'/name/'ambiguity_audit.json'),'-c',str(cp),'-t',str(R/'experiments/kamp_pilot_v1/dfine_s_init.pth'),'--device','cuda:0','--seed',str(seed)]))
provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E/'ambiguity_patch.py',E/'train.py',R/'experiments/kamp_ablation_v3/loss_patch.py',R/'experiments/kamp_pilot_v1/dfine_s_init.pth',R/'experiments/kamp_v2_baselines/data640/annotations/train.json',R/'experiments/kamp_v2_baselines/data640/annotations/val.json',R/'experiments/kamp_v2_baselines/data_audit.json']}
manifest=dict(jobs=jobs,variants=VARIANTS,prior={'offsets':OFFSETS,'probabilities':PROBS,'type':'discrete symmetric triangular five-point','units':'original pixels'},provenance=provenance,epochs=30,train_images=350,train_boxes=797,val_images=66,val_boxes=144,batch=8,amp=False,input=640,NMS=.7,box_scale=1,UQ=False,test_used=False,initialization='same COCO-derived checkpoint, independent fresh runs',checkpoint_selection='best official validation COCO AP in 640 coordinates, same as existing baseline; common original-coordinate NMS evaluation afterwards')
assert not (E/'queue_status.json').exists(),'Queue already prepared; do not reset it'
(E/'manifest.json').write_text(json.dumps(manifest,indent=2));(E/'queue_status.json').write_text(json.dumps(dict(status='prepared',jobs=jobs),indent=2));print(len(jobs),'jobs prepared')
