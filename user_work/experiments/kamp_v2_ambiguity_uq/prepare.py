from pathlib import Path
import json,hashlib
R=Path('/home/viplab/contest/experiments'); E=Path(__file__).parent; old=R/'kamp_v2_uq'; A=R/'kamp_v2_ambiguity'
s=(old/'train.py').read_text().replace('str(HERE)]','str(HERE), str(ROOT / "experiments/kamp_v2_uq")]')
s=s.replace('HERE / "configs" / config_name','ROOT / "experiments/kamp_v2_uq/configs" / config_name')
s=s.replace("control_path = ROOT / f'experiments/{parent}/runs/{base_name}_seed{seed}/best_stg1.pth'","control_path = ROOT / f'experiments/kamp_v2_ambiguity/runs/{base}_seed{seed}/best_stg1.pth'")
s=s.replace('choices=["base", "mal"]','choices=["mal", "edge", "soft10"]')
(E/'train.py').write_text(s)
jobs=[]
for v in ['mal','edge','soft10']:
 for seed in [20260929,20260930,20261001]:
  name=f'dfine_{v}_UQ_seed{seed}'; cp=A/f'runs/{v}_seed{seed}/best_stg1.pth'
  jobs.append(dict(name=name,variant=v,seed=seed,config=str(old/'configs/dfine_UQ_seed20260929.yml'),base_checkpoint=str(cp),base_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
(E/'manifest.json').write_text(json.dumps(dict(jobs=jobs,train_images=350,val_images=66,epochs=30,max_concurrent=3,test_used=False,NMS=.7,box_scale=1,selection='fixed last epoch; no UQ validation checkpoint selection',adoption_rule='Soft1+UQ versus each of MAL+UQ and edge+UQ: positive mean AP and AP75, each positive in at least 2/3 paired seeds, mean validation-selected FP<=6 TP not lower; exploratory evidence only, otherwise stop model search',recipe_source=str(old/'train.py'),recipe_sha256=hashlib.sha256((old/'train.py').read_bytes()).hexdigest()),indent=2))
s=(A/'evaluate.py').read_text().replace("name=sys.argv[1];job=", "sys.path.insert(0,str(R/'experiments/kamp_v2_uq'))\nfrom selector_patch import install_model\ninstall_model('unary')\nname=sys.argv[1];job=")
s=s.replace("E/'runs'/name", "E/'runs_frozen'/name").replace("'best_stg1.pth'","'last.pth'").replace('UQ=False','UQ=True')
(E/'evaluate.py').write_text(s)
