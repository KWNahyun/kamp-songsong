from pathlib import Path
import yaml,json,hashlib
R=Path('/home/viplab/contest'); E=R/'experiments/kamp_ablation_v3'; P=R/'experiments/kamp_pilot_v1'; PY=str(R/'.detector-venv/bin/python')
(E/'configs').mkdir(exist_ok=True); (E/'runs').mkdir(exist_ok=True); (E/'logs').mkdir(exist_ok=True)
jobs=[]
variants=[('G01',.1,False,False),('M',0,True,False),('GM01',.1,True,False),('C0',0,False,True),('G03',.3,False,False),('GM03',.3,True,False)]
for seed in (20260929,20260930,20261001):
 for name,weight,mal,control in variants:
    run=f'dfine_{name}_seed{seed}'; cfg={'__include__':[str(P/'dfine_s.yml')],'output_dir':str(E/'runs'/run)}
    if control: cfg['DFINECriterion']={'weight_dict':{'loss_vfl':1,'loss_bbox':7.5,'loss_giou':2,'loss_fgl':.15,'loss_ddf':1.5}}
    path=E/'configs'/f'{run}.yml';path.write_text(yaml.safe_dump(cfg))
    cmd=[PY,str(E/'train_dfine.py'),'-c',str(path),'-t',str(P/'dfine_s_init.pth'),'--device','cuda:0','--seed',str(seed),'--size-weight',str(weight)]
    if mal:cmd+=['--mal']
    jobs.append(dict(name=run,family='dfine',seed=seed,variant=name,size_weight=weight,mal=mal,command=cmd,status='pending'))
for seed in (20260930,20261001):
 for model in ('dfine_s','yolov8s','dfine_s_p2','yolov8s_p2'):
    run=f'{model}_seed{seed}';p2=model.endswith('p2');family='dfine' if model.startswith('dfine') else 'yolo'
    if family=='dfine':
        path=E/'configs'/f'{run}.yml';path.write_text(yaml.safe_dump({'__include__':[str(P/f'{model}.yml')],'output_dir':str(E/'runs'/run)}))
        cmd=[PY,str(E/'train_dfine.py'),'-c',str(path),'-t',str(P/f'{model}_init.pth'),'--device','cuda:0','--seed',str(seed)]
    else:cmd=[PY,str(E/'train_yolo.py'),'--seed',str(seed),'--name',run]
    if p2:cmd+=['--p2']
    jobs.append(dict(name=run,family=family,seed=seed,variant=model,command=cmd,status='pending'))
manifest={'jobs':jobs,'count':len(jobs),'epochs':30,'batch':8,'input':640,'precision':'FP32','test_used':False,'analysis_requested':False,'reused_seed20260929_runs':[str(P/x) for x in ('dfine_s','dfine_s_p2','yolov8s','yolov8s_p2')],'initialization':'Fixed pilot init checkpoint across seeds; seeds vary training RNG/data ordering, not initial weights','MAL':'official default gamma2 mal_alphaNone; all original VFL branches replaced; matching unchanged','size':'final regular decoder O2O only; Huber delta1 eps1e-6; .1/.3 fixed exploration, no validation tuning','C0':'bbox L1 weight 5 to 7.5, other weights unchanged','sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(E.glob('*.py'))+list((E/'configs').glob('*.yml'))+[P/'dfine_s_init.pth',P/'dfine_s_p2_init.pth',P/'yolov8s_init.pt',P/'yolov8s_p2_init.pt',R/'data/processed/kamp500_telea_v1_640/annotations/train.json',R/'data/processed/kamp500_telea_v1_640/annotations/val.json']}}
assert len(jobs)==26
(E/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Prepared',len(jobs),'jobs')
