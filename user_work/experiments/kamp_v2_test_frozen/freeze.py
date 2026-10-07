from pathlib import Path
import json,csv,hashlib,datetime
R=Path('/home/viplab/contest');E=Path(__file__).parent;B=E.parent/'kamp_v2_baselines';M=E.parent/'kamp_v2_mal';U=E.parent/'kamp_v2_uq';jobs=[]
for seed in [20260929,20260930,20261001]:
 for variant in ['yolov8s','yolov8s_p2','dfine_s','dfine_s_p2','dfine_M','dfine_mal_UQ']:
  name=f'{variant}_seed{seed}';parent=U if variant=='dfine_mal_UQ' else M if variant=='dfine_M' else B
  ck=parent/('runs_frozen' if variant=='dfine_mal_UQ' else 'runs')/name/('weights/best.pt' if variant.startswith('yolo') else 'last.pth' if variant=='dfine_mal_UQ' else 'best_stg1.pth')
  rows=list(csv.DictReader((parent/'analysis/metrics.csv').open(encoding='utf-8-sig')));row=next(r for r in rows if r['run']==name and r['setting']=='nms07')
  cfg=U/'configs/dfine_UQ_seed20260929.yml' if variant=='dfine_mal_UQ' else parent/'configs'/f'{name}.yml'
  jobs.append(dict(name=name,variant=variant,seed=seed,checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),config=str(cfg) if not variant.startswith('yolo') else None,val_threshold=float(row['threshold']),val_AP=float(row['AP']),val_AP75=float(row['AP75'])))
protocol=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),user_authorized_test=True,selected_model_before_test='dfine_mal_UQ',jobs=jobs,settings=dict(input=640,NMS=.7,box_scale=1,score_floor=.001,max_det=300,COCO_maxDets=100,operating_thresholds='locked per-run validation FP<=6 thresholds; no test threshold tuning'),limitations=['teammate used test in preprocessing selection','positive-only test cannot estimate normal false alarm rate'],prohibited_followup='Do not tune or select new settings on this test and call a rerun independent validation')
assert not (E/'frozen_protocol.json').exists();(E/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2));print('18 checkpoints and validation thresholds frozen before test access')
