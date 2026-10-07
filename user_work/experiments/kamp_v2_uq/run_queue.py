from pathlib import Path
import json,datetime,subprocess,os,fcntl,time,math
E=Path(__file__).parent;PY='/home/viplab/contest/.detector-venv/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
lock=(E/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for base in ['base','mal']:
 m=json.loads((E/'runs_frozen'/f'dfine_{base}_UQ_seed20260929_smoke/metadata.json').read_text());assert m['base_bitwise_identical'] and m['train_images']==350
path=E/'queue_status.json';assert not path.exists()
jobs=[{'name':f'dfine_{base}_UQ_seed{seed}','base':base,'seed':seed,'status':'pending'} for seed in [20260929,20260930,20261001] for base in ['base','mal']]
state={'status':'running','started':now(),'test_used':False,'jobs':jobs,'pid':os.getpid()}
def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
save()
for j in jobs:
 while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())<10000:
  j['status']='waiting_for_vram';save();time.sleep(30)
 j.update(status='running',started=now());save();print('START',j['name'],now(),flush=True)
 current=E/'current.log'
 if current.is_symlink():current.unlink()
 current.symlink_to(E/'logs'/f"{j['name']}.log")
 with (E/'logs'/f"{j['name']}.log").open('x') as log:
  p=subprocess.Popen([PY,str(E/'train.py'),'--mode','unary','--base',j['base'],'--seed',str(j['seed'])],cwd=E,env=env,stdout=log,stderr=subprocess.STDOUT);j['pid']=p.pid;save();code=p.wait()
 try:
  assert code==0,f'Exit {code}'
  out=E/'runs_frozen'/j['name'];meta=json.loads((out/'metadata.json').read_text());rr=[json.loads(x) for x in (out/'log.jsonl').read_text().splitlines()]
  assert meta['base_bitwise_identical'] and len(rr)==30 and all(math.isfinite(r['train_selector_quality']) for r in rr)
  j['status']='trained'
 except Exception as ex:j.update(status='failed',error=str(ex))
 j.update(finished=now(),returncode=code);save();print('END',j['name'],j['status'],now(),flush=True)
 if j['status']=='failed':state['status']='stopped_on_failure';save();raise SystemExit(1)
state.update(status='finished',finished=now());save()
