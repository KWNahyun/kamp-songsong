import subprocess,json,os,datetime,sys,fcntl
from pathlib import Path
E=Path(__file__).resolve().parents[1];D=Path(__file__).parent
lock=(D/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
original=json.loads((E/'queue_status.json').read_text())
jobs=[dict(j) for j in original['jobs'] if j['status']=='failed']
state={'started':now(),'pid':os.getpid(),'objective_changed':False,'resume':False,'test_used':False,'jobs':[]}
def save():
 p=D/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(D/'status.json')
for j in jobs:
 run=D/j['name'];cmd=j['command'].copy();cmd[1]=str(D/'train.py');cmd+=['--output-dir',str(run)]
 item={'name':j['name'],'variant':j['variant'],'seed':j['seed'],'status':'running','started':now(),'command':cmd,'run':str(run)};state['jobs'].append(item);save()
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',RETRY_RUN_DIR=str(run))
 with (D/f"{j['name']}.log").open('x') as log:
  p=subprocess.Popen(cmd,cwd=E,env=env,stdout=log,stderr=subprocess.STDOUT);item['pid']=p.pid;save();rc=p.wait()
 item.update(status='trained' if rc==0 else 'failed',returncode=rc,finished=now());save()
state['finished']=now();save()
