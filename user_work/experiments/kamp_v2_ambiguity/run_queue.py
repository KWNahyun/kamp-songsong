"""Sequential GPU queue with durable per-job status and exclusive lock."""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,datetime,hashlib
E=Path(__file__).parent;PY=sys.executable
lock=(E/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((E/'preflight.json').read_text())['passed']
manifest=json.loads((E/'manifest.json').read_text())
for path,digest in manifest['provenance'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Changed source: '+path
state=json.loads((E/'queue_status.json').read_text());assert state['status']=='prepared','Do not blindly restart queue'
state.update(status='running',pid=os.getpid(),started=time.time());failures=0

def save():
 state['updated']=time.time();tmp=E/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(E/'queue_status.json')
def run(cmd,log):
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
  state['active_pid']=proc.pid;save();return proc.wait()
save()
for job in state['jobs']:
 job.update(status='running',started=time.time());state['active']=job['name'];save();print('START',job['name'],flush=True)
 rc=run(job['command'],E/'logs'/f"{job['name']}.log");job['train_exit_code']=rc;job['train_seconds']=time.time()-job['started']
 if rc==0 and (E/'runs'/job['name']/'best_stg1.pth').exists():
  job['status']='evaluating';save();rc=run([PY,str(E/'evaluate.py'),job['name']],E/'logs'/f"{job['name']}_eval.log");job['eval_exit_code']=rc
  job['status']='completed' if rc==0 else 'evaluation_failed'
 else:job['status']='training_failed'
 job['ended']=time.time();failures=failures+1 if job['status']=='training_failed' else 0;save();print('END',job['name'],job['status'],flush=True)
 subprocess.run([PY,str(E/'aggregate.py')],check=False)
 if failures>=3:
  state['status']='stopped_repeated_failures';save();break
else:state['status']='completed' if all(j['status']=='completed' for j in state['jobs']) else 'completed_with_failures'
state.update(ended=time.time(),active=None,active_pid=None);save();(E/'completion.json').write_text(json.dumps(dict(status=state['status'],ended=state['ended'],elapsed_seconds=state['ended']-state['started'],completed=sum(j['status']=='completed' for j in state['jobs']),jobs=len(state['jobs'])),indent=2))
print(state['status'],flush=True)
