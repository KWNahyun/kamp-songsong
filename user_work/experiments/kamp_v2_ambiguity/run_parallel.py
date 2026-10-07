"""Single-writer, up-to-three GPU process scheduler, including evaluation slots.
Adopts the first training process without interrupting its optimizer state.
"""
from pathlib import Path
import json,time,os,sys,subprocess,fcntl,hashlib
E=Path(__file__).parent;PY=sys.executable
lock=(E/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads((E/'queue_status.json').read_text());migration=json.loads((E/'parallel_migration.json').read_text())
assert state['status']=='running' and state['active']==migration['adopt_name']
state.update(pid=os.getpid(),execution='parallel',max_parallel=3,launch_min_free_MiB=7500,launch_spacing_seconds=20)
jobs={j['name']:j for j in state['jobs']};slots={migration['adopt_name']:{'pid':migration['adopt_pid'],'process':None,'phase':'train','file':None}}
state['parallel_started']=time.time();last_launch=time.time()-20;training_failures=0

def save():
 state['updated']=time.time();state['active_jobs']=[dict(name=n,pid=s['pid'],phase=s['phase']) for n,s in slots.items()];state['active']=next(iter(slots),None);state['active_pid']=slots[state['active']]['pid'] if state['active'] else None
 tmp=E/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(E/'queue_status.json')
def launch(job,phase):
 global last_launch
 name=job['name'];cmd=job['command'] if phase=='train' else [PY,str(E/'evaluate.py'),name]
 log=E/'logs'/(name+('.log' if phase=='train' else '_eval.log'));f=log.open('w');p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
 slots[name]={'pid':p.pid,'process':p,'phase':phase,'file':f};job['status']='running' if phase=='train' else 'evaluating'
 if phase=='train':job['started']=time.time()
 job[phase+'_pid']=p.pid;last_launch=time.time();print('START',name,phase,p.pid,flush=True);save()
def adopted_status(slot,job):
 p=Path('/proc')/str(slot['pid'])/'stat'
 if p.exists():
  try:
   fields=p.read_text().rsplit(')',1)[1].split()
   if fields[0]!='Z':return None
  except FileNotFoundError:pass
 # Orphan process exit code is unavailable; require all epochs AND trainer completion
 # AND the checkpoint. Never infer success merely from process disappearance.
 log=E/'logs'/f"{job['name']}.log";history=E/'runs'/job['name']/'log.txt'
 rows=[json.loads(x) for x in history.read_text().splitlines()] if history.exists() else []
 good=len(rows)==30 and rows[-1]['epoch']==29 and 'Training time ' in log.read_text() and (E/'runs'/job['name']/'best_stg1.pth').exists()
 job['adopted_completion_verified']=good
 return 0 if good else -999

def free_gpu():
 try:return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
 except Exception:return 0
save()
while True:
 changed=False
 for name,slot in list(slots.items()):
  job=jobs[name];rc=slot['process'].poll() if slot['process'] else adopted_status(slot,job)
  if rc is None:continue
  if slot['file']:slot['file'].close()
  del slots[name];changed=True
  if slot['phase']=='train':
   job['train_exit_code']=rc;job['train_seconds']=time.time()-job['started']
   if rc==0 and (E/'runs'/name/'best_stg1.pth').exists():
    training_failures=0;launch(job,'eval')
   else:job.update(status='training_failed',ended=time.time());training_failures+=1
  else:
   job.update(eval_exit_code=rc,status='completed' if rc==0 else 'evaluation_failed',ended=time.time())
   subprocess.run([PY,str(E/'aggregate.py')],check=False)
  print('END',name,slot['phase'],rc,flush=True);save()
 pending=[j for j in state['jobs'] if j['status']=='pending']
 if training_failures>=3:state['stop_new_launches']='three consecutive observed training failures'
 if pending and len(slots)<3 and time.time()-last_launch>=20 and not state.get('stop_new_launches'):
  free=free_gpu();state['gpu_free_MiB']=free
  if free>=7500:launch(pending[0],'train')
 if not slots and (not pending or state.get('stop_new_launches')):break
 save();time.sleep(2)
state.update(status='completed' if all(j['status']=='completed' for j in state['jobs']) else ('stopped_repeated_failures' if state.get('stop_new_launches') else 'completed_with_failures'),ended=time.time());save()
(E/'completion.json').write_text(json.dumps(dict(status=state['status'],ended=state['ended'],elapsed_seconds=state['ended']-state['started'],completed=sum(j['status']=='completed' for j in state['jobs']),jobs=len(state['jobs']),max_parallel=3),indent=2));print(state['status'],flush=True)
