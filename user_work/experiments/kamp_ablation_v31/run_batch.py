import subprocess,json,os,datetime,fcntl
from pathlib import Path
E=Path(__file__).parent;PY='/home/viplab/contest/.detector-venv/bin/python'
lock=(E/'batch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (E/'status.json').exists(),'Refuse to overwrite prior queue'
for f in ['preflight_G.json','preflight_GM.json']:assert json.loads((E/f).read_text())['passed']
assert json.loads((E/'trace_smoke/dfine_s/verification.json').read_text())['max_final_output_difference']<=1e-6
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
state={'started':now(),'pid':os.getpid(),'test_used':False,'phase':'trace','jobs':json.loads((E/'jobs.json').read_text())}
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
def save():
 p=E/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(E/'status.json')
def call(cmd,log):
 with (E/'logs'/log).open('x') as f:
  proc=subprocess.Popen(cmd,cwd=E,env=env,stdout=f,stderr=subprocess.STDOUT);state['active_pid']=proc.pid;save();return proc.wait()
save()
state['trace_returncode']=call([PY,str(E/'trace_decoder.py')],'trace.log');save()
# Trace failures are kept and do not change the independently validated loss experiment.
state['phase']='training';blocked=set();save()
for j in state['jobs']:
 if j['variant'] in blocked:
  j['status']='blocked_after_same_variant_failure';save();continue
 j.update(status='running',started=now());env['KAMP_RUN_DIR']=str(E/'runs'/j['name']);save()
 rc=call(j['command'],j['name']+'.log');j.update(status='trained' if rc==0 else 'failed',returncode=rc,finished=now());save()
 if rc:
  blocked.add(j['variant']);continue
 j['evaluation_returncode']=call([PY,str(E/'evaluate_v31.py'),j['name']],j['name']+'_eval.log');save()
state['phase']='analysis';save()
state['analysis_returncode']=call([PY,str(E/'analyze_v31.py')],'analysis.log');state['phase']='finished';state['finished']=now();state.pop('active_pid',None);save()
