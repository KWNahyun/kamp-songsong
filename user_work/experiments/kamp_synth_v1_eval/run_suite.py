from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,time,os,threading,datetime
E=Path(__file__).parent;PY='/home/viplab/contest/.detector-venv/bin/python';P=json.loads((E/'protocol.json').read_text());start=time.time();state={};lock=threading.Lock()
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def update(name,stage,**kw):
 with lock:
  state[name]=dict(stage=stage,updated_KST=stamp(),**kw);tmp=E/'queue_status.tmp';tmp.write_text(json.dumps(dict(started_KST=began,elapsed_seconds=time.time()-start,jobs=state),indent=2));tmp.replace(E/'queue_status.json')
def work(j):
 name=j['name'];update(name,'queued');env=os.environ.copy();env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='1')
 for stage,script in [('inference','infer.py'),('analysis','analyze_run.py')]:
  flag=E/'runs'/name/('complete.json' if stage=='inference' else 'analysis/complete.json')
  if flag.exists():continue
  update(name,stage);args=[PY,str(E/script),name]
  if stage=='inference':args+=['--batch','1']
  with (E/f'{name}_{stage}.log').open('a') as log:
   result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env)
  if result.returncode:update(name,'failed',failed_stage=stage,returncode=result.returncode);return False
 update(name,'complete');print(stamp(),'DONE',name,flush=True);return True
began=stamp();(E/'queue.pid').write_text(str(os.getpid()));print(began,'START',len(P['jobs']),'jobs; 2 workers',flush=True)
with ThreadPoolExecutor(max_workers=2) as ex:ok=list(ex.map(work,P['jobs']))
if all(ok):
 with (E/'cross_analysis.log').open('a') as log:r=subprocess.run([PY,str(E/'cross_analysis.py')],stdout=log,stderr=subprocess.STDOUT)
 (E/'completion.json').write_text(json.dumps(dict(complete=r.returncode==0,runs=len(ok),seconds=time.time()-start,finished_KST=stamp(),cross_returncode=r.returncode),indent=2))
else:(E/'completion.json').write_text(json.dumps(dict(complete=False,successful=sum(ok),failed=len(ok)-sum(ok),finished_KST=stamp()),indent=2))
