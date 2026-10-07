import os,sys,json,time,subprocess,concurrent.futures,traceback
from pathlib import Path
E=Path(__file__).parent; P='/home/viplab/contest/.detector-venv/bin/python'
(E/'logs').mkdir(exist_ok=True); jobs=json.loads((E/'manifest.json').read_text())['jobs']; start=time.time()
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
def run(j):
 name=j['name']; result=dict(name=name,started=time.time())
 with (E/'logs'/f'{name}.log').open('w') as f:
  for stage,cmd in [('train',[P,str(E/'train.py'),'--mode','unary','--base',j['variant'],'--seed',str(j['seed'])]),('eval',[P,str(E/'evaluate.py'),name])]:
   result[stage+'_exit']=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd='/home/viplab/contest/experiments/kamp_v2_uq')
   if result[stage+'_exit']:break
 result['ended']=time.time();(E/'logs'/f'{name}.status.json').write_text(json.dumps(result,indent=2));return result
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  results.append(future.result());(E/'queue_status.json').write_text(json.dumps(dict(completed=results,total=9,elapsed_seconds=time.time()-start),indent=2))
(E/'completion.json').write_text(json.dumps(dict(results=results,elapsed_seconds=time.time()-start,failed=sum(r.get('train_exit')!=0 or r.get('eval_exit')!=0 for r in results)),indent=2))
if all(r.get('eval_exit')==0 for r in results):
 subprocess.run([P,str(E/'aggregate.py')],check=True,env=env)
