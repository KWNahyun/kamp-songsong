from pathlib import Path
import json,subprocess,time
E=Path(__file__).parent;py='/home/viplab/contest/.detector-venv/bin/python';start=time.time();jobs=json.loads((E/'frozen_protocol.json').read_text())['jobs']
subprocess.run([py,str(E/'prepare_test.py')],check=True)
for i,j in enumerate(jobs):
 log=E/(j['name']+'.log')
 with log.open('w') as f:p=subprocess.run([py,str(E/'evaluate.py'),j['name']],stdout=f,stderr=subprocess.STDOUT)
 (E/'status.json').write_text(json.dumps(dict(completed=i+(p.returncode==0),total=len(jobs),last=j['name'],returncode=p.returncode,seconds=time.time()-start),indent=2))
 if p.returncode:raise RuntimeError('Evaluation failed: '+str(log))
 print('DONE',j['name'],round(time.time()-start,1),flush=True)
(E/'completion.json').write_text(json.dumps(dict(completed=True,runs=len(jobs),test_used=True,seconds=time.time()-start),indent=2))
