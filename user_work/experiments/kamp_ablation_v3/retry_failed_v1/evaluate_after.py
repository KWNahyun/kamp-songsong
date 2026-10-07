import sys,json,subprocess,time
from pathlib import Path
E=Path(__file__).resolve().parents[1];D=Path(__file__).parent
while True:
 if not (D/'status.json').exists():
  time.sleep(1);continue
 s=json.loads((D/'status.json').read_text())
 if 'finished' in s:break
 time.sleep(15)
for j in s['jobs']:
 if j['status']!='trained':continue
 with (D/f"{j['name']}_evaluation.log").open('w') as f:
  p=subprocess.run([sys.executable,str(E/'evaluate_retry.py'),j['name']],stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(j['name'])
with (D/'combined_analysis.log').open('w') as f:
 subprocess.run([sys.executable,str(E/'analyze_with_retries.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
(D/'analysis_complete.json').write_text(json.dumps({'complete':True,'test_used':False,'retry_results':[{k:j[k] for k in ['name','status','returncode']} for j in s['jobs']]},indent=2))
