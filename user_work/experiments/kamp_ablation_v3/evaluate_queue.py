import subprocess,json,sys
from pathlib import Path
E=Path(__file__).parent
for j in json.loads((E/'queue_status.json').read_text())['jobs']:
 if j['status']!='trained':continue
 if (E/'runs'/j['name']/'common_eval/metrics.json').exists():continue
 with (E/'logs'/f"{j['name']}_evaluation.log").open('w') as f:
  r=subprocess.run([sys.executable,str(E/'evaluate_v3.py'),j['name']],stdout=f,stderr=subprocess.STDOUT)
 print(j['name'],r.returncode,flush=True)
 if r.returncode:raise RuntimeError(j['name'])
