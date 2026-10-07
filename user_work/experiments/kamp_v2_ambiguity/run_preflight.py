import subprocess,json,os
from pathlib import Path
E=Path(__file__).parent;P='/home/viplab/contest/.detector-venv/bin/python';manifest=json.loads((E/'manifest.json').read_text())
for v in manifest['variants']:
 if v=='soft_fgl10' and (E/'preflight'/f'{v}.json').exists():continue
 with (E/'preflight'/f'{v}.log').open('w') as f:
  r=subprocess.run([P,str(E/'preflight.py'),v],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
 print(v,r.returncode,flush=True)
 if r.returncode:raise SystemExit('Preflight failed: '+v)
results=[json.loads((E/'preflight'/f'{v}.json').read_text()) for v in manifest['variants']]
(E/'preflight.json').write_text(json.dumps(dict(passed=all(x['passed'] for x in results),variants=results),indent=2));print('ALL PASS',flush=True)
