from pathlib import Path
import subprocess,os,json
E=Path(__file__).parent
for mode,name in [('old','old_a'),('old','old_b'),('new','new_a')]:
 with (E/f'{name}.log').open('w') as f:
  r=subprocess.run(['/home/viplab/contest/.detector-venv/bin/python',str(E/'replay.py'),mode,name],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'},cwd='/home/viplab/contest/experiments/kamp_v2_mal')
 print(name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
