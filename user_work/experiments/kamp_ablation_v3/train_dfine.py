import sys,runpy,argparse
from pathlib import Path
ROOT=Path('/home/viplab/contest')
sys.path.insert(0,str(ROOT/'models/D-FINE'))
sys.path.insert(0,str(ROOT/'experiments/kamp_pilot_v1'))
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--size-weight',type=float,default=0)
p.add_argument('--mal',action='store_true')
p.add_argument('--p2',action='store_true')
a,rest=p.parse_known_args()
if a.p2:
    import dfine_p2
    dfine_p2.install()
# Only install for new losses, keep baseline behavior identical to pilot.
if a.size_weight or a.mal:
    from loss_patch import install
    install(a.size_weight,a.mal)
sys.argv=[sys.argv[0]]+rest
runpy.run_path(str(ROOT/'models/D-FINE/train.py'),run_name='__main__')
