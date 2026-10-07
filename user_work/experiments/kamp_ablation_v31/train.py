import sys,runpy,argparse
from pathlib import Path
R=Path('/home/viplab/contest');sys.path.insert(0,str(R/'models/D-FINE'))
p=argparse.ArgumentParser(add_help=False);p.add_argument('--mal',action='store_true');a,rest=p.parse_known_args()
from stable_loss import install
install(.1,a.mal)
sys.argv=[sys.argv[0]]+rest
runpy.run_path(str(R/'models/D-FINE/train.py'),run_name='__main__')
