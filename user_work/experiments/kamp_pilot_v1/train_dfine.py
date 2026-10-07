import os, sys, runpy
from pathlib import Path
ROOT=Path('/home/viplab/contest')
sys.path.insert(0,str(ROOT/'models/D-FINE'))
if '--p2' in sys.argv:
    sys.argv.remove('--p2')
    import dfine_p2
    dfine_p2.install()
if __name__=='__main__':
    runpy.run_path(str(ROOT/'models/D-FINE/train.py'),run_name='__main__')

