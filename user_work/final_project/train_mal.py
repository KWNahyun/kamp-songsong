from pathlib import Path
import sys,runpy
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'vendor/dfine'),str(R/'src')]
from loss_patch import install
install(0.0,True)
runpy.run_path(str(R/'vendor/dfine/train.py'),run_name='__main__')
