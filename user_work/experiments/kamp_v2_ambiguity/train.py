import sys,runpy,argparse
from pathlib import Path
import torch
R=Path('/home/viplab/contest');E=Path(__file__).parent
sys.path[:0]=[str(E),str(R/'models/D-FINE'),str(R/'experiments/kamp_ablation_v3')]
p=argparse.ArgumentParser(add_help=False);p.add_argument('--variant',required=True);p.add_argument('--audit',required=True);a,rest=p.parse_known_args()
from ambiguity_patch import install
install(a.variant,a.audit)
# Preserve global clipping but fail on invalid gradients before optimizer updates.
original_clip=torch.nn.utils.clip_grad_norm_
def clip(*args,**kwargs):
 kwargs['error_if_nonfinite']=True
 return original_clip(*args,**kwargs)
torch.nn.utils.clip_grad_norm_=clip
sys.argv=[sys.argv[0]]+rest
runpy.run_path(str(R/'models/D-FINE/train.py'),run_name='__main__')
