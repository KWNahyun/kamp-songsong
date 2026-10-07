"""v3.1: GT-normalized relative-size Huber, final regular decoder only."""
import torch
import torch.nn.functional as F
from pathlib import Path
import os,json

def size_loss(pred,target):
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all() or (target[:,2:]<=0).any():
        raise FloatingPointError('Invalid input to relative-size loss')
    error=(pred[:,2:]-target[:,2:])/(target[:,2:]+1e-6)
    result=F.huber_loss(error,torch.zeros_like(error),delta=1.0,reduction='sum')/max(len(pred),1)
    if not torch.isfinite(result):
        dest=Path(os.environ.get('KAMP_RUN_DIR','.'));dest.mkdir(parents=True,exist_ok=True)
        torch.save({'pred':pred.detach().cpu(),'target':target.detach().cpu()},dest/'nonfinite_size_inputs.pt')
        raise FloatingPointError('Nonfinite relative-size loss')
    return result

def install(weight=.1,mal=False):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kamp_ablation_v3'))
    import loss_patch
    loss_patch.size_loss=size_loss
    loss_patch.install(weight,mal)
