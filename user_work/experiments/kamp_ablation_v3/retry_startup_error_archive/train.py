"""Exact original objective; diagnostic capture only, no clipping or loss change."""
import sys,runpy,json,os
from pathlib import Path
import torch
E=Path(__file__).resolve().parents[1];D=Path(__file__).parent
sys.path.insert(0,str(E));sys.path.insert(0,'/home/viplab/contest/models/D-FINE')
import loss_patch
original=loss_patch.size_loss
calls=0

def checked(pred,target):
 global calls
 calls+=1
 value=original(pred,target)
 if not torch.isfinite(value):
  p=Path(os.environ['RETRY_RUN_DIR']);p.mkdir(exist_ok=True,parents=True)
  torch.save({'pred':pred.detach().cpu(),'target':target.detach().cpu()},p/'nonfinite_size_inputs.pt')
  (p/'nonfinite_size_inputs.json').write_text(json.dumps({'call':calls,'min_pred_wh':pred[:,2:].min().item(),'min_target_wh':target[:,2:].min().item(),'nonpositive_pred_dimensions':int((pred[:,2:]+1e-6<=0).sum()),'loss':str(value.item())},indent=2))
 return value
loss_patch.size_loss=checked
runpy.run_path(str(E/'train_dfine.py'),run_name='__main__')
