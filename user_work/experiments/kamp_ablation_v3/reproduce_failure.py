"""Bounded reproduction of one interrupted trial; no replacement result."""
import sys,runpy,json
from pathlib import Path
import torch
E=Path(__file__).parent
sys.path.insert(0,'/home/viplab/contest/models/D-FINE')
import loss_patch
original=loss_patch.size_loss

def diagnostic(pred,target):
 value=original(pred,target)
 if not torch.isfinite(value):
  result={'cause':'nonfinite_size_loss','min_pred_wh':pred[:,2:].min().item(),'min_GT_wh':target[:,2:].min().item(),'invalid_pred_count':int((pred[:,2:]<=-1e-6).any(1).sum()),'pred_wh':pred[:,2:].detach().cpu().tolist(),'GT_wh':target[:,2:].detach().cpu().tolist()}
  (E/'failure_reproduction.json').write_text(json.dumps(result,indent=2))
  raise FloatingPointError('Captured invalid log-size input; see failure_reproduction.json')
 return value
loss_patch.size_loss=diagnostic
j=next(j for j in json.loads((E/'queue_status.json').read_text())['jobs'] if j['name']=='dfine_G01_seed20260930')
sys.argv=j['command'][1:]+['--output-dir',str(E/'failure_reproduction_run')]
runpy.run_path(str(E/'train_dfine.py'),run_name='__main__')
