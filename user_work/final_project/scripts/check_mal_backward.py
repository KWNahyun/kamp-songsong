from pathlib import Path
import sys,json,argparse,torch
r=Path(__file__).resolve().parents[1];sys.path[:0]=[str(r/'vendor/dfine'),str(r/'src')]
from loss_patch import install
install(0,True)
from src.core import YAMLConfig
from src.solver import TASKS
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
torch.set_num_threads(4)
c=YAMLConfig(a.config,tuning=str(r/'checkpoints/dfine_s_coco_init.pth'),device='cuda:0')
s=TASKS[c.yaml_cfg['task']](c);s.train()
x,t=next(iter(s.train_dataloader));x=x.cuda();t=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in z.items()} for z in t]
s.model.train();out=s.model(x,targets=t);losses=s.criterion(out,t);loss=sum(losses.values());loss.backward()
assert torch.isfinite(loss) and any('mal' in k for k in losses)
g=[p.grad for p in s.model.parameters() if p.grad is not None];assert g and all(torch.isfinite(v).all() for v in g)
report={'mal_backward_passed':True,'batch':len(t),'loss':float(loss.detach()),'finite_gradient_tensors':len(g),'optimizer_step_executed':False,'test_used':False}
(r/'analysis/mal_backward_check.json').write_text(json.dumps(report,indent=2));print(report)
