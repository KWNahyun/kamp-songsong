import sys,json,argparse
from pathlib import Path
import torch
R=Path('/home/viplab/contest');E=Path(__file__).parent;sys.path.insert(0,str(R/'models/D-FINE'))
from stable_loss import size_loss,install
from src.core import YAMLConfig
p=argparse.ArgumentParser();p.add_argument('--mal',action='store_true');args=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(20260929)
cases=[]
for value in [-.1,-.004882931709289551,0.,1e-12,.005,.01,1.,100.]:
 x=torch.tensor([[.5,.5,value,value]],requires_grad=True);y=torch.tensor([[.5,.5,.01,.01]])
 loss=size_loss(x,y);loss.backward()
 assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
 if value<.01:assert (x.grad[:,2:]<0).all()
 if value>.01:assert (x.grad[:,2:]>0).all()
 assert (x.grad[:,:2]==0).all();cases.append(value)
x=torch.empty(0,4,requires_grad=True);size_loss(x,x.detach()).backward();assert x.grad.shape==(0,4)
for wh in [1e-12,1e-6,.01]:
 x=torch.tensor([[.5,.5,-.01,0]],requires_grad=True);y=torch.tensor([[.5,.5,wh,wh]])
 loss=size_loss(x,y);loss.backward();assert torch.isfinite(x.grad).all()
cfg=YAMLConfig(str(R/'experiments/kamp_pilot_v1/dfine_s.yml'));model=cfg.model.cuda().train()
model.load_state_dict(torch.load(R/'experiments/kamp_pilot_v1/dfine_s_init.pth',map_location='cpu',weights_only=False)['model'])
criterion=cfg.criterion;install(.1,args.mal)
opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
steps=[]
for i,(images,targets) in enumerate(cfg.train_dataloader):
 if i==4:break
 images=images.cuda();targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
 opt.zero_grad();ls=criterion(model(images,targets),targets);total=sum(ls.values());total.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 torch.nn.utils.clip_grad_norm_(model.parameters(),.1);opt.step()
 steps.append({'step':i,'loss':float(total),'size_loss':float(ls['loss_size'])})
out={'passed':True,'loss':'relative_size_huber_v31','mal':args.mal,'edge_cases':cases,'empty_positive_passed':True,'finite_gradient_and_recovery_direction':True,'smoke_steps':steps,'smoke_is_not_comparative_training':True,'peak_vram_MiB':torch.cuda.max_memory_allocated()/2**20}
(E/('preflight_GM.json' if args.mal else 'preflight_G.json')).write_text(json.dumps(out,indent=2));print(out)
