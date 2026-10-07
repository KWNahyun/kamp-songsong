import sys,json
from pathlib import Path
import torch
R=Path('/home/viplab/contest');E=Path(__file__).parent
sys.path[:0]=[str(R/'models/D-FINE'),str(R/'experiments/kamp_ablation_v3')]
from src.core import YAMLConfig
from loss_patch import install
torch.set_num_threads(4);torch.manual_seed(20260929);install(0,True)
cfg=YAMLConfig(str(E/'configs/dfine_M_seed20260929.yml'));m=cfg.model.cuda().train()
m.load_state_dict(torch.load(R/'experiments/kamp_pilot_v1/dfine_s_init.pth',map_location='cpu',weights_only=False)['model'])
opt=torch.optim.AdamW(m.parameters(),lr=.0002);steps=[]
for idx,(im,targets) in enumerate(cfg.train_dataloader):
 if idx==3:break
 im=im.cuda();targets=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
 opt.zero_grad();ls=cfg.criterion(m(im,targets),targets);assert 'loss_mal' in ls and 'loss_vfl' not in ls
 loss=sum(ls.values());loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 torch.nn.utils.clip_grad_norm_(m.parameters(),.1);opt.step();steps.append({'step':idx,'loss':float(loss.detach()),'mal':float(ls['loss_mal'].detach())})
result={'passed':True,'train_smoke_steps':steps,'test_used':False,'peak_vram_MiB':torch.cuda.max_memory_allocated()/2**20,'smoke_weights_discarded':True}
(E/'preflight.json').write_text(json.dumps(result,indent=2));print(result)
