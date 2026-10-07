import sys,ast,types,json,copy
from pathlib import Path
import torch
import torch.nn.functional as F
R=Path('/home/viplab/contest');E=Path(__file__).parent
sys.path.insert(0,str(R/'models/D-FINE'))
from src.core import YAMLConfig
from src.zoo.dfine.box_ops import box_iou,box_cxcywh_to_xyxy
from loss_patch import size_loss,mal,install

torch.set_num_threads(4);torch.manual_seed(20260929)
# Compare independent official MAL implementation including gradients.
tree=ast.parse((E/'sources/deim_criterion.py').read_text())
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef))
fn=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='loss_labels_mal')
ns=dict(torch=torch,F=F,box_iou=box_iou,box_cxcywh_to_xyxy=box_cxcywh_to_xyxy)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'official_mal','exec'),ns)
cfg=YAMLConfig(str(E/'configs/dfine_GM03_seed20260929.yml'))
c=cfg.criterion;c.mal_alpha=None
for empty in (False,True):
 logits=torch.randn(2,5,1,requires_grad=True);boxes=torch.rand(2,5,4)*.4+.1
 targets=[{'boxes':torch.tensor([[.3,.3,.1,.1]]),'labels':torch.zeros(1,dtype=torch.long)} for _ in range(2)]
 inds=[(torch.tensor([0]),torch.tensor([0])) for _ in range(2)]
 if empty:
    targets=[{'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long)} for _ in range(2)]
    inds=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)) for _ in range(2)]
 out={'pred_logits':logits,'pred_boxes':boxes}
 a=mal(c,out,targets,inds,2)['loss_mal'];b=ns['loss_labels_mal'](c,out,targets,inds,2)['loss_mal']
 torch.testing.assert_close(a,b)
 torch.testing.assert_close(torch.autograd.grad(a,logits,retain_graph=True)[0],torch.autograd.grad(b,logits)[0])
gt=torch.tensor([[.5,.5,.1,.1]])
x=torch.tensor([[.5,.5,.12,.08]],requires_grad=True)
l=size_loss(x,gt);l.backward();assert x.grad[0,2]>0 and x.grad[0,3]<0 and x.grad[0,:2].abs().sum()==0
assert size_loss(gt,gt)==0
assert size_loss(torch.empty(0,4),torch.empty(0,4))==0
# Actual training batch at full resolution and batch size.
model=cfg.model.cuda().train();model.load_state_dict(torch.load(R/'experiments/kamp_pilot_v1/dfine_s_init.pth',map_location='cpu',weights_only=False)['model'],strict=True)
images,targets=next(iter(cfg.train_dataloader))
images=images.cuda();targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
assert images.shape==(8,3,640,640)
out=model(images,targets)
base=c(out,targets)
install(.3,True)
losses=c(out,targets)
for k,v in base.items():
 if 'vfl' not in k:torch.testing.assert_close(v,losses[k])
assert 'loss_size' in losses and sum(k.startswith('loss_size') for k in losses)==1
assert any(k.startswith('loss_mal') for k in losses) and not any('vfl' in k for k in losses)
sum(losses.values()).backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
result={'passed':True,'official_MAL_value_and_gradient':True,'empty_targets':True,'size_gradient_direction':True,'unchanged_original_regression_losses':True,'full_batch_forward_backward':True,'batch_shape':list(images.shape),'peak_vram_MiB':torch.cuda.max_memory_allocated()/2**20,'loss_keys':list(losses)}
(E/'preflight.json').write_text(json.dumps(result,indent=2));print(result)
