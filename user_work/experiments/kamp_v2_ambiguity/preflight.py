import sys,json,torch,random,numpy as np
from pathlib import Path
R=Path('/home/viplab/contest');E=Path(__file__).parent
sys.path[:0]=[str(E),str(R/'models/D-FINE'),str(R/'experiments/kamp_ablation_v3')]
from ambiguity_patch import install,dense_encode,original_encode,tolerant_penalty
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.core import YAMLConfig
variant=sys.argv[1];torch.set_num_threads(4);torch.manual_seed(20260929);random.seed(20260929);np.random.seed(20260929)
# Zero-uncertainty distribution equivalence, including gradients and outside-range labels.
ref=torch.tensor([[.5,.5,.03,.04],[.6,.3,.05,.03]])
boxes=torch.tensor([[.483,.481,.519,.525],[.1,.1,.9,.9]])
s=torch.tensor([4.]);up=torch.tensor([.5]);left,wr,wl=original_encode(ref,boxes,32,s,up);q=dense_encode(ref,boxes,32,s,up)
x=torch.randn(8,33,requires_grad=True);old=(torch.nn.functional.cross_entropy(x,left.long(),reduction='none')*wl+torch.nn.functional.cross_entropy(x,left.long()+1,reduction='none')*wr).sum();new=-(q*x.log_softmax(-1)).sum();g0=torch.autograd.grad(old,x,retain_graph=True)[0];g1=torch.autograd.grad(new,x)[0]
assert torch.allclose(old,new,atol=2e-6) and torch.allclose(g0,g1,atol=2e-7)
e=torch.tensor([0.,.2,1.,2.],requires_grad=True);z=tolerant_penalty(e,torch.zeros_like(e),.25);assert torch.equal(e,z);assert torch.allclose(torch.autograd.grad(z.sum(),e)[0],torch.ones_like(e))
audit=install(variant,E/'preflight'/f'{variant}_audit.json')
cfg=YAMLConfig(str(E/'configs'/f'{variant}_seed20260929.yml'));m=cfg.model.cuda().train();m.load_state_dict(torch.load(R/'experiments/kamp_pilot_v1/dfine_s_init.pth',map_location='cpu',weights_only=False)['model']);criterion=cfg.criterion
opt=torch.optim.AdamW(m.parameters(),lr=.0002);records=[]
for i,(images,targets) in enumerate(cfg.train_dataloader):
 if i==2:break
 images=images.cuda();targets=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
 orig=[t['boxes'].clone() for t in targets];opt.zero_grad();out=m(images,targets);indices0=criterion.matcher({k:v for k,v in out.items() if 'aux' not in k},targets)['indices'];losses=criterion(out,targets);indices1=criterion.matcher({k:v for k,v in out.items() if 'aux' not in k},targets)['indices']
 assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(indices0,indices1));assert all(torch.equal(t['boxes'],b) for t,b in zip(targets,orig));assert 'loss_mal' in losses and 'loss_vfl' not in losses
 assert 'loss_fgl' in losses and any('loss_fgl_dn' in k for k in losses) and any('loss_fgl_aux' in k for k in losses)
 loss=sum(losses.values());loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),.1,error_if_nonfinite=True);opt.step();records.append(dict(step=i,loss=float(loss.detach()),grad_norm=float(norm),mal=float(losses['loss_mal'].detach()),bbox=float(losses['loss_bbox'].detach()),fgl=float(losses['loss_fgl'].detach())))
result=dict(passed=True,variant=variant,steps=records,zero_target_loss_diff=float((old-new).detach().abs()),zero_target_gradient_max_diff=float((g0-g1).abs().max()),peak_vram_MiB=torch.cuda.max_memory_allocated()/2**20,GT_unchanged=True,matching_unchanged_on_fixed_outputs=True,regular_aux_DN_verified=True,test_used=False,smoke_weights_discarded=True)
(E/'preflight'/f'{variant}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
