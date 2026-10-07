"""Eight-step diagnostic replay through the real trainer; discard diagnostic weights."""
from pathlib import Path
import sys,json,runpy,hashlib,torch,yaml,os
R=Path('/home/viplab/contest');E=Path(__file__).parent
sys.path[:0]=[str(R/'models/D-FINE')]
from src.solver._solver import BaseSolver
mode,name=sys.argv[1:3];out=E/name;out.mkdir(exist_ok=False)
source=R/('experiments/kamp_v2_mal/configs/dfine_M_seed20260929.yml' if mode=='old' else 'experiments/kamp_v2_ambiguity/configs/mal_seed20260929.yml');cfg=yaml.safe_load(source.read_text());cfg['output_dir']=str(out/'trainer');(out/'config.yml').write_text(yaml.safe_dump(cfg))
def digest(ts):
 h=hashlib.sha256()
 for t in ts:h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
records=[];modelref=[];setup=BaseSolver._setup
class StopReplay(Exception):pass

def patched_setup(self):
 setup(self);m=self.model;modelref.append(m)
 (out/'initial.json').write_text(json.dumps(dict(initial_state=digest(list(m.state_dict().values())),torch_rng=digest([torch.get_rng_state()]),cuda_rng=digest(torch.cuda.get_rng_state_all()),torch=str(torch.__version__),cuda=torch.version.cuda,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,num_threads=torch.get_num_threads()),indent=2))
 def pre(module,args,kw):
  module._replay_recording='targets' in kw
  if not module._replay_recording:return
  if len(records)>=8:raise StopReplay()
  targets=kw['targets'];records.append(dict(step=len(records),images_hash=digest([args[0]]),image_ids=[t['image_id'].tolist() for t in targets],boxes_hash=digest([t['boxes'] for t in targets]),rng_cpu=digest([torch.get_rng_state()]),rng_cuda=digest(torch.cuda.get_rng_state_all())))
 def post(module,args,outputs):
  if module._replay_recording:records[-1]['outputs_hash']=digest([outputs['pred_logits'],outputs['pred_boxes']])
 def loss(module,args,result):records[-1]['losses']={k:float(v.detach()) for k,v in result.items()}
 m.register_forward_pre_hook(pre,with_kwargs=True);m.register_forward_hook(post);self.criterion.register_forward_hook(loss)
BaseSolver._setup=patched_setup
step=torch.optim.AdamW.step

def optim_step(self,*args,**kwargs):
 m=modelref[0];grads=[p.grad for p in m.parameters() if p.grad is not None];records[-1]['gradient_hash']=digest(grads)
 if len(records)==1:torch.save([g.detach().cpu() for g in grads],out/'first_gradients.pt')
 result=step(self,*args,**kwargs);records[-1]['updated_weights_hash']=digest(list(m.parameters()));(out/'steps.json').write_text(json.dumps(records,indent=2));return result

torch.optim.AdamW.step=optim_step
if mode=='old':
 script=R/'experiments/kamp_ablation_v3/train_dfine.py';sys.path.insert(0,str(script.parent));extra=['--mal']
else:
 script=R/'experiments/kamp_v2_ambiguity/train.py';sys.path.insert(0,str(script.parent));extra=['--variant','mal','--audit',str(out/'audit.json')]
sys.argv=[str(script),'-c',str(out/'config.yml'),'-t',str(R/'experiments/kamp_pilot_v1/dfine_s_init.pth'),'--device','cuda:0','--seed','20260929']+extra
try:runpy.run_path(str(script),run_name='__main__')
except StopReplay:print('REPLAY_COMPLETE',len(records),flush=True)
assert len(records)==8
