"""Isolated MAL/annotation-ambiguity ablations. Never edits GT or matching."""
import atexit,json
from pathlib import Path
import torch
import torch.nn.functional as F
from src.zoo.dfine import dfine_criterion as cm
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy
from src.zoo.dfine.dfine_utils import bbox2distance as original_encode
from src.data.dataset.coco_dataset import CocoDetection
R=Path('/home/viplab/contest')
VARIANTS={
 'mal':{},
 'cx_l1half':{'l1_factor':.5},
 'edge':{'edge':True},
 'edge_half':{'edge':True,'l1_factor':.5},
 'soft05':{'edge':True,'epsilon':.5,'alpha':.25},
 'soft10':{'edge':True,'epsilon':1.,'alpha':.25},
 'soft20':{'edge':True,'epsilon':2.,'alpha':.25},
 'hard10':{'edge':True,'epsilon':1.,'alpha':0.},
 'fgl05':{'fgl':'mean','fgl_epsilon':.5},
 'fgl10':{'fgl':'mean','fgl_epsilon':1.},
 'jitter10':{'fgl':'jitter','fgl_epsilon':1.},
 'soft_fgl10':{'edge':True,'epsilon':1.,'alpha':.25,'fgl':'mean','fgl_epsilon':1.},
}
OFFSETS=(-1.,-.5,0.,.5,1.)
PROBS=(1/9,2/9,3/9,2/9,1/9)

def dense_encode(ref,box,reg_max,reg_scale,up):
 left,wr,wl=original_encode(ref,box,reg_max,reg_scale,up)
 left=left.long();q=box.new_zeros((left.numel(),reg_max+1))
 q.scatter_add_(1,left[:,None],wl.reshape(-1,1));q.scatter_add_(1,(left+1)[:,None],wr.reshape(-1,1))
 return q

def tolerant_penalty(error,epsilon,alpha):
 return alpha*error+(1-alpha)*(error-epsilon).clamp_min(0)

def install(variant,audit_path=None):
 spec=VARIANTS[variant]
 from loss_patch import install as install_mal
 install_mal(0,True)
 audit={'variant':variant,'spec':spec,'calls':0,'fgl_calls':0,'fgl_changed_rows':0,'fgl_rows':0,'clipped_coordinates':0,'jitter_coordinates':0,'pixel_prior':{'offsets':OFFSETS,'probabilities':PROBS},'matching_unchanged':True,'GT_unchanged':True}
 transforms=json.loads((R/'experiments/kamp_v2_baselines/data_audit.json').read_text())['transforms']
 original_load=CocoDetection.load_item
 def load_item(self,idx):
  image,target=original_load(self,idx)
  name=Path(self.coco.loadImgs(self.ids[idx])[0]['file_name']).stem
  t=transforms[name]
  # Current fixed pipeline is already letterboxed at 640; no later geometric augmentation.
  assert image.size==(640,640)
  target['ambiguity_pixel_scale']=torch.tensor([t['sx']/640,t['sy']/640,t['sx']/640,t['sy']/640],dtype=torch.float32)
  target['ambiguity_image_bounds']=torch.tensor([t['pad_x']/640,t['pad_y']/640,(t['pad_x']+round(t['width']*t['sx']))/640,(t['pad_y']+round(t['height']*t['sy']))/640],dtype=torch.float32)
  return image,target
 CocoDetection.load_item=load_item
 def matched_meta(targets,indices,key):
  return torch.cat([t[key].reshape(1,4).expand(len(j),4) for t,(_,j) in zip(targets,indices)],0)
 original_boxes=cm.DFINECriterion.loss_boxes
 def loss_boxes(self,outputs,targets,indices,num_boxes,boxes_weight=None):
  losses=original_boxes(self,outputs,targets,indices,num_boxes,boxes_weight)
  if spec.get('edge'):
   idx=self._get_src_permutation_idx(indices)
   pred=box_cxcywh_to_xyxy(outputs['pred_boxes'][idx])
   gt=box_cxcywh_to_xyxy(torch.cat([t['boxes'][j] for t,(_,j) in zip(targets,indices)],0))
   eps=matched_meta(targets,indices,'ambiguity_pixel_scale')*spec.get('epsilon',0)
   error=(pred-gt).abs()
   losses['loss_bbox']=tolerant_penalty(error,eps,spec.get('alpha',1.)).sum()/num_boxes
  losses['loss_bbox']=losses['loss_bbox']*spec.get('l1_factor',1.)
  return losses
 cm.DFINECriterion.loss_boxes=loss_boxes
 original_local=cm.DFINECriterion.loss_local
 original_fgl=cm.DFINECriterion.unimodal_distribution_focal_loss
 context={}
 def encode(ref,box,reg_max,reg_scale,up,eps=.1):
  if not spec.get('fgl'):return original_encode(ref,box,reg_max,reg_scale,up,eps)
  scale=context['scale'];bounds=context['bounds'];epsilon=spec['fgl_epsilon']
  base=dense_encode(ref,box,reg_max,reg_scale,up)
  def one(offset):
   proposed=box+offset*scale*epsilon
   # Clip to true image content, not padded canvas. Enforce each edge stays on its
   # own side of the original center, preserving a positive box under independent jitter.
   center=(box[:,:2]+box[:,2:])/2
   lo=torch.cat([bounds[:,:2],center+scale[:,:2]*.01],-1)
   hi=torch.cat([center-scale[:,:2]*.01,bounds[:,2:]],-1)
   adjusted=proposed.maximum(lo).minimum(hi)
   audit['clipped_coordinates']+=int((proposed!=adjusted).sum().item());audit['jitter_coordinates']+=proposed.numel()
   return dense_encode(ref,adjusted,reg_max,reg_scale,up)
  if spec['fgl']=='jitter':
   # Private generator: jitter must not consume the detector's DN/data RNG stream.
   g=context['rng'];p=torch.tensor(PROBS)
   ids=torch.multinomial(p,box.numel(),replacement=True,generator=g).reshape_as(box)
   q=one(torch.tensor(OFFSETS,device=box.device,dtype=box.dtype)[ids.to(box.device)])
  else:
   q=sum(prob*one(offset) for offset,prob in zip(OFFSETS,PROBS))
  if not torch.isfinite(q).all() or (q<0).any() or not torch.allclose(q.sum(-1),torch.ones_like(q[:,0]),atol=2e-6):raise FloatingPointError('Invalid localization target distribution')
  audit['fgl_calls']+=1;audit['fgl_rows']+=len(q);audit['fgl_changed_rows']+=int(((q-base).abs().sum(-1)>1e-6).sum().item())
  return q,q.new_zeros(len(q)),q.new_zeros(len(q))
 cm.bbox2distance=encode
 def loss_local(self,outputs,targets,indices,num_boxes,T=5):
  if spec.get('fgl') and 'pred_corners' in outputs:
   context['scale']=matched_meta(targets,indices,'ambiguity_pixel_scale')
   context['bounds']=matched_meta(targets,indices,'ambiguity_image_bounds')
   if 'rng' not in context:context['rng']=torch.Generator().manual_seed(torch.initial_seed()+9173)
  return original_local(self,outputs,targets,indices,num_boxes,T)
 cm.DFINECriterion.loss_local=loss_local
 def fgl(self,pred,label,wr,wl,weight=None,reduction='sum',avg_factor=None):
  if label.ndim==1:return original_fgl(self,pred,label,wr,wl,weight,reduction,avg_factor)
  loss=-(label*F.log_softmax(pred,dim=-1)).sum(-1)
  if weight is not None:loss=loss*weight.float()
  if avg_factor is not None:return loss.sum()/avg_factor
  if reduction=='mean':return loss.mean()
  if reduction=='sum':return loss.sum()
  return loss
 cm.DFINECriterion.unimodal_distribution_focal_loss=fgl
 # Inspect each loss before original forward's nan_to_num; preserve matcher and GO union.
 original_get=cm.DFINECriterion.get_loss
 def get_loss(self,*args,**kwargs):
  result=original_get(self,*args,**kwargs)
  for key,value in result.items():
   if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite pre-sanitization loss: '+key)
  return result
 cm.DFINECriterion.get_loss=get_loss
 original_forward=cm.DFINECriterion.forward
 def forward(self,outputs,targets,**kwargs):
  snapshots=[t['boxes'].clone() for t in targets] if audit['calls']<2 else None
  result=original_forward(self,outputs,targets,**kwargs)
  if snapshots is not None:
   assert all(torch.equal(old,t['boxes']) for old,t in zip(snapshots,targets))
  audit['calls']+=1
  if audit_path and audit['calls']%44==0:save()
  return result
 cm.DFINECriterion.forward=forward
 def save():
  if audit_path:
   p=Path(audit_path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(audit,indent=2));tmp.replace(p)
 atexit.register(save)
 return audit
