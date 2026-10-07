"""Eval-only query tracing; expanded encoder boxes are diagnostic counterfactuals."""
import sys,json,inspect,textwrap,types,csv,argparse,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image
from torchvision.ops import box_iou,nms
R=Path('/home/viplab/contest');E=Path(__file__).parent;V=E.parent/'kamp_ablation_v3';P=E.parent/'kamp_pilot_v1'
sys.path.insert(0,str(R/'models/D-FINE'))
from src.core import YAMLConfig
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy
import src.zoo.dfine.dfine_decoder as module
p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);args=p.parse_args()
torch.set_num_threads(4)
D=R/'data/processed/kamp500_telea_v1_640';gt=json.loads((D/'annotations/val.json').read_text());original=json.loads((R/'data/processed/kamp500_telea_v1/annotations/val.json').read_text());orig_ann={a['id']:a for a in original['annotations']}
O=E/('trace_smoke' if args.limit else 'trace');O.mkdir(exist_ok=True)
source=textwrap.dedent(inspect.getsource(module.TransformerDecoder.forward))
assert 'if self.training or i == self.eval_idx:' in source
source=source.replace('if self.training or i == self.eval_idx:','if True:').replace('if not self.training:\n                break','if not self.training and i == self.eval_idx:\n                break')
ns=dict(module.__dict__);exec(compile(source,'<eval_trace_forward>','exec'),ns);expanded=ns['forward']
jobs=json.loads((V/'queue_status.json').read_text())['jobs']
runs=[{'name':'dfine_s','variant':'dfine_s','seed':20260929,'config':str(P/'dfine_s.yml'),'path':str(P/'dfine_s')}]
runs += [dict(name=j['name'],variant=j['variant'],seed=j['seed'],config=str(V/'configs'/f"{j['name']}.yml"),path=str(V/'runs'/j['name'])) for j in jobs if j['variant'] in ['dfine_s','M'] and j['status']=='trained']
if args.limit:runs=runs[:1]
all_summary=[]
for run in runs:
 outdir=O/run['name'];outdir.mkdir(exist_ok=True)
 cfg=YAMLConfig(run['config']);model=cfg.model.cuda().eval();ckpt=Path(run['path'])/'best_stg1.pth'
 state=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model']);del state
 tr=model.decoder;dec=tr.decoder;original_forward=dec.forward;original_select=tr._select_topk
 capture={};records=[];transitions=[];maxdiff=0.;image_checks=0
 def select(self,memory,logits,anchors,k):
  assert self.query_select_method=='default' and logits.shape[-1]==1
  capture['encoder_all_counterfactual']=(torch.sigmoid(self.enc_bbox_head(memory)+anchors)[0].detach(),logits[0,:,0].sigmoid().detach())
  inds=logits.max(-1).values.topk(k,dim=-1).indices[0];capture['encoder_indices']=inds.detach()
  box,score=capture['encoder_all_counterfactual'];capture['topk']=(box[inds],score[inds])
  return original_select(memory,logits,anchors,k)
 def hook(self,inputs,outputs):capture['decoder_output']=outputs
 for im in gt['images'][:args.limit or None]:
  x=torch.from_numpy(np.array(Image.open(D/'images/val'/im['file_name'])).copy()).permute(2,0,1)[None].float().cuda()/255
  with torch.no_grad():reference=model(x)
  tr._select_topk=types.MethodType(select,tr);dec.forward=types.MethodType(expanded,dec);h=dec.register_forward_hook(hook)
  with torch.no_grad():traced=model(x)
  h.remove();dec.forward=original_forward;tr._select_topk=original_select
  for key in ['pred_boxes','pred_logits']:
   torch.testing.assert_close(traced[key],reference[key],rtol=1e-5,atol=1e-6);maxdiff=max(maxdiff,float((traced[key]-reference[key]).abs().max()))
  image_checks+=1
  boxes,logits,corners,refs,prebox,prelog=capture['decoder_output']
  stages={'encoder_all_counterfactual':capture['encoder_all_counterfactual'],'topk':capture['topk'],'pre_bbox':(prebox[0],prelog[0,:,0].sigmoid())}
  for i in range(len(boxes)):stages[f'decoder_{i+1}']=(boxes[i,0],logits[i,0,:,0].sigmoid())
  finalname=f'decoder_{len(boxes)}';fb,fs=stages[finalname]
  valid=torch.isfinite(fb).all(-1)&(fb[:,2:]>0).all(-1)&(fs>=.001)
  ix=torch.where(valid)[0];keep=ix[nms(box_cxcywh_to_xyxy(fb[ix]),fs[ix],.7)]
  stages['final_nms07']=(fb[keep],fs[keep])
  anns=[a for a in gt['annotations'] if a['image_id']==im['id']]
  gxy=torch.tensor([a['bbox'] for a in anns],device='cuda');gxy[:,2:]+=gxy[:,:2];gxy/=640
  buffers={};overlaps={}
  for name,(b,s) in stages.items():
   b=b.detach();s=s.detach();overlap=box_iou(gxy,box_cxcywh_to_xyxy(b));overlap=torch.nan_to_num(overlap,nan=0);overlap[:,(b[:,2:]<=0).any(-1)]=0;overlaps[name]=overlap
   buffers[name+'_boxes']=b.cpu().numpy();buffers[name+'_scores']=s.cpu().numpy()
   for ai,a in enumerate(anns):
    io=overlap[ai];best=int(io.argmax()) if len(io) else -1;near=torch.where(io>=.1)[0];leader=int(near[s[near].argmax()]) if len(near) else -1
    rr={'image_id':im['id'],'gt_id':a['id'],'short_side_original_px':min(orig_ann[a['id']]['bbox'][2:]),'stage':name,'candidates':len(b),'best_iou':float(io[best]) if best>=0 else 0,'best_query':best,'leader_query':leader,'leader_iou':float(io[leader]) if leader>=0 else 0,'best_score':float(s[best]) if best>=0 else None,'leader_score':float(s[leader]) if leader>=0 else None}
    for threshold in [0,.001,.05]:
     good=s>=threshold;rr[f'best_iou_score_ge_{threshold}']=float(io[good].max()) if good.any() else 0
     rr[f'count_iou50_score_ge_{threshold}']=int(((io>=.5)&good).sum())
    if best>=0:
     gx,gy,gw,gh=a['bbox'];q=b[best].cpu().numpy()*640;rr.update(best_width_relative_error=float((q[2]-gw)/gw),best_height_relative_error=float((q[3]-gh)/gh),best_center_error_input_px=float(np.hypot(q[0]-gx-gw/2,q[1]-gy-gh/2)))
    records.append(rr)
  qstages=['topk','pre_bbox']+[f'decoder_{i+1}' for i in range(len(boxes))]
  for ai,a in enumerate(anns):
   final=overlaps[finalname][ai]
   for name in qstages[:-1]:
    io=overlaps[name][ai];winner=int(io.argmax());transitions.append({'image_id':im['id'],'gt_id':a['id'],'from_stage':name,'query_id':winner,'encoder_index':int(capture['encoder_indices'][winner]),'from_iou':float(io[winner]),'same_query_final_iou':float(final[winner]),'any_query_final_best_iou':float(final.max()),'same_query_final_score':float(fs[winner])})
  buffers['encoder_indices']=capture['encoder_indices'].cpu().numpy();buffers['nms_kept_query_ids']=keep.cpu().numpy();buffers['gt_ids']=np.array([a['id'] for a in anns]);np.savez_compressed(outdir/f"image_{im['id']}.npz",**buffers)
  capture.clear()
 for file,rs in [('gt_stages.csv',records),('query_transitions.csv',transitions)]:
  with (outdir/file).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 for stage in dict.fromkeys(r['stage'] for r in records):
  rr=[r for r in records if r['stage']==stage];all_summary.append({'run':run['name'],'variant':run['variant'],'seed':run['seed'],'stage':stage,'GT':len(rr),'GT_with_candidate_iou50':sum(r['best_iou']>=.5 for r in rr),'GT_with_candidate_iou75':sum(r['best_iou']>=.75 for r in rr),'GT_with_candidate_iou50_score05':sum(r['best_iou_score_ge_0.05']>=.5 for r in rr),'GT_with_multiple_iou50_score05':sum(r['count_iou50_score_ge_0.05']>=2 for r in rr),'GT_best75_leader_below75':sum(r['best_iou_score_ge_0.05']>=.75 and r['leader_iou']<.75 for r in rr)})
 meta=dict(run,images_verified=image_checks,max_final_output_difference=maxdiff,eval_mode_preserved=True,encoder_all_is_counterfactual=True,per_gt_coverage_not_one_to_one_recall=True,test_used=False)
 (outdir/'verification.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
 del model,cfg;torch.cuda.empty_cache()
with (O/'stage_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(all_summary[0]));w.writeheader();w.writerows(all_summary)
(O/'complete.json').write_text(json.dumps({'completed':True,'runs':len(runs),'test_used':False,'warning':'Coverage is per-GT oracle diagnostic, not one-to-one detector recall. Encoder full-grid boxes are counterfactual applications of the same head, not actual selected runtime predictions. NMS normalized coordinates are unclipped diagnostic boxes.'},indent=2))
