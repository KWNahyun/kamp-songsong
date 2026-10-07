"""Same-query comparison before score floors and after fixed NMS."""
from pathlib import Path
import sys,json,csv
import numpy as np,torch
from torchvision.ops import box_iou,nms
E=Path(__file__).parent;O=E/'analysis';sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import dumpcsv
gt=json.loads((E/'data640/annotations/val.json').read_text());instances=[];summaries=[]
for job in json.loads((E/'queue_status.json').read_text())['jobs']:
 name=job['name'];folder=E/'runs_frozen'/name/'query_audit';rows=[];counts={'base_candidates':0,'uq_candidates':0,'crossed_up_001':0,'crossed_down_001':0}
 for im in gt['images']:
  z=np.load(folder/f"image_{im['id']}.npz");b=torch.tensor(z['boxes']);xy=torch.cat([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],1);valid=torch.isfinite(b).all(1)&(b[:,2:]>0).all(1)
  anns=[a for a in gt['annotations'] if a['image_id']==im['id']];g=torch.tensor([a['bbox'] for a in anns]);g[:,2:]+=g[:,:2];g/=640;overlap=torch.nan_to_num(box_iou(g,xy),nan=0);overlap[:,~valid]=0
  sb=torch.tensor(z['base_scores']);su=torch.tensor(z['uq_scores']);mb=valid&(sb>=.001);mu=valid&(su>=.001)
  counts['base_candidates']+=int(mb.sum());counts['uq_candidates']+=int(mu.sum());counts['crossed_up_001']+=int((mu&~mb).sum());counts['crossed_down_001']+=int((mb&~mu).sum())
  for i,a in enumerate(anns):
   io=overlap[i];near=torch.where(valid&(io>=.1))[0];r={'run':name,'base':job['base'],'seed':job['seed'],'gt_id':a['id'],'best_iou_all_valid':float(io.max())}
   for tag,s,mask in [('base',sb,mb),('uq',su,mu)]:
    winner=int(near[s[near].argmax()]) if len(near) else -1
    ix=torch.where(mask)[0];keep=ix[nms(xy[ix],s[ix],.7)]
    r[tag+'_leader_query']=winner;r[tag+'_leader_iou']=float(io[winner]) if winner>=0 else 0
    r[tag+'_best_iou_floor']=float(io[mask].max()) if mask.any() else 0
    r[tag+'_best_iou_nms']=float(io[keep].max()) if len(keep) else 0
   r['leader_changed']=r['base_leader_query']!=r['uq_leader_query'];r['leader_iou_delta']=r['uq_leader_iou']-r['base_leader_iou'];rows.append(r)
 instances.extend(rows)
 out={'run':name,'base':job['base'],'seed':job['seed'],**counts,'leader_changed':sum(r['leader_changed'] for r in rows),'leader_improved':sum(r['leader_iou_delta']>1e-6 for r in rows),'leader_worsened':sum(r['leader_iou_delta']<-1e-6 for r in rows)}
 for tag in ['base','uq']:
  out[tag+'_precise_wrong_leader_all_valid']=sum(r['best_iou_all_valid']>=.75 and r[tag+'_leader_iou']<.75 for r in rows)
  out[tag+'_mean_gap_all_valid']=float(np.mean([r['best_iou_all_valid']-r[tag+'_leader_iou'] for r in rows]))
  out[tag+'_coverage75_floor']=sum(r[tag+'_best_iou_floor']>=.75 for r in rows)
  out[tag+'_coverage75_nms']=sum(r[tag+'_best_iou_nms']>=.75 for r in rows)
  out[tag+'_nms_lost75']=sum(r[tag+'_best_iou_floor']>=.75 and r[tag+'_best_iou_nms']<.75 for r in rows)
 summaries.append(out)
dumpcsv(O/'same_query_instances.csv',instances);dumpcsv(O/'same_query_summary.csv',summaries)
print(json.dumps(summaries,indent=2))
