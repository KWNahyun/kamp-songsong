"""Image-only registration, official-label consistency, prediction alignment and ranking."""
from pathlib import Path
import csv,json,sys,collections,itertools
import numpy as np,cv2
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
E=Path(__file__).parent;R=E.parents[1];O=E/'alignment_audit';O.mkdir(exist_ok=True)
sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import iou,dumpcsv
meta=[r for r in csv.DictReader((R/'kamp_xray_v2/split_manifest.csv').open()) if r['split'] in ['train','val']]
raw={r['stem']:R/'data/raw'/r['path'] for r in csv.DictReader((R/'analysis/review/canonical_manifest.csv').open())}
def read(r):
 im=cv2.imread(str(raw[r['image_id']]));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
 mask=(im.max(2)!=im.min(2)).astype('uint8'); mask=cv2.dilate(mask,np.ones((7,7),np.uint8))
 boxes=[];h,w=gray.shape
 for line in (R/'kamp_xray_v2'/r['label_path']).read_text().splitlines():
  _,cx,cy,bw,bh=map(float,line.split());boxes.append([(cx-bw/2)*w,(cy-bh/2)*h,bw*w,bh*h])
 return gray,mask,np.array(boxes)
def centers(b):return b[:,:2]+b[:,2:]/2
def secs(r):
 t=r['image_id'].split('_')[2].split('(')[0];return int(t[:2])*3600+int(t[2:4])*60+int(t[4:6])
groups=collections.defaultdict(list)
for r in meta:groups[(r['split'],r['group'],r['session'],r['image_id'][:3],r['width'],r['height'])].append(r)
pairs=[];boxes_out=[];cache={}
for key,rs in groups.items():
 rs=sorted(rs,key=secs)
 for a,b in zip(rs,rs[1:]):
  if not 0<secs(b)-secs(a)<=120:continue
  ga,ma,ba=cache.setdefault(a['image_id'],read(a));gb,mb,bb=cache.setdefault(b['image_id'],read(b))
  valid=((ma|mb)==0).astype('uint8')*255;warp=np.eye(2,3,dtype=np.float32);reverse=warp.copy()
  rec={'a':a['image_id'],'b':b['image_id'],'split':a['split'],'group':a['group'],'session':a['session'],'gap_seconds':secs(b)-secs(a)}
  try:
   cc,warp=cv2.findTransformECC(ga,gb,warp,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6),valid,5)
   cc2,reverse=cv2.findTransformECC(gb,ga,reverse,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6),valid,5)
   tx,ty=warp[:,2];aligned=cv2.warpAffine(gb,warp,(ga.shape[1],ga.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
   bm=cv2.warpAffine(mb,warp,(ga.shape[1],ga.shape[0]),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderValue=1)
   good=(ma==0)&(bm==0);good[:10]=False;good[-10:]=False;good[:,:10]=False;good[:,-10:]=False
   ncc=float(np.corrcoef(ga[good],aligned[good])[0,1]);cycle=float(np.linalg.norm(warp[:,2]+reverse[:,2]));ok=ncc>=.97 and cycle<=.5 and np.linalg.norm(warp[:,2])<=8
   rec.update(ecc=float(cc),ncc=ncc,cycle_px=cycle,dx=float(tx),dy=float(ty),accepted=bool(ok))
   if ok:
    shifted=bb.copy();shifted[:,:2]-=warp[:,2];dist=np.linalg.norm(centers(ba)[:,None]-centers(shifted)[None],axis=2);ii,jj=linear_sum_assignment(dist)
    for i,j in zip(ii,jj):
     if dist[i,j]>3:continue
     aa=ba[i];bt=shifted[j];edge=np.r_[bt[:2]-aa[:2],bt[:2]+bt[2:]-aa[:2]-aa[2:]]
     boxes_out.append({'a':a['image_id'],'b':b['image_id'],'split':a['split'],'group':a['group'],'a_box':int(i),'b_box':int(j),'ncc':ncc,'dx':float(tx),'dy':float(ty),'center_distance':float(dist[i,j]),'width_ratio':float(bt[2]/aa[2]),'height_ratio':float(bt[3]/aa[3]),'GT_IoU':iou(aa,bt),'max_edge_difference':float(abs(edge).max()),'a_bbox':aa.tolist(),'aligned_b_bbox':bt.tolist()})
  except cv2.error as ex:rec.update(accepted=False,error=str(ex).splitlines()[-1])
  pairs.append(rec)
with (O/'registration.json').open('w') as f:json.dump(pairs,f,indent=2)
dumpcsv(O/'matched_GT_pairs.csv',boxes_out)
gt=json.loads((E/'original/annotations/val.json').read_text());geom=[];rank=[];cross=[]
for job in json.loads((E/'queue_status.json').read_text())['jobs']:
 name=job['name'];pred=json.loads((E/'runs'/name/'common_eval/predictions_original.json').read_text())
 for a in gt['annotations']:
  pp=[p for p in pred if p['image_id']==a['image_id']];near=[p for p in pp if iou(a['bbox'],p['bbox'])>=.1]
  best=max(pp,key=lambda p:iou(a['bbox'],p['bbox']),default=None);leader=max(near,key=lambda p:p['score'],default=None)
  if leader:
   g=np.array(a['bbox']);b=np.array(leader['bbox']);delta=centers(b[None])[0]-centers(g[None])[0];edge=np.r_[b[:2]-g[:2],b[:2]+b[2:]-g[:2]-g[2:]]
   geom.append({'run':name,'variant':name.split('_seed')[0],'seed':job['seed'],'gt_id':a['id'],'dx':delta[0],'dy':delta[1],'width_error':b[2]-g[2],'height_error':b[3]-g[3],'left_error':edge[0],'top_error':edge[1],'right_error':edge[2],'bottom_error':edge[3],'IoU':iou(g,b),'bbox':b.tolist()})
  if best and leader:rank.append({'run':name,'gt_id':a['id'],'best_iou':iou(a['bbox'],best['bbox']),'leader_iou':iou(a['bbox'],leader['bbox']),'iou_gap':iou(a['bbox'],best['bbox'])-iou(a['bbox'],leader['bbox']),'best_score':best['score'],'leader_score':leader['score'],'precise_but_bad_leader':iou(a['bbox'],best['bbox'])>=.75 and iou(a['bbox'],leader['bbox'])<.75})
for a in gt['annotations']:
 rr=[r for r in geom if r['gt_id']==a['id']];out={'gt_id':a['id'],'runs':len(rr),'all_runs_below075':all(r['IoU']<.75 for r in rr)}
 for key in ['dx','dy','width_error','height_error','left_error','top_error','right_error','bottom_error']:
  vals=np.array([r[key] for r in rr]);out[key+'_median']=float(np.median(vals));out[key+'_std']=float(vals.std());out[key+'_same_sign_count']=int(max((vals>.5).sum(),(vals<-.5).sum()))
 cross.append(out)
dumpcsv(O/'prediction_alignment.csv',geom);dumpcsv(O/'ranking.csv',rank);dumpcsv(O/'cross_run_agreement.csv',cross)
summary={'test_used':False,'image_pairs':len(pairs),'accepted_pairs':sum(r['accepted'] for r in pairs),'accepted_groups':len({r['group'] for r in pairs if r['accepted']}),'matched_GT_pairs':len(boxes_out),'registration':'raw grayscale, chroma pixels dilated 3px excluded, translation ECC, NCC>=.97, cycle<=.5px, shift<=8px; GT used only after registration for matching within 3px','pair_statistics':{k:{'median':float(np.median([r[k] for r in boxes_out])),'p90':float(np.quantile([r[k] for r in boxes_out],.9))} for k in ['GT_IoU','center_distance','max_edge_difference']},'GT_pairs_iou_below075':sum(r['GT_IoU']<.75 for r in boxes_out),'GT_pairs_iou_below05':sum(r['GT_IoU']<.5 for r in boxes_out),'all12_leaders_below075':sum(r['all_runs_below075'] and r['runs']==12 for r in cross)}
(O/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda v:v.item()));print(json.dumps(summary,indent=2,default=lambda v:v.item()))
# Inspect worst aligned label differences. Image A with its GT and registered B GT.
examples=sorted(boxes_out,key=lambda r:r['GT_IoU'])[:8];canvas=Image.new('RGB',(900,220*len(examples)),(20,20,20));dd=ImageDraw.Draw(canvas)
for k,r in enumerate(examples):
 ga,_,_=cache[r['a']];gb,_,_=cache[r['b']];warp=np.array([[1,0,r['dx']],[0,1,r['dy']]],np.float32);aligned=cv2.warpAffine(gb,warp,(ga.shape[1],ga.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
 b=np.array(r['a_bbox']);cx,cy=centers(b[None])[0];x,y=max(0,int(cx)-25),max(0,int(cy)-25)
 for col,im in enumerate([ga,aligned]):
  crop=Image.fromarray(im.astype('uint8')).convert('RGB').crop((x,y,x+50,y+50)).resize((200,200),Image.Resampling.NEAREST);d=ImageDraw.Draw(crop);bb=r['a_bbox'] if col==0 else r['aligned_b_bbox'];xx,yy,w,h=bb;d.rectangle(((xx-x)*4,(yy-y)*4,(xx+w-x)*4,(yy+h-y)*4),outline='lime' if col==0 else 'orange',width=2);canvas.paste(crop,(col*210,k*220))
 dd.text((430,k*220+10),f"{r['a']}\n{r['b']}\nAligned GT IoU {r['GT_IoU']:.3f}, NCC {r['ncc']:.3f}\nCenter gap {r['center_distance']:.2f}px\nWidth ratio {r['width_ratio']:.2f}, height {r['height_ratio']:.2f}",fill='white')
canvas.save(O/'aligned_GT_examples.png')
