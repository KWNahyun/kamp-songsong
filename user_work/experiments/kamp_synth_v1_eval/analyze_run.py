from pathlib import Path
import json,sys,time
import pandas as pd,numpy as np
E=Path(__file__).parent;P=json.loads((E/'protocol.json').read_text());S=Path(P['dataset']);name=sys.argv[1];j=next(x for x in P['jobs'] if x['name']==name);D=E/(sys.argv[2] if len(sys.argv)>2 else 'runs')/name;A=D/'analysis';A.mkdir(exist_ok=True);I=pd.read_csv(D/'images.csv');O=pd.read_csv(S/'objects.csv');G={k:v for k,v in O.groupby('image_id')};grid=np.unique(P['threshold_grid']+[j['val_threshold']]);obj=[];ims=[];curves=[];start=time.time()
def mx(s,m):return float(s[m].max()) if m.any() else 0.
for file in sorted(D.glob('batch_*.npz')):
 z=np.load(file)
 for k,idx in enumerate(z['index']):
  im=I.iloc[int(idx)];iid=im.image_id;b=z['boxes'][k];s=z['scores'][k];bs=z['base'][k];q=z['quality'][k];v=z['valid'][k]&(s>=.001);keep=z['keep'][k];bk=z['basekeep'][k];cx=(b[:,0]+b[:,2])/2;cy=(b[:,1]+b[:,3])/2;g=G.get(iid);n=0 if g is None else len(g);near=np.zeros((n,len(b)),bool);ious=np.zeros((n,len(b)));hit=[]
  if g is not None:
   for h,o in enumerate(g.itertuples()):
    near[h]=(np.hypot(cx-o.obj_cx,cy-o.obj_cy)<=8)|((np.floor(cx)>=o.ex1)&(np.floor(cx)<o.ex2)&(np.floor(cy)>=o.ey1)&(np.floor(cy)<o.ey2))
    lt=np.maximum(b[:,:2],[o.x1,o.y1]);rb=np.minimum(b[:,2:],[o.x2,o.y2]);inter=np.maximum(rb-lt,0).prod(1);area=(b[:,2:]-b[:,:2]).prod(1);ga=(o.x2-o.x1)*(o.y2-o.y1);io=inter/np.maximum(area+ga-inter,1e-9);ious[h]=io
    pre=mx(s,near[h]&v);post=mx(s,near[h]&keep);base=mx(bs,near[h]&bk);qual=mx(np.nan_to_num(q),near[h]&keep)
    cand=np.where(near[h]&keep)[0];best=cand[np.argmax(s[cand])] if len(cand) else None
    failure='hit' if post>=j['val_threshold'] else ('no_saved_near_candidate' if pre==0 else ('NMS_removed_operating_candidate' if pre>=j['val_threshold'] else 'below_operating_threshold'))
    rec=o._asdict();rec.update(source_image=im.source_image,pre_score=pre,post_score=post,base_score=base,near_quality=qual,selected_quality=float(q[best]) if best is not None else np.nan,max_iou_pre=float(io[v].max()) if v.any() else 0,max_iou_post=float(io[keep].max()) if keep.any() else 0,selected_iou=float(io[best]) if best is not None else 0,failure=failure,found=post>=j['val_threshold'],base_found_at_same_threshold=base>=j['val_threshold'])
    for pad in [0,2,4,8]:
     pb=b[keep & (s>=j['val_threshold'])].copy();pb[:,:2]-=pad;pb[:,2:]+=pad;inter2=np.maximum(np.minimum(pb[:,2:],[o.x2,o.y2])-np.maximum(pb[:,:2],[o.x1,o.y1]),0).prod(1);rec[f'coverage95_pad{pad}']=bool(np.any(inter2>=.95*ga))
    obj.append(rec);hit.append(post)
  matched=near.any(0);high=keep&(s>=j['val_threshold']);scores=np.sort(s[keep])[::-1];qa=q[high];validq=qa[np.isfinite(qa)]
  ims.append(dict(image_id=iid,split=im['split'],set=im['set'],axis=im.axis,machine=im.machine,source_image=im.source_image,n_objects=n,found=sum(t>=j['val_threshold'] for t in hit),max_score=float(scores[0]) if len(scores) else 0,score_margin=float(scores[0]-scores[1]) if len(scores)>1 else (float(scores[0]) if len(scores) else 0),count=int(high.sum()),unmatched=int((high&~matched).sum()),quality_min=float(validq.min()) if len(validq) else np.nan,score_quality_gap=float(np.nanmax(np.abs(s[high]-q[high]))) if len(validq) else np.nan))
  for t in grid:
   h=keep&(s>=t);curves.append(dict(image_id=iid,split=im['split'],set=im['set'],machine=im.machine,source_image=im.source_image,threshold=t,objects=n,found=sum(x>=t for x in hit),unmatched=int((h&~matched).sum()),alarm=int(h.any())))
F=pd.DataFrame(obj);V=pd.DataFrame(ims);C=pd.DataFrame(curves);F.to_csv(A/'objects.csv',index=False);V.to_csv(A/'images.csv',index=False)
for keys,label in [(['split','set'],'condition'),(['split','axis'],'axis'),(['split','machine'],'machine')]:
 agg=F.groupby(keys).agg(objects=('found','size'),found=('found','sum'),mean_score=('post_score','mean'),mean_quality=('selected_quality','mean'),mean_max_iou=('max_iou_post','mean'));agg['detection_rate']=agg.found/agg.objects
 vi=V.groupby(keys).agg(images=('image_id','size'),unmatched=('unmatched','sum'),alarm_images=('count',lambda s:(s>0).sum()));agg=vi.join(agg);agg['unmatched_per_image']=agg.unmatched/agg.images;agg['alarm_rate']=agg.alarm_images/agg.images;agg.to_csv(A/(label+'_summary.csv'))
C.groupby(['split','set','machine','threshold']).agg(images=('image_id','size'),objects=('objects','sum'),found=('found','sum'),unmatched=('unmatched','sum'),alarm_images=('alarm','sum')).to_csv(A/'threshold_curves.csv')
F.groupby(['split','set','failure']).size().to_csv(A/'failure_counts.csv');F.groupby(['split','set'])[[f'coverage95_pad{x}' for x in [0,2,4,8]]].mean().to_csv(A/'padding_coverage.csv')
# Source-cluster ratio bootstrap: synthetic variants of one original resampled together.
rng=np.random.default_rng(71006);ci=[]
for (sp,se),g in F.groupby(['split','set']):
 a=g.groupby('source_image').agg(n=('found','size'),tp=('found','sum'));ix=rng.integers(0,len(a),(500,len(a)));rr=a.tp.to_numpy()[ix].sum(1)/a.n.to_numpy()[ix].sum(1);ci.append(dict(split=sp,set=se,sources=len(a),lower=float(np.quantile(rr,.025)),upper=float(np.quantile(rr,.975))))
pd.DataFrame(ci).to_csv(A/'source_bootstrap_ci.csv',index=False)
# Secondary synthetic-only calibration, pre-registered grid, never optimize on test.
cal=[]
for mode in ['global','machine','set']:
 vals=C[(C['split']=='val')&(C.objects>0)]
 groups=[('all',vals)] if mode=='global' else vals.groupby(mode)
 for key,g in groups:
  a=g.groupby('threshold')[['objects','found']].sum();a['rate']=a.found/a.objects
  for target in P['calibration_targets']:
   ok=a[a.rate>=target];row=dict(mode=mode,key=key,target=target,feasible=len(ok)>0)
   if len(ok):
    t=float(ok.index.max());te=C[(C['split']=='test')&(C.threshold==t)]
    if mode!='global':te=te[te[mode]==key]
    positive=te[te.objects>0];neg=te[te.objects==0];row.update(threshold=t,test_objects=int(positive.objects.sum()),test_found=int(positive.found.sum()),test_detection_rate=float(positive.found.sum()/positive.objects.sum()),test_unmatched_per_image=float(positive.unmatched.mean()),erased_images=len(neg),erased_alarm_rate=float(neg.alarm.mean()) if len(neg) else None)
   cal.append(row)
pd.DataFrame(cal).to_csv(A/'secondary_calibration.csv',index=False)
# All two-threshold pairs are descriptive; no "best test" selection.
tri=[]
for sp,g in V.groupby('split'):
 for lo in P['threshold_grid']:
  for hi in P['threshold_grid']:
   if hi<=lo:continue
   for control,gg in g.groupby(g.n_objects.eq(0)):
    tri.append(dict(split=sp,erased_control=control,low=lo,high=hi,images=len(gg),pass_candidates=int((gg.max_score<lo).sum()),review=int(((gg.max_score>=lo)&(gg.max_score<hi)).sum()),isolate=int((gg.max_score>=hi).sum())))
pd.DataFrame(tri).to_csv(A/'three_action_scenarios.csv',index=False)
(A/'complete.json').write_text(json.dumps(dict(complete=True,images=len(V),objects=len(F),seconds=time.time()-start,model=name),indent=2));print('ANALYZED',name,len(V),len(F),flush=True)
