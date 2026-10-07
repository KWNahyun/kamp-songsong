"""Completed-run audit and validation-only stratified analysis. No training or reports."""
from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/viplab/contest');E=Path(__file__).parent;O=E/'analysis'
manifest=json.loads((E/'manifest.json').read_text());state=json.loads((E/'queue_status.json').read_text());assert len(state['jobs'])==36 and all(j['status']=='completed' for j in state['jobs'])
audits=[];gts=[];fps=[];metrics=[]
for j in manifest['jobs']:
 run=E/'runs'/j['name'];a=json.loads((run/'ambiguity_audit.json').read_text());m=json.loads((run/'common_eval/metrics.json').read_text());logs=[json.loads(x) for x in (run/'log.txt').read_text().splitlines()]
 assert len(logs)==30 and logs[-1]['epoch']==29 and a['calls']==1320
 assert all(np.isfinite(float(v)) for x in logs for k,v in x.items() if k.startswith('train_'))
 assert 'Training time ' in (E/'logs'/f"{j['name']}.log").read_text()
 assert hashlib.sha256((run/'best_stg1.pth').read_bytes()).hexdigest()==m['checkpoint_sha256']
 metrics.append(m)
 audits.append(dict(name=j['name'],variant=j['variant'],seed=j['seed'],epochs=len(logs),calls=a['calls'],changed_fraction=a['fgl_changed_rows']/a['fgl_rows'] if a['fgl_rows'] else None,clipped_coordinates=a['clipped_coordinates'],jitter_coordinates=a['jitter_coordinates'],finite_epoch_logs=True,checkpoint_hash_ok=True,best_epoch=m.get('epoch')))
 d=pd.read_csv(run/'common_eval/gt_diagnostics.csv');d['name']=j['name'];d['variant']=j['variant'];d['seed']=j['seed'];d['size_bin']=pd.cut(d[['width','height']].min(axis=1),bins=[0,8,12,16,float('inf')],right=False,labels=['lt8','8to12','12to16','ge16'])
 for kind in ['best','leader']:
  d[kind+'_edge_MAE']=d[[kind+'_'+v for v in ['left','top','right','bottom']]].abs().mean(axis=1)
  d[kind+'_center_error']=np.sqrt(d[kind+'_center_x']**2+d[kind+'_center_y']**2)
 d['precise_candidate']=d.best_iou>=.75;d['precise_leader']=d.leader_iou>=.75;d['precise_after_nms']=d.best_nms_iou>=.75;d['precise_ranking_gap']=d.precise_candidate & ~d.precise_leader;d['precise_nms_loss']=d.precise_candidate & ~d.precise_after_nms
 gts.append(d)
 events=pd.read_csv(run/'common_eval/events.csv',float_precision='round_trip');sel=events[events.score>=m['threshold']];assert int(sel.tp.sum())==m['TP'] and int((sel.tp==0).sum())==m['FP']
 for cat in ['duplicate','localization','background_or_unlabeled']:
  fps.append(dict(name=j['name'],variant=j['variant'],seed=j['seed'],category=cat,count=int((sel.category==cat).sum())))
g=pd.concat(gts,ignore_index=True);g.to_csv(O/'all_gt_diagnostics.csv',index=False);pd.DataFrame(audits).to_csv(O/'training_audit.csv',index=False);pd.DataFrame(fps).to_csv(O/'fp_categories.csv',index=False)
condition=[]
for axis in ['machine','size_bin']:
 for (name,variant,seed,key),d in g.groupby(['name','variant','seed',axis],observed=True):
  condition.append(dict(name=name,variant=variant,seed=seed,axis=axis,condition=str(key),GT=len(d),TP=int(d.TP.sum()),recall=d.TP.mean(),precise_candidate=int(d.precise_candidate.sum()),precise_leader=int(d.precise_leader.sum()),precise_after_nms=int(d.precise_after_nms.sum()),leader_edge_MAE=d.leader_edge_MAE.mean(),leader_width_bias=d.leader_width.mean(),leader_height_bias=d.leader_height.mean()))
pd.DataFrame(condition).to_csv(O/'conditions_per_run.csv',index=False)
geo=g.groupby(['variant','seed'],observed=True).agg(leader_edge_MAE=('leader_edge_MAE','mean'),best_edge_MAE=('best_edge_MAE','mean'),leader_center_error=('leader_center_error','mean'),leader_width_bias=('leader_width','mean'),leader_height_bias=('leader_height','mean'),precise_ranking_gap=('precise_ranking_gap','sum'),precise_nms_loss=('precise_nms_loss','sum'));geo.to_csv(O/'geometry_per_run.csv');geo.groupby('variant').mean().to_csv(O/'geometry_summary.csv')
repeated=g.groupby(['variant','gt_id'],observed=True).agg(misses=('TP',lambda x:int((~x).sum())),machine=('machine','first'),group=('group','first'));repeated[repeated.misses>0].to_csv(O/'repeated_misses.csv')
comp=pd.DataFrame(metrics);keys=['AP','AP75','TP','precise_candidate','precise_leader','precise_after_nms'];means=comp.groupby('variant')[keys].mean().reindex(manifest['variants']);std=comp.groupby('variant')[['AP','AP75']].std().reindex(manifest['variants']);means.to_csv(O/'means.csv')
fig,axs=plt.subplots(1,2,figsize=(12,6));x=np.arange(len(means))
for ax,key in zip(axs,['AP','AP75']):
 ax.errorbar(means[key],x,xerr=std[key],fmt='o',capsize=3,color='#226688');ax.axvline(means.loc['mal',key],color='gray',ls='--');ax.set_yticks(x,means.index);ax.invert_yaxis();ax.set_xlabel(key+' (%)');ax.grid(axis='x',alpha=.2)
fig.suptitle('Validation after NMS 0.7: mean and SD over 3 seeds');fig.tight_layout();fig.savefig(O/'mean_seed_metrics.png',dpi=160);plt.close(fig)
sourcecheck={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in manifest['provenance'].items()};assert all(sourcecheck.values()),sourcecheck
completion=json.loads((E/'completion.json').read_text());result=dict(completed=36,epochs_each=30,all_finite=True,all_source_hashes_match=True,all_checkpoint_hashes_match=True,GT_source_files_unchanged=True,clipped_coordinates=sum(a['clipped_coordinates'] for a in audits),started_KST=datetime.datetime.fromtimestamp(state['started']).isoformat(),ended_KST=datetime.datetime.fromtimestamp(state['ended']).isoformat(),elapsed_minutes=completion['elapsed_seconds']/60,test_used=False,notes=['Original FGL distance support clipping is not separately counted by the added image/box clipping counter.','No per-step gradient magnitudes were persisted; fail-fast gradient checks and successful trainer completion are evidence against detected nonfinite gradients.','orig official GT, matching implementation, NMS, box_scale unchanged.'])
(O/'completed_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));print(means.round(3).to_string());print('GEOMETRY');print(geo.groupby('variant').mean().round(3).to_string());print('FP');print(pd.DataFrame(fps).groupby(['variant','category'])['count'].mean().unstack().round(2).to_string())
