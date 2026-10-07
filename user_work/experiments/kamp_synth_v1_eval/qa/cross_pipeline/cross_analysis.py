from pathlib import Path
import json,time
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
E=Path(__file__).parent;P=json.loads((E/'protocol.json').read_text());S=Path(P['dataset']);A=E/'summary';A.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':42});start=time.time();jobs=P['jobs'];tables=[];objects={};images={}
for j in jobs:
 d=E/'runs'/j['name']/'analysis';f=pd.read_csv(d/'objects.csv');v=pd.read_csv(d/'images.csv');objects[j['name']]=f;images[j['name']]=v
 t=pd.read_csv(d/'condition_summary.csv');t['model']=j['variant'];t['seed']=j['seed'];t['threshold']=j['val_threshold'];tables.append(t)
T=pd.concat(tables,ignore_index=True);T.to_csv(A/'all_condition_results.csv',index=False);T.groupby(['model','split','set']).agg(detection_mean=('detection_rate','mean'),detection_seed_sd=('detection_rate','std'),unmatched_mean=('unmatched_per_image','mean'),alarm_mean=('alarm_rate','mean')).to_csv(A/'three_seed_condition_summary.csv')
# Matched object improvements and complementarity at frozen, model-specific thresholds.
pairs=[('yolov8s','yolov8s_p2'),('dfine_s','dfine_s_p2'),('dfine_s','dfine_M'),('dfine_M','dfine_mal_UQ'),('yolov8s','dfine_mal_UQ')];rows=[];rng=np.random.default_rng(90106)
for a,b in pairs:
 for seed in [20260929,20260930,20261001]:
  x=objects[f'{a}_seed{seed}'];y=objects[f'{b}_seed{seed}'];m=x[['image_id','obj','split','set','source_image','found']].merge(y[['image_id','obj','found']],on=['image_id','obj'],suffixes=('_a','_b'),validate='one_to_one')
  for (sp,se),g in m.groupby(['split','set']):
   d=g.found_b.astype(int)-g.found_a.astype(int);cl=pd.DataFrame(dict(source=g.source_image,delta=d,n=1)).groupby('source').sum();ix=rng.integers(0,len(cl),(500,len(cl)));delta=cl.delta.to_numpy()[ix].sum(1)/cl.n.to_numpy()[ix].sum(1)
   rows.append(dict(a=a,b=b,seed=seed,split=sp,set=se,objects=len(g),a_only=int((g.found_a&~g.found_b).sum()),b_only=int((~g.found_a&g.found_b).sum()),both=int((g.found_a&g.found_b).sum()),neither=int((~g.found_a&~g.found_b).sum()),paired_delta=float(d.mean()),source_ci_low=float(np.quantile(delta,.025)),source_ci_high=float(np.quantile(delta,.975))))
pd.DataFrame(rows).to_csv(A/'paired_effects_and_complementarity.csv',index=False)
# Review signals: image-level selection; outcome is included missed objects, not human recovery.
review=[]
for seed in [20260929,20260930,20261001]:
 v=images[f'dfine_mal_UQ_seed{seed}'].copy();y=images[f'yolov8s_seed{seed}'][['image_id','count']].rename(columns={'count':'yolo_count'});v=v.merge(y,on='image_id');v['miss']=v.n_objects-v.found;v['low_score']=-v.max_score;v['low_quality']=-v.quality_min.fillna(0);v['low_margin']=-v.score_margin;v['count_disagreement']=abs(v['count']-v.yolo_count);v['alarm_disagreement']=((v['count']>0)!=(v.yolo_count>0)).astype(float)
 for (sp,se),g in v[v.n_objects>0].groupby(['split','set']):
  for budget in P['review_budgets']:
   k=max(1,int(np.ceil(len(g)*budget)));random=np.array([rng.choice(g['miss'].to_numpy(),size=k,replace=False).sum() for _ in range(1000)])
   for signal in ['low_score','low_quality','low_margin','count_disagreement','alarm_disagreement','score_quality_gap']:
    top=g.sort_values([signal,'image_id'],ascending=[False,True]).head(k);review.append(dict(seed=seed,split=sp,set=se,signal=signal,budget=budget,review_images=k,images=len(g),missed_objects=int(g['miss'].sum()),included_misses=int(top['miss'].sum()),random_mean=float(random.mean()),random_low=float(np.quantile(random,.025)),random_high=float(np.quantile(random,.975)),tie_break='image_id ascending'))
pd.DataFrame(review).to_csv(A/'review_signal_comparison.csv',index=False)
# Paired condition effects use original backgrounds as the comparison unit.
condition_pairs=[('pos_canon_k1.0','pos_random_k1.0'),('pos_canon_k0.5','pos_random_k0.5'),('pos_random_k1.0','pos_random_k0.5'),('gvxr_size_x1','gvxr_size_x0.5'),('gvxr_size_x1','gvxr_size_x5'),('gvxr_shape_sphere','gvxr_shape_wire20')]
ce=[];quality_rows=[]
for j in jobs:
 f=objects[j['name']]
 for aa,bb in condition_pairs:
  for sp in ['val','test']:
   x=f[(f.split==sp)&(f['set']==aa)].groupby('source_image').found.mean();y=f[(f.split==sp)&(f['set']==bb)].groupby('source_image').found.mean();xy=pd.concat([x,y],axis=1,keys=['a','b']).dropna();delta=(xy.b-xy.a).to_numpy();rr=rng.choice(delta,(1000,len(delta)),replace=True).mean(1);ce.append(dict(model=j['variant'],seed=j['seed'],split=sp,condition_a=aa,condition_b=bb,sources=len(xy),mean_source_delta=float(delta.mean()),ci_low=float(np.quantile(rr,.025)),ci_high=float(np.quantile(rr,.975))))
 if j['variant']=='dfine_mal_UQ':
  f=f.copy();f['quality_bin']=pd.cut(f.selected_quality,np.linspace(0,1,11),include_lowest=True)
  for (sp,qb),g in f.groupby(['split','quality_bin'],observed=True):quality_rows.append(dict(seed=j['seed'],split=sp,quality_bin=str(qb),objects=len(g),mean_quality=float(g.selected_quality.mean()),mean_selected_iou=float(g.selected_iou.mean()),meaning='synthetic TXT localization agreement, not contamination probability'))
pd.DataFrame(ce).to_csv(A/'paired_condition_effects.csv',index=False);pd.DataFrame(quality_rows).to_csv(A/'uq_localization_reliability.csv',index=False)

# Posthoc contrast/edge strata, derived from synthetic validation only; thresholds are not GT-dependent at inference.
strata=[]
for j in jobs:
 f=objects[j['name']]
 for field in ['contrast','edge_dist','area']:
  good=f.loc[f.split=='val',field].dropna()
  if len(good)==0:continue
  q=good.quantile([1/3,2/3]).to_numpy();bins=np.unique(np.r_[-np.inf,q,np.inf]);ff=f.copy();ff['bin']=pd.cut(ff[field],bins=bins)
  for (sp,bin),g in ff.groupby(['split','bin'],observed=True):strata.append(dict(model=j['variant'],seed=j['seed'],split=sp,field=field,bin=str(bin),objects=len(g),found=int(g.found.sum()),rate=float(g.found.mean())))
pd.DataFrame(strata).to_csv(A/'posthoc_condition_strata.csv',index=False)
# Invisible positive stratum compared separately, never removed silently.
ident=pd.read_csv(E.parent/'kamp_synth_v1_audit/positive_images_identical_to_erased.csv');invisible=[]
for j in jobs:
 f=objects[j['name']];g=f[f.image_id.isin(ident.image_id)]
 for sp,a in g.groupby('split'):invisible.append(dict(model=j['variant'],seed=j['seed'],split=sp,images=a.image_id.nunique(),objects=len(a),found=int(a.found.sum()),meaning='identical to erased: found alarms cannot distinguish presence'))
pd.DataFrame(invisible).to_csv(A/'identical_pixel_positive_results.csv',index=False)
# Visual summaries: each line is three-seed mean; source CIs remain in run tables.
models=['yolov8s','dfine_s','dfine_M','dfine_mal_UQ'];fig,axs=plt.subplots(1,2,figsize=(12,4.8));size=[.5,.75,1,1.5,2,3,5]
for model in models:
 a=T[(T.model==model)&(T.split=='test')];g=a.groupby('set').detection_rate.mean();axs[0].plot(size,[g.get('gvxr_size_x'+str(x).replace('.0',''),np.nan) for x in size],marker='o',label=model);axs[1].plot(range(4),[g.get(s,np.nan) for s in ['pos_canon_k1.0','pos_random_k1.0','pos_canon_k0.5','pos_random_k0.5']],marker='o',label=model)
axs[0].set(xlabel='시편 대비 크기 배수',ylabel='이물 주변 경보 도달률',title='크기 변화');axs[1].set_xticks(range(4),['원래 자리 1배','이동 자리 1배','원래 자리 0.5배','이동 자리 0.5배'],rotation=15);axs[1].set(title='위치·농도 변화',ylabel='이물 주변 경보 도달률')
for ax in axs:ax.set_ylim(0,1.03);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('고정 임계값의 합성 시험 결과 — 기존 가중치 3회 평균');fig.tight_layout();fig.savefig(A/'size_position.png',dpi=190);fig.savefig(A/'size_position.pdf');plt.close(fig)
f=T[(T.model=='dfine_mal_UQ')&(T.split=='test')].groupby('set').detection_rate.mean();materials=['SUS304','Al','glass','stone','bone','plastic'];arr=np.array([[f.get(f'gvxr_material_{m}_x{x}',np.nan) for x in [1,2,4,8]] for m in materials]);fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(arr,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(4),['1배','2배','4배','8배']);ax.set_yticks(range(6),['SUS304','알루미늄','유리','돌','뼈','플라스틱']);ax.set_title('MAL+UQ: 재질·크기별 경보 도달률\n합성 시험 / 고정 임계값 / 3회 평균')
for y in range(6):
 for x in range(4):ax.text(x,y,f'{arr[y,x]:.1%}',ha='center',va='center',color='white' if arr[y,x]>.55 else 'black')
fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(A/'material_size.png',dpi=190);fig.savefig(A/'material_size.pdf');plt.close(fig)
# Frozen representative threshold curves in a selection of conditions.
j=next(x for x in jobs if x['name']=='dfine_mal_UQ_seed20260930');c=pd.read_csv(E/'runs'/j['name']/'analysis/threshold_curves.csv');c=c[c.split=='test'].groupby(['set','threshold'])[['objects','found','unmatched','images']].sum().reset_index();fig,ax=plt.subplots(figsize=(8,5))
for se in ['pos_canon_k1.0','pos_random_k1.0','gvxr_size_x0.5','gvxr_size_x5','gvxr_shape_wire20','gvxr_material_plastic_x1']:
 g=c[c['set']==se];ax.plot(g.threshold,g.found/g.objects,label=se)
ax.axvline(j['val_threshold'],color='black',ls='--',label='기존 고정 임계값');ax.set(xlabel='검출 점수 임계값',ylabel='이물 주변 경보 도달률',title='임계값에 따른 조건별 반응 — 진단 곡선');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(A/'threshold_conditions.png',dpi=190);fig.savefig(A/'threshold_conditions.pdf');plt.close(fig)
# Eight illustrative failure overlays, model choice fixed beforehand.
f=objects[j['name']];v=images[j['name']].reset_index(drop=True);chosen=f[(f.split=='test')&(~f.found)].sort_values(['set','image_id']).drop_duplicates('set').head(8);fig,axs=plt.subplots(2,4,figsize=(14,8))
for ax,(_,o) in zip(axs.flat,chosen.iterrows()):
 index=int(v.index[v.image_id==o.image_id][0]);z=np.load(E/'runs'/j['name']/f'batch_{index:05d}.npz');a=np.array(Image.open(S/'images/test'/(o.image_id+'.png')));ax.imshow(a,cmap='gray',vmin=0,vmax=255)
 for b,sc in zip(z['boxes'][0][z['keep'][0]],z['scores'][0][z['keep'][0]]):
  if sc>=j['val_threshold']:ax.add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,ec='#FF8C00',lw=.8))
 ax.plot(o.obj_cx,o.obj_cy,'+',color='cyan',ms=8);ax.set_title(o['set']+'\n'+o.failure,fontsize=8);ax.axis('off')
fig.suptitle('합성 시험의 대표 미검출 — 주황: 최종 경보 / 청록: 대상 이물');fig.tight_layout();fig.savefig(A/'failure_examples.png',dpi=170);plt.close(fig)
(A/'complete.json').write_text(json.dumps(dict(complete=True,runs=len(jobs),seconds=time.time()-start,scope='frozen synthetic diagnostics, no training or deployment threshold change'),indent=2));print('CROSS ANALYSIS COMPLETE',flush=True)
