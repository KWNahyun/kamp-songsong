"""Exploratory group transfer and review-region analysis; frozen val outputs only."""
from pathlib import Path
import json,csv,sys,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).resolve().parents[1];R=F.parent;O=F/'analysis/operational';O.mkdir(exist_ok=True);sys.path.insert(0,str(F/'src'))
from metrics import nms,match,iou
protocol={'scope':'validation-only exploratory operational analysis; validation previously used for model selection','seed':20260930,'model_changed':False,'test_read':False,'threshold_methods':['FP/image <=0.1','pooled recall >=0.95','pooled recall >=0.98','minimum calibration-group recall >=0.95'],'infeasible_fallback':'lowest saved calibration score; flagged infeasible, not certified','review_margins_original_pixels':[0,1,2,4,8],'review_coverage':'continuous union intersection area with official GT; 95% and 100% coverage; no one-to-one requirement','region_threshold':'existing frozen representative threshold','normal_images_available':False}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
G=json.loads((F/'reference_results/val_annotations.json').read_text());raw=json.loads((F/'reference_results/val_raw_predictions.json').read_text());P=nms(raw,.7);manifest=json.loads((F/'manifests/model.json').read_text());t=manifest['val_threshold'];ims={x['id']:x for x in G['images']};gt={i:[a for a in G['annotations'] if a['image_id']==i] for i in ims};meta={x['image_id']:x for x in csv.DictReader((F/'manifests/split_manifest.csv').open())};groups={i:meta[Path(im['file_name']).stem]['group'] for i,im in ims.items()};groupnames=sorted(set(groups.values()));events=pd.DataFrame(match(P,gt));events['group']=events.image_id.map(groups)
def measure(ids,threshold):
 e=events[(events.image_id.isin(ids))&(events.score>=threshold)];n=sum(len(gt[i]) for i in ids);tp=int(e.tp.sum());fp=len(e)-tp
 return dict(GT=n,TP=tp,FN=n-tp,FP=fp,recall=tp/n,FP_per_image=fp/len(ids),images=len(ids))
rows=[]
for held in groupnames:
 trainids=[i for i in ims if groups[i]!=held];validids=[i for i in ims if groups[i]==held];cal=events[events.image_id.isin(trainids)];thresholds=np.array(sorted(set(cal.score),reverse=True));den=sum(len(gt[i]) for i in trainids)
 tab=cal.groupby('score').agg(TP=('tp','sum'),N=('tp','size')).sort_index(ascending=False).cumsum();rec=tab.TP.to_numpy()/den;fp=(tab.N-tab.TP).to_numpy()/len(trainids)
 mins=[]
 for g in groupnames:
  if g==held:continue
  e=cal[(cal.group==g)&(cal.tp==1)];scores=np.sort(e.score.to_numpy());nd=sum(len(gt[i]) for i in trainids if groups[i]==g);mins.append((len(scores)-np.searchsorted(scores,thresholds,side='left'))/nd)
 worst=np.min(mins,axis=0)
 masks={'fp_budget':fp<=.1,'recall95':rec>=.95,'recall98':rec>=.98,'group95':worst>=.95}
 for method,mask in masks.items():
  indices=np.flatnonzero(mask);feasible=bool(len(indices));idx=(indices[-1] if method=='fp_budget' else indices[0]) if feasible else len(thresholds)-1;threshold=float(thresholds[idx]);m=measure(validids,threshold)
  rows.append(dict(held_group=held,method=method,threshold=threshold,calibration_feasible=feasible,calibration_recall=float(rec[idx]),calibration_worst_recall=float(worst[idx]),**m))
D=pd.DataFrame(rows);D.to_csv(O/'group_transfer.csv',index=False);summary=[]
for method,d in D.groupby('method'):
 summary.append(dict(method=method,TP=int(d.TP.sum()),FN=int(d.FN.sum()),FP=int(d.FP.sum()),recall=d.TP.sum()/144,FP_per_image=d.FP.sum()/66,worst_group_recall=d.recall.min(),infeasible_folds=int((~d.calibration_feasible).sum()),policy_defined_all_folds=bool(d.calibration_feasible.all()),fallback_diagnostic_only=bool((~d.calibration_feasible).any()),threshold_min=d.threshold.min(),threshold_max=d.threshold.max()))
S=pd.DataFrame(summary)
for col in ['TP','FN','FP','recall','FP_per_image','worst_group_recall']:
 S['fallback_diagnostic_'+col]=S[col].where(~S.policy_defined_all_folds)
 S.loc[~S.policy_defined_all_folds,col]=np.nan
S.to_csv(O/'group_transfer_summary.csv',index=False)
# Exact continuous union area: no raster rounding for tiny official boxes.
def area_union(rects):
 if not rects:return 0.
 xs=sorted(set(x for r in rects for x in [r[0],r[2]]));total=0.
 for a,b in zip(xs[:-1],xs[1:]):
  seg=sorted((r[1],r[3]) for r in rects if r[0]<b and r[2]>a);height=0.;end=-float('inf')
  for lo,hi in seg:
   height+=max(0.,hi-max(lo,end));end=max(end,hi)
  total+=(b-a)*height
 return total
regions=[];detail=[]
for margin in [0,1,2,4,8]:
 for iid,im in ims.items():
  rect=[]
  for p in P:
   if p['image_id']!=iid or p['score']<t:continue
   x,y,w,h=p['bbox'];rect.append((max(0,x-margin),max(0,y-margin),min(im['width'],x+w+margin),min(im['height'],y+h+margin)))
  for a in gt[iid]:
   x,y,w,h=a['bbox'];clips=[(max(x,u),max(y,v),min(x+w,uu),min(y+h,vv)) for u,v,uu,vv in rect if min(x+w,uu)>max(x,u) and min(y+h,vv)>max(y,v)]
   coverage=area_union(clips)/(w*h);detail.append(dict(margin=margin,gt_id=a['id'],image_id=iid,coverage=coverage,center_covered=any(u<=x+w/2<=uu and v<=y+h/2<=vv for u,v,uu,vv in rect)))
  regions.append(dict(margin=margin,image_id=iid,area_fraction=area_union(rect)/(im['width']*im['height']),region_count=len(rect)))
A=pd.DataFrame(regions);B=pd.DataFrame(detail);A.to_csv(O/'review_image_burden.csv',index=False);B.to_csv(O/'review_gt_coverage.csv',index=False)
rr=[]
for margin,d in B.groupby('margin'):
 a=A[A.margin==margin];rr.append(dict(margin=int(margin),GT=144,center_covered=int(d.center_covered.sum()),GT_coverage95=int((d.coverage>=.95-1e-10).sum()),GT_coverage100=int((d.coverage>=1-1e-10).sum()),mean_GT_coverage=d.coverage.mean(),mean_image_area_fraction=a.area_fraction.mean(),p95_image_area_fraction=a.area_fraction.quantile(.95),mean_regions=a.region_count.mean()))
V=pd.DataFrame(rr);V.to_csv(O/'review_region_summary.csv',index=False)
# Complementarity of the existing YOLO baseline; GT oracle is diagnosis only.
yfile=R/'experiments/kamp_v2_baselines/runs/yolov8s_seed20260930/common_eval/predictions_original.json';Y=nms(json.loads(yfile.read_text()),.7);ym=pd.read_csv(R/'experiments/kamp_v2_baselines/analysis/metrics.csv');yt=float(ym[(ym.variant=='yolov8s')&(ym.seed==20260930)&(ym.setting=='nms07')].threshold.iloc[0]);dm={e['gt_id'] for e in match(P,gt) if e['tp'] and e['score']>=t};ymatched={e['gt_id'] for e in match(Y,gt) if e['tp'] and e['score']>=yt};ysaved={e['gt_id'] for e in match(Y,gt) if e['tp']};allids={a['id'] for a in G['annotations']};comp=[]
for a in G['annotations']:
 if a['id'] in dm:continue
 yp=[p for p in Y if p['image_id']==a['image_id']];comp.append(dict(gt_id=a['id'],YOLO_fixed_matched=a['id'] in ymatched,YOLO_saved_matched=a['id'] in ysaved,YOLO_saved_best_iou=max([iou(a['bbox'],p['bbox']) for p in yp],default=0),YOLO_fixed_best_iou=max([iou(a['bbox'],p['bbox']) for p in yp if p['score']>=yt],default=0)))
C={'dfine_fixed_TP':len(dm),'yolo_fixed_TP':len(ymatched),'either_model_matched_GT_upper_bound':len(dm|ymatched),'incremental_fixed_GT':sorted(ymatched-dm),'incremental_saved_GT':sorted(ysaved-dm),'not_a_fused_detector_metric':True,'FN_details':comp,'yolo_threshold':yt};(O/'complementarity.json').write_text(json.dumps(C,indent=2))
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10})
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/'figures'/f'{name}.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
names={'fp_budget':'오검출 예산','recall95':'전체 재현율 95%','recall98':'전체 재현율 98%','group95':'그룹별 재현율 95%'}
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for method,d in D.groupby('method'):
 ok=d[d.calibration_feasible];axs[0].scatter(ok.FP_per_image,ok.recall*100,label=names[method],s=55,alpha=.65)
axs[0].set(xlabel='제외 그룹의 오검출 박스 / 영상',ylabel='제외 그룹 재현율 (%)',title='(a) 보정 목표를 달성한 fold만 표시');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
q=S.set_index('method').loc[list(names)];axs[1].bar(range(4),q.infeasible_folds,color='#D55E00');axs[1].set(xticks=range(4),xticklabels=list(names.values()),ylabel='조건을 만족하지 못한 보정 fold 수 / 11',ylim=(0,12),title='(b) 보정 단계에서 목표 달성 가능성');axs[1].tick_params(axis='x',rotation=18)
fig.suptitle('촬영 그룹 간 임계값 전이: 모델 고정, validation 내부 탐색',fontsize=14);fig.tight_layout();save(fig,'figure_11_group_transfer')
fig,axs=plt.subplots(1,2,figsize=(11,4.5));axs[0].plot(V.margin,V.GT_coverage95,'o-',label='정답 면적 95% 이상 포함');axs[0].plot(V.margin,V.GT_coverage100,'s--',label='정답 전체 면적 포함');axs[0].plot(V.margin,V.center_covered,':',label='정답 중심 포함');axs[0].set(xlabel='예측 박스에 추가한 여백 (원본 픽셀)',ylabel='조건을 만족하는 정답 수 / 144',ylim=(0,150));axs[0].legend(fontsize=9)
axs[1].plot(V.margin,V.mean_image_area_fraction*100,'o-',label='영상별 표시 면적 비율 평균');axs[1].plot(V.margin,V.p95_image_area_fraction*100,'s--',label='영상별 표시 면적 비율 95백분위');axs[1].set(xlabel='추가 여백 (원본 픽셀)',ylabel='영상에서 차지하는 합집합 면적 (%)');axs[1].legend(fontsize=9)
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('검토 영역의 포함 정도와 표시 부담 — 탐지 AP와 별도 평가',fontsize=14);fig.tight_layout();save(fig,'figure_12_review_regions')
inputs=[F/'reference_results/val_annotations.json',F/'reference_results/val_raw_predictions.json',yfile];(O/'sources.json').write_text(json.dumps([{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],indent=2))
print(S.to_string(index=False));print(V.to_string(index=False));print(json.dumps(C,indent=2))
