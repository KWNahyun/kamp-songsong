"""Train-defined descriptors; validation-only frozen-model analysis and publication figures."""
from pathlib import Path
import sys,json,csv
import numpy as np,pandas as pd
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
F=Path(__file__).resolve().parents[1];R=F.parent;O=F/'figures';O.mkdir(exist_ok=True);sys.path.insert(0,str(F/'src'))
from metrics import nms,iou
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.facecolor':'white','pdf.fonttype':42})
BLUE='#0072B2';RED='#D55E00';GRAY='#6B7280';GREEN='#009E73'
def save(fig,name):
 fig.savefig(O/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(O/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
meta={r['image_id']:r for r in csv.DictReader((F/'manifests/split_manifest.csv').open())};datasets={s:json.loads((R/f'experiments/kamp_v2_baselines/original/annotations/{s}.json').read_text()) for s in ['train','val']};des=[]
for split,G in datasets.items():
 for im in G['images']:
  anns=[a for a in G['annotations'] if a['image_id']==im['id']];arr=np.array(Image.open(R/'kamp_xray_v2/images'/split/im['file_name']).convert('L'),dtype=float);h,w=arr.shape;occupied=np.zeros((h,w),bool)
  for a in anns:
   x,y,bw,bh=a['bbox'];occupied[max(0,int(np.floor(y))):min(h,int(np.ceil(y+bh))),max(0,int(np.floor(x))):min(w,int(np.ceil(x+bw)))]=True
  for a in anns:
   x,y,bw,bh=a['bbox'];x1,y1=max(0,int(np.floor(x))),max(0,int(np.floor(y)));x2,y2=min(w,int(np.ceil(x+bw))),min(h,int(np.ceil(y+bh)));margin=max(3,round(max(bw,bh)/2));u,v=max(0,x1-margin),max(0,y1-margin);uu,vv=min(w,x2+margin),min(h,y2+margin);inside=arr[y1:y2,x1:x2];ring=arr[v:vv,u:uu][~occupied[v:vv,u:uu]]
   contrast=abs(inside.mean()-ring.mean())/(ring.std()+1.) if len(ring) else np.nan;complexity=ring.std() if len(ring) else np.nan;m=meta[Path(im['file_name']).stem]
   des.append(dict(split=split,gt_id=a['id'],image_id=a['image_id'],file_name=im['file_name'],equipment={'M1':'1호기','M2':'2호기','M3':'3호기'}[m['machine']],group=m['group'],short=min(bw,bh),area=bw*bh,aspect=bw/bh,x=(x+bw/2)/w,y=(y+bh/2)/h,border=min(x/w,y/h,(w-x-bw)/w,(h-y-bh)/h),contrast=contrast,background_std=complexity))
D=pd.DataFrame(des);train=D[D.split=='train'];V=D[D.split=='val'].copy();fail=pd.read_csv(F/'analysis/gt_failures.csv');V=V.merge(fail[['gt_id','TP','stage']],on='gt_id',validate='one_to_one');edges={};rows=[]
for c in ['x','y','border','aspect','contrast','background_std']:
 cut=np.quantile(train[c].dropna(),[1/3,2/3]);edges[c]=cut.tolist();V[c+'_bin']=np.where(V[c]<cut[0],'하위 구간',np.where(V[c]<cut[1],'중간 구간','상위 구간'))
 for label in ['하위 구간','중간 구간','상위 구간']:
  a=V[V[c+'_bin']==label];rows.append(dict(variable=c,bin=label,GT=len(a),TP=int(a.TP.sum()),FN=int((~a.TP).sum()),recall=float(a.TP.mean()),images=a.image_id.nunique(),groups=a.group.nunique()))
D.to_csv(F/'analysis/descriptors_train_val.csv',index=False);V.to_csv(F/'analysis/validation_descriptors.csv',index=False);pd.DataFrame(rows).to_csv(F/'analysis/condition_extended.csv',index=False);(F/'analysis/descriptor_protocol.json').write_text(json.dumps(dict(train_quantile_cutpoints=edges,contrast='abs(mean_inside_GT - mean_surrounding_ring)/(std_surrounding_ring+1 DN); all GT excluded from ring',background='std_surrounding_ring in 8bit gray DN; texture/noise proxy, not pure structural complexity',ring_margin='max(3, round(max(box_width,box_height)/2)) pixels',posthoc_only=True,test_read=False,split_counts=D.groupby('split').size().to_dict()),ensure_ascii=False,indent=2))
# Fig 1: distribution and sampling
fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
for i,c in enumerate(['1호기','2호기','3호기']):
 counts=[D[(D.split==s)&(D.equipment==c)].image_id.nunique() for s in ['train','val']];axs[0].bar(np.array([0,1])+i*.23-.23,counts,.23,label=c,color=[BLUE,GREEN,GRAY][i])
axs[0].set(xticks=[0,1],xticklabels=['학습','검증'],ylabel='영상 수',title='(a) 검사장비별 영상 구성');axs[0].legend(frameon=False)
for split,color,label in [('train',GRAY,'학습: 797개'),('val',BLUE,'검증: 144개')]:
 axs[1].hist(D[D.split==split]['short'],bins=np.arange(4,23),density=True,histtype='step',lw=1.8,color=color,label=label)
axs[1].set(xlabel='공식 박스 짧은 변 (원본 픽셀)',ylabel='확률밀도',title='(b) 주석 박스 크기 분포');axs[1].legend(frameon=False)
for machine,color in [('1호기',BLUE),('2호기',GREEN),('3호기',GRAY)]:
 a=V[V.equipment==machine];axs[2].scatter(a.x,a.y,s=18,alpha=.5,color=color,label=machine)
a=V[~V.TP];axs[2].scatter(a.x,a.y,s=85,marker='x',lw=2,color=RED,label='미탐 2개');axs[2].set(xlim=(0,1),ylim=(1,0),xlabel='박스 중심 x / 영상 폭',ylabel='박스 중심 y / 영상 높이',title='(c) 검증 정답의 위치');axs[2].legend(fontsize=8,frameon=False)
save(fig,'figure_01_sampling')
# Fig 2: continuous descriptors with all values; failures in foreground
fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
for ax,c,title,xlabel in zip(axs,['contrast','background_std','aspect'],['(a) 국소 대비 대리지표','(b) 주변 밝기 변동','(c) 공식 박스 종횡비'],['대비 지표 (무단위)','주변 픽셀 표준편차 (DN)','박스 폭 / 높이']):
 for tp,col,lab in [(True,BLUE,'검출 성공'),(False,RED,'미탐')]:
  a=V[V.TP==tp];ax.scatter(a[c],a['short'],s=25 if tp else 85,marker='o' if tp else 'X',alpha=.45 if tp else 1,color=col,label=lab,zorder=3 if tp else 5)
 for v in edges[c]:ax.axvline(v,color=GRAY,ls=':',lw=1)
 ax.set(xlabel=xlabel,ylabel='박스 짧은 변 (픽셀)',title=title);ax.legend(frameon=False,fontsize=8)
save(fig,'figure_02_descriptors')
# Fig 3: recalls and exact denominators, no invented binomial CI
fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained');labels=['가로 위치','세로 위치','영상 테두리 거리','박스 종횡비','국소 대비','주변 밝기 변동']
for ax,c,title in zip(axs.flat,edges,labels):
 rr=[r for r in rows if r['variable']==c];ax.bar(range(3),[r['recall']*100 for r in rr],color=BLUE,width=.55)
 for i,r in enumerate(rr):ax.text(i,103,f"{r['TP']}/{r['GT']}\n미탐 {r['FN']}개",ha='center',fontsize=9)
 ax.set(xticks=range(3),xticklabels=['하위','중간','상위'],ylim=(0,125),yticks=[0,25,50,75,100],ylabel='객체 재현율 (%)',title=title);ax.axhline(100,color=GRAY,lw=.5)
save(fig,'figure_03_condition_recall')
# Fig 4: frozen output vs oracle on same 40x40 original-pixel crop
G=datasets['val'];anns={a['id']:a for a in G['annotations']};ims={im['id']:im for im in G['images']};raw=json.loads((F/'reference_results/val_raw_predictions.json').read_text());pp=nms(raw,.7);threshold=json.loads((F/'manifests/model.json').read_text())['val_threshold'];cases=[106,110,135,136];fig,axs=plt.subplots(4,3,figsize=(10,12),layout='constrained');case_rows=[]
for row,aid in enumerate(cases):
 a=anns[aid];im=ims[a['image_id']];arr=np.array(Image.open(R/'kamp_xray_v2/images/val'/im['file_name']).convert('L'));x,y,w,h=a['bbox'];cx,cy=x+w/2,y+h/2;left,top=int(cx)-20,int(cy)-20;crop=arr[top:top+40,left:left+40];rs=[p for p in raw if p['image_id']==a['image_id']];alarms=[p for p in pp if p['image_id']==a['image_id'] and p['score']>=threshold and iou(a['bbox'],p['bbox'])>=.1];leader=max(alarms,key=lambda p:p['score'],default=None);best=max(rs,key=lambda p:iou(a['bbox'],p['bbox']));case_rows.append(dict(gt_id=aid,file_name=im['file_name'],crop=[left,top,40,40],best_iou=iou(a['bbox'],best['bbox']),leader_iou=iou(a['bbox'],leader['bbox']) if leader else 0))
 for col in range(3):
  ax=axs[row,col];ax.imshow(crop,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=[left,left+40,top+40,top]);ax.set_xticks([]);ax.set_yticks([])
  if col>0:
   ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=GREEN,lw=1.5))
   pred=leader if col==1 else best
   if pred:
    u,v,bw,bh=pred['bbox'];ax.add_patch(Rectangle((u,v),bw,bh,fill=False,edgecolor=RED if col==1 else BLUE,lw=1.5,ls='--'));ax.text(.02,.02,f"IoU={iou(a['bbox'],pred['bbox']):.3f}",transform=ax.transAxes,color='white',bbox=dict(facecolor='black',alpha=.6,pad=2),fontsize=10)
  else:ax.plot([left+3,left+13],[top+36,top+36],color='white',lw=2);ax.text(left+3,top+34,'10 픽셀',color='white',fontsize=8)
  if row==0:ax.set_title(['(a) 처리 영상 확대','(b) 실제 경보의 최고점수 근처 후보','(c) 정답 기준 최고 IoU 후보'][col],fontsize=10)
 axs[row,0].set_ylabel(f"3호기 · 정답 {aid}\n{'검출 성공' if aid in [106,135] else '미탐'}",fontsize=11)
fig.legend(handles=[Line2D([0],[0],color=GREEN,label='공식 정답 박스'),Line2D([0],[0],color=RED,ls='--',label='실제 경보 후보'),Line2D([0],[0],color=BLUE,ls='--',label='진단용 최선 후보')],loc='outside lower center',ncol=3,frameon=False)
save(fig,'figure_04_cases');(F/'analysis/figure_case_manifest.json').write_text(json.dumps(case_rows,indent=2))
# Fig 5 paired seeds model comparison
A=pd.read_csv(R/'experiments/kamp_v2_baselines/analysis/metrics.csv');B=pd.read_csv(R/'experiments/kamp_v2_uq/analysis/metrics.csv');A=A[(A.setting=='nms07')&A.variant.isin(['yolov8s','yolov8s_p2','dfine_s','dfine_s_p2'])];B=B[(B.setting=='nms07')&B.variant.isin(['dfine_M','dfine_mal_UQ'])];C=pd.concat([A,B]);order=['yolov8s','yolov8s_p2','dfine_s','dfine_s_p2','dfine_M','dfine_mal_UQ'];names=['YOLOv8s','YOLOv8s\n+ 고해상도 경로','D-FINE-S','D-FINE-S\n+ 고해상도 경로','D-FINE-S\n+ MAL','D-FINE-S\n+ MAL + UQ'];summary=C.groupby('variant')[['AP','AP75','TP_FP6']].agg(['mean','std']).reindex(order);summary.to_csv(F/'analysis/model_summary_publication.csv')
fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
for ax,key,title in zip(axs,['AP','AP75','TP_FP6'],['(a) 평균 정밀도','(b) 정밀한 위치 검출','(c) 오탐 예산 내 검출']):
 for k,v in enumerate(order):
  a=C[C.variant==v][key].to_numpy();ax.scatter(np.full(3,k)+[-.12,0,.12],a,color=GRAY,s=20,zorder=3);ax.errorbar(k,a.mean(),yerr=a.std(ddof=1),fmt='D',color=BLUE,capsize=4,ms=5)
 ax.set(xticks=range(6),xticklabels=names,title=title,ylabel={'AP':'AP (0–100)','AP75':'AP75 (0–100)','TP_FP6':'참양성 수 / 정답 144개'}[key]);ax.tick_params(axis='x',labelsize=8,rotation=45);ax.grid(axis='y',alpha=.2)
save(fig,'figure_05_ablation')
# Fig 6 full and low-FP FROC
curve=pd.read_csv(F/'analysis/FROC.csv');fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,limit,title in zip(axs,[None,1],['(a) 저장 후보 전체 범위','(b) 낮은 오탐 구간 확대']):
 ax.step(np.r_[0,curve.FP_per_image],np.r_[0,curve.recall*100],where='post',color=BLUE);ax.scatter([6/66],[142/144*100],color=RED,label='고정 임계값의 운영점',zorder=4);ax.set(xlabel='오탐 박스 수 / 영상',ylabel='객체 재현율 (%)',ylim=(0,103),title=title)
 if limit:ax.set_xlim(0,limit)
 ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
save(fig,'figure_06_froc')
print(pd.DataFrame(rows).to_string(index=False));print('Figure files created',len(list(O.glob('*.png'))))
