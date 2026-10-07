"""Publication figures from saved results only; no training or test inference."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
F=Path(__file__).resolve().parents[1];R=F.parent;O=F/'figures'
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42})
blue='#0072B2';orange='#D55E00';green='#009E73';gray='#6B7280'
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(O/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def box(ax,x,y,w,h,text,color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor='white',edgecolor=color,linewidth=1.6));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=gray,lw=1.5))
fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.02,.70,.20,.20,'마커 제거 영상\n640 × 640 입력')
box(ax,.30,.70,.26,.20,'D-FINE-S 검출기\nMAL로 학습 후 고정')
box(ax,.68,.70,.28,.20,'최종 후보\n박스 좌표 · 기존 점수 p')
arrow(ax,(.22,.80),(.30,.80));arrow(ax,(.56,.80),(.68,.80))
box(ax,.05,.30,.40,.24,'후보별 특징 283차원\nquery 특징 256 + 분포 통계 20\n기존 점수 1 + 박스 기하 6')
box(ax,.55,.32,.38,.20,'후보 품질 head: UQ\n62,007개 파라미터 → 품질 q',green)
arrow(ax,(.43,.70),(.25,.54));arrow(ax,(.45,.42),(.55,.42))
box(ax,.55,.03,.38,.18,'최종 점수 √(p × q)\nNMS 0.7 → 고정 임계값 적용')
arrow(ax,(.74,.32),(.74,.21));ax.text(.04,.12,'UQ 학습: 검출기·좌표 고정\n목표: 후보와 정답의 최대 IoU\n추론 시 정답은 사용하지 않음',fontsize=11,color=gray)
ax.set_title('최종 방법: 박스 좌표를 유지하면서 후보 점수를 보정',loc='left',fontsize=15,pad=15);save(fig,'figure_07_method')
# paired negative result
p=R/'experiments/kamp_v2_ambiguity_uq/analysis/paired_UQ.csv';d=pd.read_csv(p);d=d[d.control=='edge'].sort_values('seed')
fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for ax,col,title in zip(axs,['AP','AP75'],['평균 정밀도 변화','정밀 위치 평가 변화']):
 vals=d[col].to_numpy();ax.bar(range(3),vals,color=[green if v>0 else orange for v in vals],width=.5);ax.axhline(0,color=gray,lw=1);ax.set_xticks(range(3),[f'반복 {i+1}' for i in range(3)]);ax.set(ylabel=f'Δ{col} (점)',title=title,ylim=(-1.2,3.6));ax.grid(axis='y',alpha=.18)
 for i,v in enumerate(vals):ax.text(i,v+(.1 if v>=0 else -.12),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top')
fig.suptitle('경계 1픽셀 완화 + 품질 보정 − 경계 표현 대조군 + 품질 보정',fontsize=14);fig.tight_layout();save(fig,'figure_08_ambiguity_control')
# transfer: saved test summary only
u=pd.read_csv(R/'experiments/kamp_v2_uq/analysis/metrics.csv');u=u[(u.setting=='nms07')&u.variant.isin(['dfine_M','dfine_mal_UQ'])]
t=json.loads((F/'reference_results/frozen_test_summary.json').read_text());ag={x['variant']:x for x in t['aggregate']}
fig,axs=plt.subplots(1,3,figsize=(12,4.3));xs=np.arange(2)
for variant,label,c,offset in [('dfine_M','MAL',gray,-.17),('dfine_mal_UQ','MAL + 품질 보정',blue,.17)]:
 v=u[u.variant==variant];a=ag[variant]
 values=[[v.TP_FP6.mean()/144*100,a['recall']*100],[v.FP.mean()/66,a['FP_per_image']],[66/66*100,a['alarm_images']/84*100]]
 for ax,z in zip(axs,values):
  ax.bar(xs+offset,z,.32,label=label,color=c)
  for x,y in zip(xs+offset,z):ax.text(x,y+(1 if max(z)>1 else .003),f'{y:.2f}' if max(z)>1 else f'{y:.3f}',ha='center',fontsize=9)
for ax,title,ylim,ylabel in zip(axs,['객체 단위 재현율','영상당 오검출 박스','양성 영상 경보율'],[(0,115),(0,.24),(0,115)],['재현율 (%)','오검출 박스 수 / 영상','경보 영상 비율 (%)']):
 ax.set_xticks(xs,['검증','기존 test']);ax.set(title=title,ylim=ylim,ylabel=ylabel);ax.grid(axis='y',alpha=.15)
axs[0].legend(fontsize=9,loc='lower left');fig.suptitle('고정 검증 임계값의 전이: 높은 경보율과 낮은 오검출은 다른 지표',fontsize=14);fig.tight_layout();save(fig,'figure_09_transfer')
# reinspection no new fitting
p2=R/'experiments/kamp_v2_reinspection/review_curves.csv';d=pd.read_csv(p2);d=d[d.budget_images.isin([4,7,14])]
fig,axs=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
names={'low_confidence':'낮은 후보 점수','cross_model_disagreement':'검출기 간 박스 불일치','count_difference':'검출 개수 차이'}
for ax,(seed,g) in zip(axs,d.groupby('seed')):
 q=g[g.signal=='low_confidence'].sort_values('budget_images');x=q.review_fraction*100
 ax.fill_between(x,q.random_p025,q.random_p975,color='lightgray',alpha=.65,label='무작위 95% 범위');ax.plot(x,q.random_expected,'--',color=gray,label='무작위 평균')
 for (signal,label),c in zip(names.items(),[blue,orange,green]):
  z=g[g.signal==signal].sort_values('budget_images');ax.plot(z.review_fraction*100,z.miss_gt_in_selected,'o-',color=c,label=label)
 ax.set(title=f'반복 {list(sorted(d.seed.unique())).index(seed)+1}: 전체 미탐 {int(q.miss_gt_total.iloc[0])}개',xlabel='실제 검토 영상 비율 (%)',ylim=(-.15,3.4),xticks=[6.06,10.61,21.21]);ax.grid(alpha=.15)
axs[0].set_ylabel('검토 목록에 포함된 미탐 객체 수');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,-.09));fig.suptitle('기존 재검사 우선순위 분석: 미탐 포함은 실제 회수를 뜻하지 않음',fontsize=14);fig.tight_layout();save(fig,'figure_10_review')
# tables and provenance
review=d.groupby(['signal','budget_images'])[['miss_gt_in_selected','random_expected','tie_min','tie_max']].mean().reset_index();review.to_csv(F/'analysis/review_summary_publication.csv',index=False)
sources=[p,p2,R/'experiments/kamp_v2_uq/analysis/metrics.csv',F/'reference_results/frozen_test_summary.json']
(F/'analysis/synthesis_sources.json').write_text(json.dumps([{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sources],indent=2))
print(review.to_string(index=False))
