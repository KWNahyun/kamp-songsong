from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':42,'font.size':11})
fig,ax=plt.subplots(figsize=(13,9)); ax.set(xlim=(0,13),ylim=(0,9));ax.axis('off')
def box(x,y,w,h,title,sub,color='#EEF3F8',edge='#52718A',dashed=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.09',facecolor=color,edgecolor=edge,lw=1.3,linestyle='--' if dashed else '-'))
 ax.text(x+w/2,y+h*.68,title,ha='center',va='center',fontsize=12,fontweight='bold',color='#243443')
 ax.text(x+w/2,y+h*.29,sub,ha='center',va='center',fontsize=9.5,color='#34495A',linespacing=1.5)
def arrow(a,b,dash=False):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.2,color='#526372',linestyle='--' if dash else '-'))
ax.text(.25,8.65,'선택적 검사 운영 구조',fontsize=18,fontweight='bold',color='#203849')
ax.text(.25,8.23,'고정 검출 모델의 출력을 제품 격리 · 작업자 검토 · 통과 후보로 연결하는 현장 확장안',fontsize=11,color='#526372')
box(.3,6.65,2.2,1.05,'X-ray 입력','영상·촬영 조건 확인')
box(3.1,6.65,3.5,1.05,'D-FINE-S + MAL → UQ','후보 박스 · 검출 점수 · 위치 품질')
box(7.2,6.65,2.5,1.05,'NMS 0.7','중복 정리 · 후보 정보 유지')
arrow((2.55,7.18),(3.05,7.18));arrow((6.65,7.18),(7.15,7.18))
box(3.65,4.8,5.6,1.1,'제품 단위 판정 규칙','현장 보정 점수 + 촬영 상태 + 검토 우선 조건',color='#F1F0F8',edge='#81739B',dashed=True)
arrow((8.45,6.6),(8.45,5.95))
box(10.2,4.8,2.45,1.1,'현장 보정 자료','정상·불량 제품\n장비·제품별 기준 설정',color='#F7F7F7',edge='#89949C',dashed=True)
arrow((10.15,5.35),(9.3,5.35),True)
ax.text(6.4,4.4,'이물 존재의 근거와 박스 위치 품질을 구분',ha='center',fontsize=10,color='#635578',bbox=dict(facecolor='white',edgecolor='none',pad=2),zorder=10)
box(.5,2.55,3.65,1.25,'이물 근거 충분','강한 경보 → 제품 격리',color='#FBECE8',edge='#BD6D58',dashed=True)
box(4.7,2.55,3.65,1.25,'판단 불확실','작업자 검토 → 확인·재촬영',color='#FFF5DC',edge='#B79442',dashed=True)
box(8.9,2.55,3.65,1.25,'현장 통과 기준 충족','자동 통과 후보\n정상·불량 자료로 기준 검증 후 적용',color='#EAF4EF',edge='#62917C',dashed=True)
for x in [2.32,6.52,10.72]:
 ax.plot([6.45,x],[4.05,4.05],color='#526372',lw=1.2)
 arrow((x,4.05),(x,3.85))
ax.plot([6.45,6.45],[4.75,4.05],color='#526372',lw=1.2)
box(3.4,.65,6.2,1,'처리 결과 기록 · 정기 재평가','격리 결과 · 작업자 판단 · 사후 검사 → 운영 기준 갱신',color='#F3F5F7',edge='#89949C',dashed=True)
for x in [2.32,6.52,10.72]:
 ax.plot([x,x],[2.5,2.05],color='#526372',lw=1.1)
ax.plot([2.32,10.72],[2.05,2.05],color='#526372',lw=1.1)
arrow((6.52,2.05),(6.52,1.7))
# independent quality override route
ax.plot([1.4,1.4,3.3,3.3],[6.6,6.13,6.13,3.15],color='#AA8640',lw=1.1,linestyle='--')
ax.plot([3.3,3.3,4.35,4.35],[3.15,3.98,3.98,3.15],color="#AA8640",lw=1.1,linestyle="--")
arrow((4.35,3.15),(4.65,3.15),True)
ax.text(.35,5.63,'촬영 이상·새로운 입력 조건은\n점수와 관계없이 검토',fontsize=9.5,color='#8D703B')
ax.text(.3,.13,'실선 상자: 현재 모델 처리   |   점선 상자: 제안하는 운영 확장   |   UQ는 이물 확률이 아닌 박스 위치 품질',fontsize=9.5,color='#526372')
out=Path('/home/viplab/contest/final_project/figures/figure_14_selective_inspection')
fig.savefig(out.with_suffix('.png'),dpi=220,bbox_inches='tight',facecolor='white')
fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
