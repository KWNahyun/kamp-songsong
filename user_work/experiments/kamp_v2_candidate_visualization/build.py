from pathlib import Path
import json,csv,sys,copy,base64,io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
R=Path('/home/viplab/contest');E=Path(__file__).parent;U=E.parent/'kamp_v2_uq';sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import iou,match,operating,nms,dumpcsv
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
G=json.loads((U/'original/annotations/val.json').read_text());ims={im['id']:im for im in G['images']};ground={iid:[a for a in G['annotations'] if a['image_id']==iid] for iid in ims};meta={r['image_id']:r for r in csv.DictReader((R/'kamp_xray_v2/split_manifest.csv').open())};old=list(csv.DictReader((U/'analysis/metrics.csv').open(encoding='utf-8-sig')))
rows=[];details=[];verification=[]
for seed in [20260929,20260930,20261001]:
 name=f'dfine_mal_UQ_seed{seed}';pred=json.loads((U/'runs_frozen'/name/'common_eval/predictions_original.json').read_text())
 for idx,p in enumerate(pred):p['source_index']=idx
 kept=nms(pred,.7);op=operating(match(kept,ground),.1,66);prev=next(r for r in old if r['run']==name and r['setting']=='nms07');assert op['TP']==int(prev['TP_FP6']) and op['FP']==int(prev['FP']) and abs(op['threshold']-float(prev['threshold']))<1e-12
 keptids={p['source_index'] for p in kept};final=[p for p in kept if p['score']>=op['threshold']]
 for a in G['annotations']:
  iid=a['image_id'];ps=[p for p in pred if p['image_id']==iid];fs=[p for p in final if p['image_id']==iid];near=[p for p in fs if iou(a['bbox'],p['bbox'])>=.1];current=max(near,key=lambda p:p['score'],default=None);oracle=max(ps,key=lambda p:iou(a['bbox'],p['bbox']),default=None)
  oi=iou(a['bbox'],oracle['bbox']) if oracle else 0;ci=iou(a['bbox'],current['bbox']) if current else 0
  oi_final=oracle and oracle['source_index'] in {p['source_index'] for p in final}
  status='최종 출력에 존재' if oi_final else ('NMS 제거 + 점수 미달' if oracle['source_index'] not in keptids and oracle['score']<op['threshold'] else 'NMS 제거' if oracle['source_index'] not in keptids else '점수 미달')
  group='최종 근접 박스 없음' if current is None else '좋음→더 좋음' if ci>=.5 and ci<.75 and oi>=.75 else '정밀→더 정밀' if ci>=.75 and oi-ci>=.05 else '부정확→개선' if ci<.5 and oi>=.5 else '정밀 후보 부족' if oi<.75 else '이미 정밀 / 차이 작음'
  namefile=ims[iid]['file_name'];row=dict(seed=seed,gt_id=a['id'],image_id=iid,file_name=namefile,machine=meta[Path(namefile).stem]['machine'],category=group,current_iou=ci,oracle_iou=oi,iou_gain=oi-ci,current_score=current['score'] if current else None,oracle_score=oracle['score'],oracle_status=status,threshold=op['threshold'],current_index=current['source_index'] if current else None,oracle_index=oracle['source_index'])
  rows.append(row);details.append(dict(**row,gt=a['bbox'],current=current,oracle=oracle,final=fs))
 verification.append(dict(seed=seed,TP=op['TP'],FP=op['FP'],threshold=op['threshold'],reference_matches=True))
dumpcsv(E/'all_cases.csv',rows);(E/'all_cases.json').write_text(json.dumps(details,ensure_ascii=False));(E/'verification.json').write_text(json.dumps(verification,indent=2))
# One predeclared seed; pick mechanism examples, not seed maxima.
rr=[d for d in details if d['seed']==20260929];selected=[]
for machine in ['M1','M2','M3']:
 pool=[d for d in rr if d['category']=='좋음→더 좋음' and d['machine']==machine]
 if pool:selected.append(max(pool,key=lambda d:d['iou_gain']))
for cat in ['정밀→더 정밀','부정확→개선']:
 pool=[d for d in rr if d['category']==cat and d['gt_id'] not in {x['gt_id'] for x in selected}]
 if pool:selected.append(max(pool,key=lambda d:d['iou_gain']))
if 110 not in {d['gt_id'] for d in selected}:selected.append(next(d for d in rr if d['gt_id']==110))
if len(selected)<6:
 pool=[d for d in rr if d['category']=='좋음→더 좋음' and d['gt_id'] not in {x['gt_id'] for x in selected}];selected.append(sorted(pool,key=lambda d:d['iou_gain'])[len(pool)//2])
selected=selected[:6];(E/'selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
colors={'gt':'#19b86a','current':'#ed583d','oracle':'#269edb'}
def drawbox(ax,b,color,style='-',width=1.6):ax.add_patch(Rectangle(b[:2],b[2],b[3],fill=False,ec=color,lw=width,ls=style))
def renderrow(axs,d):
 ar=np.array(Image.open(R/'kamp_xray_v2/images/val'/d['file_name']).convert('L'));x,y,w,h=d['gt'];cx=x+w/2;cy=y+h/2;side=max(26,w+10,h+10);bounds=[cx-side/2,cx+side/2,cy+side/2,cy-side/2]
 for j,ax in enumerate(axs):
  ax.imshow(ar,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(0,ar.shape[1],ar.shape[0],0));ax.set_xticks([]);ax.set_yticks([])
  if j==0:
   for p in d['final']:drawbox(ax,p['bbox'],colors['current'],width=.9)
   drawbox(ax,[bounds[0],bounds[3],side,side],'#ffda48',width=1.4);ax.set_title(f"{d['machine']} · GT {d['gt_id']}\n최종 출력 전체",fontsize=10)
  else:
   ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_xlabel('원본 픽셀 최근접 확대',fontsize=8)
   if j>=2:drawbox(ax,d['gt'],colors['gt'],'--',1.4)
   if j in [2,4] and d['current']:drawbox(ax,d['current']['bbox'],colors['current'])
   if j in [3,4]:drawbox(ax,d['oracle']['bbox'],colors['oracle'])
   title=['','박스 없는 영상',f"현재 선택 · IoU {d['current_iou']:.3f}\nscore {d['current_score']:.3f}" if d['current'] else '현재 근접 출력 없음',f"정답 참조 후보 · IoU {d['oracle_iou']:.3f}\nscore {d['oracle_score']:.3f}",f"겹쳐 보기 · ΔIoU +{d['iou_gain']:.3f}\n{d['oracle_status']}"][j];ax.set_title(title,fontsize=10)
for i,d in enumerate(selected,1):
 fig,axs=plt.subplots(1,5,figsize=(15,4.2),gridspec_kw={'width_ratios':[1.1,1,1,1,1]});renderrow(axs,d);fig.suptitle(f"MAL+UQ · seed 20260929 · {d['category']} · {d['file_name']}",fontsize=12);fig.text(.5,.025,'초록 점선: 공식 TXT  |  주황 실선: 최종 선택  |  파랑 실선: NMS 전 best-IoU 후보 (정답 필요, 실제 개선 모델 아님)',ha='center',fontsize=10);fig.tight_layout(rect=[0,.12,1,.91]);fig.savefig(E/f'case_{i:02d}_gt{d["gt_id"]}.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(len(selected),3,figsize=(10,3.05*len(selected)))
for i,d in enumerate(selected):
 ar=np.array(Image.open(R/'kamp_xray_v2/images/val'/d['file_name']).convert('L'));x,y,w,h=d['gt'];side=max(26,w+10,h+10);cx=x+w/2;cy=y+h/2
 for j,ax in enumerate(axs[i]):
  ax.imshow(ar,cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(0,ar.shape[1],ar.shape[0],0));ax.set_xlim(cx-side/2,cx+side/2);ax.set_ylim(cy+side/2,cy-side/2);ax.set_xticks([]);ax.set_yticks([])
  if j>0:drawbox(ax,d['gt'],colors['gt'],'--')
  if j==1 and d['current']:drawbox(ax,d['current']['bbox'],colors['current'])
  if j==2:drawbox(ax,d['oracle']['bbox'],colors['oracle'])
  title=f"{d['machine']} · GT {d['gt_id']} · 원본 확대" if j==0 else f"현재 IoU {d['current_iou']:.3f}" if j==1 else f"정답 참조 IoU {d['oracle_iou']:.3f}"
  ax.set_title(title,fontsize=11)
fig.suptitle('같은 영상, 다른 후보: 공식 TXT와의 정렬 비교\n초록 점선=TXT / 주황=최종 선택 / 파랑=정답 참조 후보',fontsize=14);fig.tight_layout(rect=[0,0,1,.965]);fig.savefig(E/'comparison_sheet.png',dpi=150);plt.close(fig)
# Small inline data: selected images only, lossless original pixels.
inline=[]
for d in selected:
 im=Image.open(R/'kamp_xray_v2/images/val'/d['file_name']).convert('L');buf=io.BytesIO();im.save(buf,format='PNG',optimize=True)
 inline.append(dict(**d,image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),width=im.width,height=im.height))
(E/'inline_data.json').write_text(json.dumps(inline,ensure_ascii=False,separators=(',',':')))
summary={str(seed):{cat:sum(d['seed']==seed and d['category']==cat for d in details) for cat in sorted({r['category'] for r in rows})} for seed in [20260929,20260930,20261001]};(E/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({'selected':[{k:d[k] for k in ['gt_id','machine','category','current_iou','oracle_iou','oracle_status']} for d in selected],'summary':summary,'inline_bytes':(E/'inline_data.json').stat().st_size},ensure_ascii=False,indent=2))
