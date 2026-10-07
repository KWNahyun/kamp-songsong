from pathlib import Path
import json,csv,collections,sys
import numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).parent;R=E.parents[1];O=E/'analysis';sys.path.insert(0,str(E.parent/'kamp_pilot_v1'))
from analyze_failures import dumpcsv,iou
gt=json.loads((E/'original/annotations/val.json').read_text()); ground={im['id']:[a for a in gt['annotations'] if a['image_id']==im['id']] for im in gt['images']}
def center(b):x,y,w,h=b;return np.array([x+w/2,y+h/2])
centers=[]; stages=[]; transitions=[]
for j in json.loads((E/'queue_status.json').read_text())['jobs']:
 name=j['name'];pred=json.loads((E/'runs'/name/'common_eval/predictions_original.json').read_text()); matched=set();tp=fp=0
 for p in sorted([p for p in pred if p['score']>=.25],key=lambda p:-p['score']):
  aa=[a for a in ground[p['image_id']] if a['id'] not in matched];a=min(aa,key=lambda a:np.linalg.norm(center(a['bbox'])-center(p['bbox'])),default=None)
  if a is not None and np.linalg.norm(center(a['bbox'])-center(p['bbox']))<=8:tp+=1;matched.add(a['id'])
  else:fp+=1
 centers.append({'run':name,'TP_conf025_center8px':tp,'FP_conf025_center8px':fp,'recall':tp/144})
 if j['family']=='dfine':
  rr=list(csv.DictReader((E/'trace'/name/'gt_stages.csv').open()));by=collections.defaultdict(dict)
  for r in rr:by[r['gt_id']][r['stage']]=r
  for floor in ['0','0.001','0.05']:
   key='best_iou_score_ge_'+floor
   for stage in dict.fromkeys(r['stage'] for r in rr):
    rs=[r for r in rr if r['stage']==stage]
    stages.append({'run':name,'variant':name.split('_seed')[0],'stage':stage,'floor':floor,'coverage50':sum(float(r[key])>=.5 for r in rs),'coverage75':sum(float(r[key])>=.75 for r in rs)})
   transitions.append({'run':name,'floor':floor,'topk_lost75_from_counterfactual':sum(float(d['encoder_all_counterfactual'][key])>=.75 and float(d['topk'][key])<.75 for d in by.values()),'middle75_lost_final':sum(any(float(d[s][key])>=.75 for s in ['pre_bbox','decoder_1','decoder_2']) and float(d['decoder_3'][key])<.75 for d in by.values()),'final75_lost_nms':sum(float(d['decoder_3'][key])>=.75 and float(d['final_nms07'][key])<.75 for d in by.values())})
dumpcsv(O/'center_diagnostics.csv',centers);dumpcsv(O/'stage_coverage.csv',stages);dumpcsv(O/'stage_losses.csv',transitions)
# Largest label plus hard M3 cases, validation only. Labels show boxes, not physical boundaries.
condition=list(csv.DictReader((O/'gt_conditions.csv').open(encoding='utf-8-sig')))
selected=sorted(condition,key=lambda d:-float(d['bbox_short_px']))[:1]
di=list(csv.DictReader((O/'gt_diagnostics.csv').open(encoding='utf-8-sig')))
fails=collections.Counter(int(d['gt_id']) for d in di if d['setting']=='nms07' and d['matched_FP6']=='False')
selected += sorted([d for d in condition if d['machine']=='M3' and d not in selected],key=lambda d:-fails[int(d['gt_id'])])[:5]
canvas=Image.new('RGB',(900,300*len(selected)),(24,24,24));draw=ImageDraw.Draw(canvas)
name='dfine_s_seed20260929';pred=json.loads((E/'runs'/name/'common_eval/predictions_original.json').read_text())
for k,d in enumerate(selected):
 a=next(a for a in gt['annotations'] if a['id']==int(d['gt_id']));im=next(i for i in gt['images'] if i['id']==a['image_id']);arr=Image.open(R/'kamp_xray_v2/images/val'/im['file_name']).convert('RGB')
 x,y,w,h=a['bbox'];cx,cy=center(a['bbox']);left,top=max(0,int(cx)-35),max(0,int(cy)-35);crop=arr.crop((left,top,min(arr.width,left+70),min(arr.height,top+70))).resize((280,280),Image.Resampling.NEAREST);canvas.paste(crop,(0,k*300))
 overlay=crop.copy();dd=ImageDraw.Draw(overlay)
 def box(b):xx,yy,ww,hh=b;return [(xx-left)*4,(yy-top)*4,(xx+ww-left)*4,(yy+hh-top)*4]
 dd.rectangle(box(a['bbox']),outline='lime',width=2)
 ps=[p for p in pred if p['image_id']==a['image_id'] and iou(p['bbox'],a['bbox'])>=.1];p=max(ps,key=lambda p:p['score'],default=None)
 if p:dd.rectangle(box(p['bbox']),outline='red',width=2)
 canvas.paste(overlay,(290,k*300));draw.text((580,k*300+15),f"GT {a['id']} | {d['machine']}\n{d['stem']}\nGT box {w:.1f} x {h:.1f}px\nFN across 12 runs: {fails[a['id']]}\nGreen: official GT\nRed: highest-score nearby D-FINE",fill='white')
canvas.save(O/'failure_examples.png')
print('extra diagnostics complete')
