"""Val-only common COCO evaluation in original pixel coordinates, plus FN audit."""
from pathlib import Path
import argparse,json,sys,csv,time,os
import numpy as np
import torch
from PIL import Image
ROOT=Path('/home/viplab/contest'); EXP=ROOT/'experiments/kamp_pilot_v1'
DATA=ROOT/'data/processed/kamp500_telea_v1_640'
ORIG=ROOT/'data/processed/kamp500_telea_v1'
sys.path.insert(0,str(ROOT/'models/D-FINE'))
os.environ['YOLO_CONFIG_DIR']=str(EXP/'ultralytics_settings')

def predict(name,ims):
    if name.startswith('yolo'):
        from ultralytics import YOLO
        model=YOLO(str(EXP/name/'weights/best.pt'))
        for im in ims:
            r=model.predict(str(DATA/'images/val'/im['file_name']),imgsz=640,device=0,conf=.001,iou=.7,max_det=300,rect=False,verbose=False)[0]
            yield im,r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy()
    else:
        if name.endswith('_p2'):
            import dfine_p2
            dfine_p2.install()
        from src.core import YAMLConfig
        cfg=YAMLConfig(str(EXP/f'{name}.yml'))
        model=cfg.model
        state=torch.load(EXP/name/'best_stg1.pth',map_location='cpu',weights_only=False)
        model.load_state_dict(state['ema']['module'] if 'ema' in state else state['model'])
        model.cuda().eval();post=cfg.postprocessor
        for im in ims:
            arr=np.array(Image.open(DATA/'images/val'/im['file_name']))
            x=torch.from_numpy(arr.copy()).permute(2,0,1).unsqueeze(0).float().cuda()/255
            with torch.no_grad():r=post(model(x),torch.tensor([[640,640]],device='cuda'))[0]
            keep=r['scores']>=.001
            yield im,r['boxes'][keep].cpu().numpy(),r['scores'][keep].cpu().numpy()

def iou(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return inter/max(aw*ah+bw*bh-inter,1e-10)

def main(name):
    torch.set_num_threads(4)
    target=EXP/name/'common_eval';target.mkdir(exist_ok=True)
    gt=json.loads((ORIG/'annotations/val.json').read_text());transforms=json.loads((DATA/'transforms.json').read_text())
    predictions=[];start=time.time()
    for im,boxes,scores in predict(name,gt['images']):
        t=transforms[im['file_name']]
        for box,score in zip(boxes,scores):
            x1,y1,x2,y2=map(float,box)
            x1,x2=np.clip([(x1-t['pad_left'])/t['scale_x'],(x2-t['pad_left'])/t['scale_x']],0,im['width'])
            y1,y2=np.clip([(y1-t['pad_top'])/t['scale_y'],(y2-t['pad_top'])/t['scale_y']],0,im['height'])
            if x2>x1 and y2>y1:predictions.append({'image_id':im['id'],'category_id':0,'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)})
    (target/'predictions_original.json').write_text(json.dumps(predictions))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco=COCO(str(ORIG/'annotations/val.json'))
    if predictions:
        dt=coco.loadRes(predictions);ev=COCOeval(coco,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
        stats=ev.stats.tolist()
    else:stats=[0.]*12
    ground={im['id']:[a for a in gt['annotations'] if a['image_id']==im['id']] for im in gt['images']}
    matched=set();events=[]
    for p in sorted(predictions,key=lambda x:x['score'],reverse=True):
        eligible=[a for a in ground[p['image_id']] if a['id'] not in matched]
        hit=max(eligible,key=lambda a:iou(a['bbox'],p['bbox']),default=None)
        good=hit is not None and iou(hit['bbox'],p['bbox'])>=.5
        if good:matched.add(hit['id'])
        events.append((p['score'],int(good),hit['id'] if good else None))
    nimg=len(gt['images']);ngt=len(gt['annotations']);operating={}
    # Thresholds include all equal-score detections together.
    for budget in [.1,.5]:
        tp=fp=0;index=0;chosen={'threshold':1.000001,'TP':0,'FP':0,'matched':set()};seen=set()
        while index<len(events):
            score=events[index][0]
            while index<len(events) and events[index][0]==score:
                _,success,gid=events[index];tp+=success;fp+=1-success
                if success:seen.add(gid)
                index+=1
            if fp/nimg<=budget:chosen={'threshold':score,'TP':tp,'FP':fp,'matched':set(seen)}
            else:break
        ids=chosen.pop('matched');bins={}
        for label,lo,hi in [('lt8',0,8),('8to12',8,12),('12to16',12,16),('ge16',16,float('inf'))]:
            group=[a for a in gt['annotations'] if lo<=min(a['bbox'][2:])<hi]
            count=sum(a['id'] in ids for a in group)
            bins[label]={'GT':len(group),'TP':count,'FN':len(group)-count,'recall':count/len(group) if group else None}
        chosen.update(recall=chosen['TP']/ngt,FP_per_image=chosen['FP']/nimg,size_bins=bins,all_objects_image_recall=sum(all(a['id'] in ids for a in ground[im['id']]) for im in gt['images'])/nimg)
        operating[str(budget)]=chosen
    # Post-output diagnostics only, not causal encoder-stage classification.
    threshold=operating['0.5']['threshold'];audit=[]
    for a in gt['annotations']:
        ps=[p for p in predictions if p['image_id']==a['image_id']]
        overlap=[p for p in ps if iou(a['bbox'],p['bbox'])>=.5]
        best=max([iou(a['bbox'],p['bbox']) for p in ps],default=0)
        stage='has_iou_match_above_threshold' if any(p['score']>=threshold for p in overlap) else ('score_below_threshold' if overlap else 'no_final_iou_match')
        audit.append({'gt_id':a['id'],'image_id':a['image_id'],'short_side_original_px':min(a['bbox'][2:]),'best_final_iou':best,'max_matching_score':max([p['score'] for p in overlap],default=None),'diagnostic':stage})
    with (target/'gt_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    result={'model':name,'split':'val','images':nimg,'objects':ngt,'AP50_95':stats[0],'AP50':stats[1],'AP75':stats[2],'AR100':stats[8],'operating_points':operating,'settings':{'score_floor':.001,'prediction_cap':300,'COCO_maxDets':[1,10,100],'IoU_operating_points':.5,'yolo_NMS_IoU':.7,'dfine_NMS':False},'caution':'single seed 30-epoch pilot; val-selected checkpoint and thresholds; not held-out test or safety estimate; preprocessing artifacts may remain','evaluation_wall_seconds':time.time()-start}
    (target/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);main(p.parse_args().model)
