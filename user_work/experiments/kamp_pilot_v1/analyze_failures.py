"""Frozen validation prediction diagnostics; no test images or retraining."""
from pathlib import Path
import json,csv,collections,contextlib,io,copy
import numpy as np
from PIL import Image,ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/viplab/contest');EXP=ROOT/'experiments/kamp_pilot_v1'
DATA=ROOT/'data/processed/kamp500_telea_v1';OUT=EXP/'failure_analysis'
MODELS=['yolov8s','yolov8s_p2','dfine_s','dfine_s_p2']

def dumpcsv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def iou(a,b):
    x,y,w,h=a;u,v,bw,bh=b
    inter=max(0,min(x+w,u+bw)-max(x,u))*max(0,min(y+h,v+bh)-max(y,v))
    return inter/max(w*h+bw*bh-inter,1e-12)

def match(preds,ground,cut=.5):
    matched=set();events=[]
    for p in sorted(preds,key=lambda x:x['score'],reverse=True):
        anns=ground[p['image_id']];eligible=[a for a in anns if a['id'] not in matched]
        best=max(eligible,key=lambda a:iou(a['bbox'],p['bbox']),default=None)
        ok=best is not None and iou(best['bbox'],p['bbox'])>=cut
        maximum=max([iou(a['bbox'],p['bbox']) for a in anns],default=0)
        category='TP' if ok else ('duplicate' if maximum>=cut else ('localization' if maximum>=.1 else 'background_or_unlabeled'))
        if ok:matched.add(best['id'])
        events.append(dict(score=p['score'],tp=int(ok),gt_id=best['id'] if ok else None,image_id=p['image_id'],category=category,max_iou=maximum,bbox=p['bbox']))
    return events

def operating(events,budget,nimages):
    tp=fp=0;matched=set();best={'threshold':1.000001,'TP':0,'FP':0,'matched':set(),'events':[]}
    idx=0
    while idx<len(events):
        score=events[idx]['score']
        while idx<len(events) and events[idx]['score']==score:
            e=events[idx];tp+=e['tp'];fp+=1-e['tp']
            if e['tp']:matched.add(e['gt_id'])
            idx+=1
        if fp/nimages>budget:break
        best={'threshold':score,'TP':tp,'FP':fp,'matched':set(matched),'events':events[:idx]}
    return best

def coco_metrics(gt,preds):
    with contextlib.redirect_stdout(io.StringIO()):
        c=COCO();c.dataset=gt;c.createIndex()
        if not preds:return [0,0,0]
        e=COCOeval(c,c.loadRes(preds),'bbox');e.evaluate();e.accumulate();e.summarize()
    return e.stats[:3].tolist()

def nms(preds,threshold):
    result=[]
    for iid in sorted({p['image_id'] for p in preds}):
        rest=sorted([p for p in preds if p['image_id']==iid],key=lambda p:p['score'],reverse=True)
        while rest:
            best=rest.pop(0);result.append(best)
            rest=[p for p in rest if iou(p['bbox'],best['bbox'])<=threshold]
    return result

def main():
    OUT.mkdir(exist_ok=True);(OUT/'examples').mkdir(exist_ok=True)
    gt=json.loads((DATA/'annotations/val.json').read_text());ims={im['id']:im for im in gt['images']}
    manifest={r['stem']:r for r in json.loads((DATA/'manifest.json').read_text())}
    ground={iid:[a for a in gt['annotations'] if a['image_id']==iid] for iid in ims}
    maskaudit={(r['stem'],int(r['box_id'])):r for r in csv.DictReader((ROOT/'analysis/review/box_mask_audit.csv').open())}
    # Condition thresholds come from train labels/images only; no test reads.
    descriptors={};train_contrasts=[]
    for split in ['train','val']:
        annotations=json.loads((DATA/'annotations'/f'{split}.json').read_text())
        for im in annotations['images']:
            arr=np.array(Image.open(DATA/'images'/split/im['file_name']))[:,:,0].astype(float)
            anns=[a for a in annotations['annotations'] if a['image_id']==im['id']]
            occupied=np.zeros(arr.shape,dtype=bool)
            for a in anns:
                x,y,w,h=a['bbox'];occupied[max(0,int(np.floor(y))):int(np.ceil(y+h)),max(0,int(np.floor(x))):int(np.ceil(x+w))]=True
            for index,a in enumerate(anns):
                x,y,w,h=a['bbox'];x1,y1=max(0,int(np.floor(x))),max(0,int(np.floor(y)));x2,y2=min(arr.shape[1],int(np.ceil(x+w))),min(arr.shape[0],int(np.ceil(y+h)))
                margin=max(3,round(max(w,h)/2));rx1,ry1=max(0,x1-margin),max(0,y1-margin);rx2,ry2=min(arr.shape[1],x2+margin),min(arr.shape[0],y2+margin)
                ring=arr[ry1:ry2,rx1:rx2][~occupied[ry1:ry2,rx1:rx2]];inside=arr[y1:y2,x1:x2]
                med=np.median(ring);mad=np.median(np.abs(ring-med));contrast=float(abs(np.median(inside)-med)/(1.4826*mad+1))
                if split=='train':train_contrasts.append(contrast)
                else:
                    stem=Path(im['file_name']).stem;r=manifest[stem]
                    descriptors[a['id']]={'gt_id':a['id'],'image_id':im['id'],'stem':stem,'date':r['date'],'machine':r['machine'],'short_px':min(w,h),'size_bin':'lt8' if min(w,h)<8 else ('8to12' if min(w,h)<12 else ('12to16' if min(w,h)<16 else 'ge16')),'contrast_proxy':contrast,'mask_overlap':float(maskaudit[(stem,index)]['raw_mask_fraction'])>0,'object_count':len(anns)}
    cuts=np.quantile(train_contrasts,[1/3,2/3]).tolist()
    for d in descriptors.values():d['contrast_bin']=['low','mid','high'][int(np.searchsorted(cuts,d['contrast_proxy'],side='right'))]
    dumpcsv(OUT/'gt_conditions.csv',list(descriptors.values()))
    results={};condition_rows=[];fp_rows=[];local_rows=[];curves={};predictions={};pair_rows=[];postprocess=[]
    for name in MODELS:
        preds=json.loads((EXP/name/'common_eval/predictions_original.json').read_text());predictions[name]=preds
        events=match(preds,ground);curves[name]=events
        op=operating(events,.1,len(ims));op5=operating(events,.5,len(ims))
        # Verify earlier reported operating points with a fresh matching implementation.
        old=json.loads((EXP/name/'common_eval/metrics.json').read_text())
        assert op['TP']==old['operating_points']['0.1']['TP'] and op['FP']==old['operating_points']['0.1']['FP']
        result={'operating_0.1':{'TP':op['TP'],'FP':op['FP'],'threshold':op['threshold']},'operating_0.5':{'TP':op5['TP'],'FP':op5['FP'],'threshold':op5['threshold']}}
        failures=collections.Counter(e['category'] for e in op['events'] if not e['tp']);result['FP_types_at_0.1']=dict(failures)
        # Top 12 false positives expose what consumes a very small FP budget.
        for rank,e in enumerate([e for e in events if not e['tp']][:12],1):
            stem=Path(ims[e['image_id']]['file_name']).stem;r=manifest[stem]
            fp_rows.append(dict(model=name,fp_rank=rank,score=e['score'],max_iou=e['max_iou'],category=e['category'],image_id=e['image_id'],stem=stem,date=r['date'],machine=r['machine'],bbox=json.dumps(e['bbox'])))
        result['iou_sensitivity']={}
        for cut in [.3,.5,.75]:
            o=operating(match(preds,ground,cut),.1,len(ims));result['iou_sensitivity'][str(cut)]={'TP':o['TP'],'FP':o['FP'],'recall':o['TP']/len(gt['annotations'])}
        for axis in ['size_bin','machine','date','contrast_bin','mask_overlap','object_count']:
            for value in sorted({str(d[axis]) for d in descriptors.values()}):
                group=[d for d in descriptors.values() if str(d[axis])==value]
                condition_rows.append({'model':name,'axis':axis,'value':value,'GT':len(group),'images':len({d['image_id'] for d in group}),'TP':sum(d['gt_id'] in op['matched'] for d in group),'FN':sum(d['gt_id'] not in op['matched'] for d in group)})
        # High-score-first loose matching, one-to-one, excludes score<.05.
        geometry=match([p for p in preds if p['score']>=.05],ground,.1)
        anns={a['id']:a for a in gt['annotations']};local=[]
        for e in geometry:
            if not e['tp']:continue
            a=anns[e['gt_id']];x,y,w,h=e['bbox'];gx,gy,gw,gh=a['bbox'];cx,cy=x+w/2,y+h/2;gcx,gcy=gx+gw/2,gy+gh/2
            ordinary=iou(e['bbox'],a['bbox']);center_fixed=iou([gcx-w/2,gcy-h/2,w,h],a['bbox']);size_fixed=iou([cx-gw/2,cy-gh/2,gw,gh],a['bbox'])
            rec={'model':name,'gt_id':a['id'],'image_id':a['image_id'],'score':e['score'],'iou':ordinary,'dx_px':cx-gcx,'dy_px':cy-gcy,'center_error_px':float(np.hypot(cx-gcx,cy-gcy)),'width_ratio':w/gw,'height_ratio':h/gh,'GT_center_oracle_iou':center_fixed,'GT_size_oracle_iou':size_fixed}
            local.append(rec);local_rows.append(rec)
        result['localization']={'n':len(local),'median_center_error_px':float(np.median([a['center_error_px'] for a in local])),'median_dx':float(np.median([a['dx_px'] for a in local])),'median_dy':float(np.median([a['dy_px'] for a in local])),'median_width_ratio':float(np.median([a['width_ratio'] for a in local])),'median_height_ratio':float(np.median([a['height_ratio'] for a in local])),'IoU75_count':sum(a['iou']>=.75 for a in local),'GT_center_oracle_IoU75_count':sum(a['GT_center_oracle_iou']>=.75 for a in local),'GT_size_oracle_IoU75_count':sum(a['GT_size_oracle_iou']>=.75 for a in local)}
        # Native training curves are used within-family only.
        if name.startswith('yolo'):
            rows=list(csv.DictReader((EXP/name/'results.csv').open()));history=[float(r['metrics/mAP50-95(B)']) for r in rows];trainloss=[float(r['train/box_loss']) for r in rows];valloss=[float(r['val/box_loss']) for r in rows]
        else:
            rows=[json.loads(s) for s in (EXP/name/'log.txt').read_text().splitlines()];history=[r['test_coco_eval_bbox'][0] for r in rows];trainloss=[r['train_loss_bbox'] for r in rows];valloss=[]
        result['curve']={'best_epoch_1based':int(np.argmax(history))+1,'last_AP':history[-1],'best_AP':max(history),'train_bbox_first':trainloss[0],'train_bbox_last':trainloss[-1],'val_bbox_first':valloss[0] if valloss else None,'val_bbox_last':valloss[-1] if valloss else None,'AP_history':history}
        # Frozen-prediction NMS ablation; no GT-dependent inference rule.
        if name.startswith('dfine'):
            for threshold in [.3,.5,.7]:
                modified=nms(preds,threshold);ap=coco_metrics(gt,modified);o=operating(match(modified,ground),.1,len(ims))
                postprocess.append({'model':name,'NMS_IoU':threshold,'AP50_95':ap[0],'AP50':ap[1],'AP75':ap[2],'TP_FP0.1':o['TP'],'FP':o['FP'],'removed_predictions':len(preds)-len(modified)})
        results[name]=result
    for name in ['yolov8s','dfine_s']:
        b=operating(curves[name],.1,len(ims))['matched'];p=operating(curves[name+'_p2'],.1,len(ims))['matched']
        for gid,d in descriptors.items():pair_rows.append(dict(family=name,**d,change='both' if gid in b&p else ('gained' if gid in p-b else ('lost' if gid in b-p else 'neither'))))
    dumpcsv(OUT/'conditions.csv',condition_rows);dumpcsv(OUT/'high_score_FP.csv',fp_rows);dumpcsv(OUT/'localization.csv',local_rows);dumpcsv(OUT/'paired_conditions.csv',pair_rows);dumpcsv(OUT/'NMS_ablation.csv',postprocess)
    (OUT/'summary.json').write_text(json.dumps({'models':results,'train_contrast_tertiles':cuts,'postprocess_ablation':postprocess,'test_used':False},indent=2))
    # Scientific plots; threshold sweeps are descriptive val curves.
    fig,axs=plt.subplots(1,3,figsize=(16,4.6))
    for name,events in curves.items():
        tp=np.cumsum([e['tp'] for e in events]);fp=np.cumsum([1-e['tp'] for e in events])/len(ims)
        axs[0].plot(np.r_[0,fp],np.r_[0,tp/142],label=name)
        loc=[r for r in local_rows if r['model']==name];errors=np.sort([r['center_error_px'] for r in loc]);axs[1].plot(errors,np.arange(1,len(errors)+1)/len(errors),label=name)
        axs[2].plot(np.arange(1,31),results[name]['curve']['AP_history'],label=name)
    axs[0].set(xlim=(0,.6),ylim=(0,1.03),xlabel='False positives / image',ylabel='Object recall (IoU >= 0.5)',title='Validation operating trade-off')
    axs[1].set(xlim=(0,5),ylim=(0,1.03),xlabel='Center error (original pixels)',ylabel='Cumulative fraction',title='Loose matched pairs; score >= 0.05')
    axs[2].set(xlabel='Epoch',ylabel='Native validation AP50:95',title='Within-family convergence only')
    for ax in axs:ax.grid(alpha=.25)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'failure_overview.png',dpi=180);plt.close(fig)
    # Inspect representative high-score FPs and P2 gains/losses, selected explicitly.
    selected=[]
    for name in MODELS:
        first=next(r for r in fp_rows if r['model']==name);selected.append((name,first['image_id'],'top-ranked FP'))
    for family in ['yolov8s','dfine_s']:
        for status in ['gained','lost']:
            candidates=[r for r in pair_rows if r['family']==family and r['change']==status]
            if candidates:selected.append((family+'_p2',candidates[0]['image_id'],status))
    example_records=[]
    for index,(name,iid,reason) in enumerate(selected):
        im=ims[iid];arr=Image.open(DATA/'images/val'/im['file_name']).convert('RGB');draw=ImageDraw.Draw(arr)
        for a in ground[iid]:
            x,y,w,h=a['bbox'];draw.rectangle((x,y,x+w,y+h),outline='#00ff70',width=1)
        threshold=results[name]['operating_0.1']['threshold']
        for p in predictions[name]:
            if p['image_id']!=iid or p['score']<threshold:continue
            x,y,w,h=p['bbox'];draw.rectangle((x,y,x+w,y+h),outline='#ff5050',width=1)
        big=arr.resize((arr.width*2,arr.height*2),Image.Resampling.NEAREST)
        canvas=Image.new('RGB',(big.width,big.height+48),'white');canvas.paste(big,(0,48));ImageDraw.Draw(canvas).text((5,5),f'{name}: {reason}\nGT green / prediction red; threshold={threshold:.4f}',fill='black')
        fn=f'case_{index:02d}.png';canvas.save(OUT/'examples'/fn);example_records.append({'file':fn,'model':name,'image_id':iid,'stem':Path(im['file_name']).stem,'selection':reason})
    dumpcsv(OUT/'examples/index.csv',example_records)
    print(json.dumps({k:{'localization':v['localization'],'FP':v['FP_types_at_0.1'],'curve':{kk:vv for kk,vv in v['curve'].items() if kk!='AP_history'}} for k,v in results.items()},indent=2))

if __name__=='__main__':main()
