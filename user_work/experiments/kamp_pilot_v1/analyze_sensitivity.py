"""Exploratory val-only inference ablations and paired date-group sensitivity."""
from pathlib import Path
import json,copy,collections
import numpy as np
from analyze_failures import ROOT,EXP,DATA,OUT,MODELS,match,operating,coco_metrics,nms,dumpcsv

def resized(preds,scale):
    out=copy.deepcopy(preds)
    for p in out:
        x,y,w,h=p['bbox'];p['bbox']=[x+w*(1-scale)/2,y+h*(1-scale)/2,w*scale,h*scale]
    return out

def main():
    gt=json.loads((DATA/'annotations/val.json').read_text());n=len(gt['images'])
    ground={im['id']:[a for a in gt['annotations'] if a['image_id']==im['id']] for im in gt['images']}
    meta={r['stem']:r for r in json.loads((DATA/'manifest.json').read_text())}
    dates={im['id']:meta[Path(im['file_name']).stem]['date'] for im in gt['images']}
    groups=sorted(set(dates.values()));groups_gt={d:[a['id'] for a in gt['annotations'] if dates[a['image_id']]==d] for d in groups}
    original={name:json.loads((EXP/name/'common_eval/predictions_original.json').read_text()) for name in MODELS}
    budgets=[];scale_rows=[];allmatched={};image_results=[]
    for name,preds in original.items():
        events=match(preds,ground)
        for maxfp in [0,3,6,7,8,12,16,32]:
            op=operating(events,maxfp/n,n)
            budgets.append({'model':name,'FP_budget_total':maxfp,'FP_per_image_budget':maxfp/n,'TP':op['TP'],'actual_FP':op['FP'],'recall':op['TP']/142,'threshold':op['threshold']})
        op=operating(events,.1,n);allmatched[name]=op['matched']
        alarm={p['image_id'] for p in preds if p['score']>=op['threshold']}
        image_results.append({'model':name,'threshold':op['threshold'],'defect_images':n,'any_box_alarm_images':len(alarm),'any_matched_object_images':len({e['image_id'] for e in op['events'] if e['tp']}),'all_objects_matched_images':sum(all(a['id'] in op['matched'] for a in ground[iid]) for iid in ground)})
        for scale in [.85,.9,.95,1.,1.05]:
            pp=resized(preds,scale);ap=coco_metrics(gt,pp);o=operating(match(pp,ground),.1,n)
            scale_rows.append({'model':name,'box_scale':scale,'NMS':False,'AP50_95':ap[0],'AP50':ap[1],'AP75':ap[2],'TP_FP0.1':o['TP'],'FP':o['FP']})
    rng=np.random.default_rng(20261001);bootstrap={}
    for name in ['yolov8s','dfine_s']:
        b=allmatched[name];p=allmatched[name+'_p2'];diff=[]
        for _ in range(10000):
            selected=rng.choice(groups,size=len(groups),replace=True);ids=[gid for d in selected for gid in groups_gt[d]]
            diff.append(sum((gid in p)-(gid in b) for gid in ids)/len(ids))
        bootstrap[name]={'delta_recall':(len(p)-len(b))/142,'conditional_date_bootstrap_95_percentile':np.quantile(diff,[.025,.975]).tolist(),'groups':len(groups),'draws':10000,'limitation':'Thresholds and checkpoints fixed after val selection; not an unbiased confidence interval; 6 dates only.'}
    # Crossed postprocessing table on D-FINE+P2: size scale vs NMS.
    factorial=[]
    for name in ['dfine_s','dfine_s_p2']:
        for scale in [1.,.95,.9]:
            for suppress in [False,True]:
                pp=nms(original[name],.7) if suppress else original[name]
                pp=resized(pp,scale);ap=coco_metrics(gt,pp);o=operating(match(pp,ground),.1,n)
                factorial.append({'model':name,'NMS_0.7':suppress,'box_scale':scale,'AP50_95':ap[0],'AP50':ap[1],'AP75':ap[2],'TP_FP0.1':o['TP'],'FP':o['FP']})
    dumpcsv(OUT/'fp_budget_sensitivity.csv',budgets);dumpcsv(OUT/'bbox_scale_ablation.csv',scale_rows);dumpcsv(OUT/'image_level_metrics.csv',image_results);dumpcsv(OUT/'postprocess_factorial.csv',factorial)
    (OUT/'date_bootstrap.json').write_text(json.dumps(bootstrap,indent=2))
    print(json.dumps({'bootstrap':bootstrap,'image_level':image_results,'factorial':factorial},indent=2))

if __name__=='__main__':main()
