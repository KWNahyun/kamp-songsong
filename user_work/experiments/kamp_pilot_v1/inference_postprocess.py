"""Optional, GT-free output policy. Not enabled by default in trained baselines."""
import copy,json,argparse
from pathlib import Path

def iou(a,b):
    x,y,w,h=a;u,v,bw,bh=b
    inter=max(0,min(x+w,u+bw)-max(x,u))*max(0,min(y+h,v+bh)-max(y,v))
    return inter/max(w*h+bw*bh-inter,1e-12)

def apply_policy(predictions,nms_iou=None,box_scale=1.):
    if not 0<box_scale<=2:raise ValueError('Invalid box scale')
    if nms_iou is not None and not 0<nms_iou<=1:raise ValueError('Invalid NMS threshold')
    out=[]
    groups=sorted({(p['image_id'],p['category_id']) for p in predictions})
    for image_id,category_id in groups:
        candidates=sorted([copy.deepcopy(p) for p in predictions if p['image_id']==image_id and p['category_id']==category_id],key=lambda p:p['score'],reverse=True)
        selected=[]
        while candidates:
            p=candidates.pop(0);selected.append(p)
            if nms_iou is not None:candidates=[v for v in candidates if iou(v['bbox'],p['bbox'])<=nms_iou]
        for p in selected:
            x,y,w,h=p['bbox'];p['bbox']=[x+w*(1-box_scale)/2,y+h*(1-box_scale)/2,w*box_scale,h*box_scale];out.append(p)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--nms-iou',type=float,default=None);p.add_argument('--box-scale',type=float,default=1.)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output; baseline predictions remain unchanged.')
    a.output.write_text(json.dumps(apply_policy(json.loads(a.input.read_text()),a.nms_iou,a.box_scale)))
