"""Identical deterministic 640-square input for both pilot detector families."""
from pathlib import Path
import json, hashlib
import cv2
import numpy as np
from PIL import Image

ROOT=Path('/home/viplab/contest')
SRC=ROOT/'data/processed/kamp500_telea_v1'
OUT=ROOT/'data/processed/kamp500_telea_v1_640'
SIZE=640

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'annotations').mkdir(exist_ok=True)
    transforms={}; hashes=set(); checks=0
    for split in ['train','val','test']:
        for d in ['images','labels']:(OUT/d/split).mkdir(parents=True,exist_ok=True)
        coco=json.loads((SRC/'annotations'/f'{split}.json').read_text())
        image_info={}
        for im in coco['images']:
            w,h=im['width'],im['height'];scale=SIZE/max(w,h)
            nw,nh=round(w*scale),round(h*scale);left,top=(SIZE-nw)//2,(SIZE-nh)//2
            sx,sy=nw/w,nh/h
            rgb=np.array(Image.open(SRC/'images'/split/im['file_name']))
            arr=np.full((SIZE,SIZE,3),114,dtype=np.uint8)
            arr[top:top+nh,left:left+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
            Image.fromarray(arr).save(OUT/'images'/split/im['file_name'])
            phash=hashlib.sha256(arr.tobytes()).hexdigest();assert phash not in hashes;hashes.add(phash)
            image_info[im['id']]=(sx,sy,left,top,im['file_name'])
            transforms[im['file_name']]={'split':split,'image_id':im['id'],'original_width':w,'original_height':h,'scale_x':sx,'scale_y':sy,'pad_left':left,'pad_top':top,'pixel_sha256':phash}
            im['width']=SIZE;im['height']=SIZE
        labels={im['id']:[] for im in coco['images']}
        for ann in coco['annotations']:
            sx,sy,left,top,name=image_info[ann['image_id']]
            x,y,w,h=ann['bbox'];converted=[x*sx+left,y*sy+top,w*sx,h*sy]
            rx,ry,rw,rh=converted
            assert np.allclose([(rx-left)/sx,(ry-top)/sy,rw/sx,rh/sy],[x,y,w,h],atol=1e-8)
            ann['bbox']=converted;ann['area']=rw*rh
            labels[ann['image_id']].append(f'0 {(rx+rw/2)/SIZE:.10f} {(ry+rh/2)/SIZE:.10f} {rw/SIZE:.10f} {rh/SIZE:.10f}')
            checks+=1
        for im in coco['images']:
            (OUT/'labels'/split/(Path(im['file_name']).stem+'.txt')).write_text('\n'.join(labels[im['id']])+'\n')
        (OUT/'annotations'/f'{split}.json').write_text(json.dumps(coco))
    (OUT/'transforms.json').write_text(json.dumps(transforms,indent=2))
    (OUT/'data.yaml').write_text(f'path: {OUT}\ntrain: images/train\nval: images/val\n# Test intentionally omitted from pilot training configuration.\nnames:\n  0: Defect\n')
    manifest=json.loads((SRC/'manifest.json').read_text())
    for a,b in [('train','val'),('train','test'),('val','test')]:
        for field in ['date','stem','pixel_sha256','processed_pixel_sha256']:
            assert not ({r[field] for r in manifest if r['split']==a}&{r[field] for r in manifest if r['split']==b})
    validation={'passed':True,'images':len(hashes),'bbox_roundtrips':checks,'date_overlap':0,'exact_pixel_overlap':0,'test_predictions_generated':False,'near_duplicate_limit':'date grouping; different-date same-product identity remains unknown'}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation))

if __name__=='__main__':main()
