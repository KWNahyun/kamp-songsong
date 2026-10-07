"""Build immutable-ID, date-grouped KAMP pilot data. Never use GT to edit pixels."""
from pathlib import Path
import csv, json, hashlib, collections
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/home/viplab/contest')
OUT = ROOT / 'data/processed/kamp500_telea_v1'
LABELS = ROOT / 'data/raw/4. X-ray 검사장비 AI 데이터셋/dataset/라벨링 6종 세트/labels'
SEED = 20260929

def digest(b):
    return hashlib.sha256(b).hexdigest()

def preprocess(rgb):
    # All decisions in this function depend only on input pixels, never GT.
    mask = ((rgb.max(2).astype(int)-rgb.min(2).astype(int)) > 30).astype('uint8')*255
    restored = cv2.inpaint(rgb, mask, 2.0, cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[..., None], 3, axis=2), mask

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT/'analysis/review/canonical_manifest.csv').open()))
    assert len(rows) == 500
    dates = sorted({r['date'] for r in rows})
    # Balance sizes, machines and counts using metadata only, never predictions.
    features = np.array([[1]+[int(r['machine']==f'{k}호기') for k in (1,2,3)]+[int(r['boxes'])] for r in rows])
    groups = np.array([features[[r['date']==d for r in rows]].sum(0) for d in dates])
    target = np.array([.7,.15,.15])[:,None]*features.sum(0)
    rng = np.random.default_rng(SEED)
    best = (float('inf'), None)
    for _ in range(40000):
        a = rng.choice(3,len(dates),p=[.60,.20,.20])
        totals = np.array([groups[a==k].sum(0) for k in range(3)])
        if min(np.bincount(a,minlength=3)) < 3 or (totals[:,1:4] < 8).any():
            continue
        cost = np.sum(((totals-target)/np.maximum(target,1))**2 * [4,1,1,1,1])
        if cost < best[0]: best = (cost,a.copy())
    split_file = OUT/'split_manifest.json'
    if split_file.exists():
        assignment = json.loads(split_file.read_text())['date_to_split']
    else:
        assert best[1] is not None
        assignment = {d:['train','val','test'][int(k)] for d,k in zip(dates,best[1])}
        split_file.write_text(json.dumps({'seed':SEED,'group':'calendar_date_across_machines','date_to_split':assignment,'selection':'metadata-only fixed random search; no model results'},indent=2))
    manifest=[]; cocos={}; annotations=collections.Counter(); summary={}; previews=[]
    for split in ['train','val','test']:
        for folder in ['images','labels']: (OUT/folder/split).mkdir(parents=True,exist_ok=True)
        cocos[split]={'info':{'description':'KAMP pilot Telea v1'},'licenses':[],'images':[],'annotations':[],'categories':[{'id':0,'name':'Defect'}]}
    for image_id,r in enumerate(rows,1):
        source=ROOT/'data/raw'/r['path']; split=assignment[r['date']]
        rgb=np.array(Image.open(source).convert('RGB')); processed,mask=preprocess(rgb)
        h,w=rgb.shape[:2]; assert (w,h)==(int(r['width']),int(r['height']))
        assert np.array_equal(processed[...,0],processed[...,1])
        # Inpainting changes only mask pixels; grayscale conversion is separate.
        original_gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        assert np.array_equal(processed[...,0][mask==0],original_gray[mask==0])
        label=LABELS/(r['stem']+'.txt'); txt=label.read_text(); boxes=[]; overlap=0
        for line in txt.splitlines():
            if not line.strip():continue
            cls,cx,cy,bw,bh=map(float,line.split()); assert cls==0 and bw>0 and bh>0
            x,y,bww,bhh=(cx-bw/2)*w,(cy-bh/2)*h,bw*w,bh*h
            assert x>=-1e-4 and y>=-1e-4 and x+bww<=w+1e-4 and y+bhh<=h+1e-4
            boxes.append([x,y,bww,bhh])
            # GT is used only for labels and audit, after processed pixels are fixed.
            overlap += int(mask[max(0,int(np.floor(y))):min(h,int(np.ceil((cy+bh/2)*h))),max(0,int(np.floor(x))):min(w,int(np.ceil((cx+bw/2)*w)))].any())
            annotations[split]+=1
            cocos[split]['annotations'].append({'id':annotations[split],'image_id':image_id,'category_id':0,'bbox':[x,y,bww,bhh],'area':bww*bhh,'iscrowd':0})
        assert len(boxes)==int(r['boxes'])
        filename=r['stem']+'.png'
        Image.fromarray(processed).save(OUT/'images'/split/filename)
        (OUT/'labels'/split/label.name).write_text(txt)
        cocos[split]['images'].append({'id':image_id,'file_name':filename,'width':w,'height':h})
        record=dict(r,split=split,image_id=image_id,source_sha256=digest(source.read_bytes()),pixel_sha256=digest(rgb.tobytes()),processed_pixel_sha256=digest(processed.tobytes()),label_sha256=digest(label.read_bytes()),mask_pixels=int((mask>0).sum()),overlap_boxes=overlap,processed_path=str(OUT/'images'/split/filename))
        manifest.append(record)
        if split!='test':previews.append((record,rgb,processed,mask,boxes))
    (OUT/'annotations').mkdir(exist_ok=True)
    for split,coco in cocos.items():
        (OUT/'annotations'/f'{split}.json').write_text(json.dumps(coco))
        items=[r for r in manifest if r['split']==split]
        summary[split]={'images':len(items),'boxes':len(coco['annotations']),'dates':sorted({r['date'] for r in items}),'machines':dict(collections.Counter(r['machine'] for r in items))}
    assert sum(v['boxes'] for v in summary.values())==1147
    assert len({r['pixel_sha256'] for r in manifest})==500
    assert len({r['processed_pixel_sha256'] for r in manifest})==500
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    with (OUT/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    (OUT/'summary.json').write_text(json.dumps({'splits':summary,'preprocessing':{'chroma_threshold':30,'dilation':0,'telea_radius':2,'output':'original resolution, grayscale RGB PNG'},'split_sha256':digest(split_file.read_bytes()),'manifest_sha256':digest((OUT/'manifest.json').read_bytes()),'GT_overlap_boxes':sum(r['overlap_boxes'] for r in manifest),'exact_pixel_duplicates':0,'normal_images':0},ensure_ascii=False,indent=2))
    (OUT/'data.yaml').write_text(f'path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Defect\n')
    # Representative train/val only; no test images inspected for model development.
    chosen=[]
    for machine in ['1호기','2호기','3호기']:
        subset=[p for p in previews if p[0]['machine']==machine]
        candidates=[max(subset,key=lambda p:int(p[0]['color_values'])),min(subset,key=lambda p:min(min(b[2:]) for b in p[4])),max(subset,key=lambda p:p[0]['overlap_boxes'])]
        for p in candidates:
            if p[0]['stem'] not in [v[0]['stem'] for v in chosen]:chosen.append(p)
    (OUT/'review').mkdir(exist_ok=True)
    for index,p in enumerate(chosen):
        r,rgb,processed,mask,boxes=p
        tiles=[Image.fromarray(rgb),Image.fromarray(mask).convert('RGB'),Image.fromarray(processed),Image.fromarray(processed)]
        dr=ImageDraw.Draw(tiles[3])
        for x,y,bw,bh in boxes:dr.rectangle((x,y,x+bw,y+bh),outline='#00ff80',width=1)
        panel=Image.new('RGB',(rgb.shape[1]*4,rgb.shape[0]+32),'white')
        for j,tile in enumerate(tiles):panel.paste(tile,(j*rgb.shape[1],32))
        ImageDraw.Draw(panel).text((4,6),f"{r['stem']} | RGB / mask / processed / GT (review only)",fill='black')
        panel.save(OUT/'review'/f'example_{index:02d}.png')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
