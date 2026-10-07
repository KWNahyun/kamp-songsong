"""Second-pass audit: independently decode files and check every annotation source."""
from pathlib import Path
import json, re, hashlib, collections, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/viplab/contest'); RAW=R/'data/raw'; O=R/'analysis/review';O.mkdir(exist_ok=True)
DS=RAW/'4. X-ray 검사장비 AI 데이터셋/dataset';LD=DS/'라벨링 6종 세트/labels'
canon=json.loads((R/'analysis/canonical_paths.json').read_text()); main={p.stem:np.array([list(map(float,l.split())) for l in p.read_text().splitlines() if l.strip()]).reshape(-1,5) for p in LD.glob('*.txt')}
rows=[];bad=[]
for p in sorted(RAW.rglob('*')):
 if p.suffix.lower() not in ['.bmp','.jpg','.png']:continue
 try:
  with Image.open(p) as im:a=np.array(im.convert('RGB'));mode=im.mode
  rel=str(p.relative_to(RAW));mach=re.search(r'/([123]호기)\(',rel)
  mask=a.max(2).astype(int)-a.min(2).astype(int)>30
  rows.append(dict(path=rel,stem=p.stem,ext=p.suffix.lower(),width=a.shape[1],height=a.shape[0],mode=mode,pixel_sha256=hashlib.sha256(a.tobytes()+str(a.shape).encode()).hexdigest(),machine=mach[1] if mach else '',chroma_pixels=int(mask.sum()),has_main_label=p.stem in main,is_raw='X선이물검출기' in rel))
 except Exception as e:bad.append(dict(path=str(p.relative_to(RAW)),error=str(e)))
i=pd.DataFrame(rows);i.to_csv(O/'fresh_image_inventory.csv',index=False);(O/'undecodable_files.json').write_text(json.dumps(bad,ensure_ascii=False,indent=2))
raw=i[i.is_raw].copy();un=raw[~raw.has_main_label].copy();un.to_csv(O/'unlabeled_raw_files.csv',index=False)
conf=raw.groupby('stem').pixel_sha256.nunique();raw[raw.stem.isin(conf[conf>1].index)].to_csv(O/'stem_pixel_conflicts.csv',index=False)
anns=[]
for p in DS.rglob('*.txt'):
 if p.parent==LD or p.parent.name in ['labels','YOLO_darknet']:
  try:
   a=np.array([list(map(float,l.split())) for l in p.read_text().splitlines() if l.strip()]).reshape(-1,5)
   same=p.stem in main and a.shape==main[p.stem].shape and np.allclose(a,main[p.stem],atol=1e-8,rtol=0)
   valid= bool(np.isfinite(a).all() and (a[:,0]==a[:,0].astype(int)).all() and (a[:,0]>=0).all() and (a[:,3:]>0).all() and (a[:,1:3]-a[:,3:]/2>=-1e-7).all() and (a[:,1:3]+a[:,3:]/2<=1+1e-7).all())
   anns.append(dict(path=str(p.relative_to(RAW)),stem=p.stem,boxes=len(a),valid=valid,in_main=p.stem in main,same_as_main=same,duplicate_rows=len(a)-len(np.unique(a,axis=0))))
  except Exception as e:anns.append(dict(path=str(p.relative_to(RAW)),error=str(e)))
pd.DataFrame(anns).to_csv(O/'all_yolo_labels.csv',index=False)
x=[]
for p in DS.rglob('*.xml'):
 try:
  t=ET.parse(p).getroot();obs=t.findall('object');x.append(dict(path=str(p.relative_to(RAW)),stem=p.stem,filename=t.findtext('filename'),objects=len(obs),classes='|'.join(sorted(set(o.findtext('name','') for o in obs))),in_main=p.stem in main))
 except Exception as e:x.append(dict(path=str(p.relative_to(RAW)),error=str(e)))
pd.DataFrame(x).to_csv(O/'all_xml_labels.csv',index=False)
# Independently check main labels and image-mask overlap; no image edits saved.
bs=[];ims=[]
for stem,rel in canon.items():
 a=np.array(Image.open(RAW/rel).convert('RGB'));h,w=a.shape[:2];cm=(a.max(2).astype(int)-a.min(2).astype(int)>30).astype('uint8'); dm=cv2.dilate(cm,np.ones((3,3),np.uint8));colors=np.unique(a[cm.astype(bool)],axis=0)
 machine=re.search(r'/([123]호기)\(',rel)[1]; lbl=main[stem]
 ims.append(dict(stem=stem,path=rel,machine=machine,prefix=stem[:3],date=stem[4:12],session=stem.split('(')[0],width=w,height=h,boxes=len(lbl),color_values=len(colors)))
 for j,(cl,cx,cy,bw,bh) in enumerate(lbl):
  x1,y1,x2,y2=(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h
  sl=(slice(max(0,int(np.floor(y1))),min(h,int(np.ceil(y2)))),slice(max(0,int(np.floor(x1))),min(w,int(np.ceil(x2)))))
  pix=a[sl];col=cm[sl].astype(bool); vals=np.unique(pix[col],axis=0)
  near=(slice(max(0,int(np.floor(y1))-5),min(h,int(np.ceil(y2))+5)),slice(max(0,int(np.floor(x1))-5),min(w,int(np.ceil(x2))+5)))
  nc=cm[near].astype(bool)
  bs.append(dict(stem=stem,box_id=j,machine=machine,width_px=bw*w,height_px=bh*h,short_side_px=min(bw*w,bh*h),area_px=bw*w*bh*h,raw_mask_fraction=float(col.mean()),dilated_mask_fraction=float(dm[sl].mean()),colors_inside=len(vals),colors_within_5px=len(np.unique(a[near][nc],axis=0)),short_side_at640=min(bw*w,bh*h)*640/max(w,h)))
b=pd.DataFrame(bs);m=pd.DataFrame(ims);b.to_csv(O/'box_mask_audit.csv',index=False);m.to_csv(O/'canonical_manifest.csv',index=False)
mg=m.groupby('machine').agg(images=('stem','size'),boxes=('boxes','sum'),dates=('date','nunique'),sessions=('session','nunique'));mg.to_csv(O/'machine_summary.csv')
# Machine coverage includes raw duplicate counts and unique pixels.
coverage=raw.groupby('machine').agg(files=('stem','size'),stems=('stem','nunique'),unique_pixels=('pixel_sha256','nunique'),with_label_files=('has_main_label','sum'));coverage.to_csv(O/'raw_machine_coverage.csv')
s=dict(image_files=len(i),unique_pixels=i.pixel_sha256.nunique(),undecodable=len(bad),raw_files=len(raw),raw_unique_stems=raw.stem.nunique(),raw_unique_pixels=raw.pixel_sha256.nunique(),unlabeled_raw_files=len(un),unlabeled_raw_stems=un.stem.nunique(),unlabeled_raw_unique_pixels=un.pixel_sha256.nunique(),unlabeled_no_color_files=int((un.chroma_pixels==0).sum()),unlabeled_pixels_matching_labeled=int(un.pixel_sha256.isin(raw[raw.has_main_label].pixel_sha256).sum()),stem_conflicts=conf[conf>1].to_dict(),yolo_files=len(anns),yolo_new_stems=sorted({a['stem'] for a in anns if not a.get('in_main',True)}),yolo_aux_differences=[a for a in anns if a.get('in_main') and not a.get('same_as_main')],yolo_invalid=[a for a in anns if not a.get('valid',False)],xml_files=len(x),xml_in_main=sum(a.get('in_main',False) for a in x),xml_classes=dict(collections.Counter(a.get('classes','ERROR') for a in x)),raw_overlap_boxes=int((b.raw_mask_fraction>0).sum()),dilated_overlap_boxes=int((b.dilated_mask_fraction>0).sum()),raw_overlap_max=float(b.raw_mask_fraction.max()),dilated_overlap_max=float(b.dilated_mask_fraction.max()),multicolor_inside_boxes=int((b.colors_inside>1).sum()),multicolor_near_boxes=int((b.colors_within_5px>1).sum()),colors_per_image=m.color_values.value_counts().to_dict(),machines=mg.to_dict('index'),coverage=coverage.to_dict('index'),prefix_by_machine=pd.crosstab(m.machine,m.prefix).to_dict(),dates=m.date.nunique(),sessions=m.session.nunique(),top3_date_images=int(m.date.value_counts().head(3).sum()),short_side_at640_min=float(b.short_side_at640.min()),short_side_at640_median=float(b.short_side_at640.median()),box_count_by_machine=pd.crosstab(m.machine,m.boxes).to_dict())
(O/'review_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,default=int));print(json.dumps(s,ensure_ascii=False,indent=2,default=int))
fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
mg.images.plot.bar(ax=axs[0],color='#347f8b');axs[0].set_xticklabels(['Machine '+str(j+1) for j in range(len(mg))],rotation=0);axs[0].set(title='Labeled image coverage',xlabel='',ylabel='Images')
axs[1].bar(['Raw color','After dilation'],[s['raw_overlap_boxes'],s['dilated_overlap_boxes']],color=['#347f8b','#b36743']);axs[1].set(title='GT boxes touched by mask',ylabel='Boxes / 1,147')
axs[2].hist(b.short_side_at640,bins=25,color='#746293');axs[2].set(title='Aspect-preserving resize to 640',xlabel='Box short side (pixels)',ylabel='Boxes')
fig.savefig(O/'review_findings.png',dpi=160);plt.close(fig)
