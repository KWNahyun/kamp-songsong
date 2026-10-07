"""Reproducible image/label audit; no model training or supplied code execution."""
from pathlib import Path
import collections, hashlib, json, re, math
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'data/raw';OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
DS=RAW/'4. X-ray 검사장비 AI 데이터셋/dataset'
LABEL=DS/'라벨링 6종 세트/labels'

def sha(b):return hashlib.sha256(b).hexdigest()
def c_mask(a,thr=30):return (a.max(2).astype(int)-a.min(2).astype(int)>thr).astype('uint8')
def clean(a):
 mask=cv2.dilate(c_mask(a),np.ones((3,3),np.uint8))
 gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 return cv2.inpaint(gray,mask,3,cv2.INPAINT_TELEA) if mask.any() else gray,mask

def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y
 return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
if (OUT/'image_inventory.csv').exists():
 images=pd.read_csv(OUT/'image_inventory.csv',keep_default_na=False,dtype={'prefix':str,'date':str,'time':str});bad=[]
else:
 rows=[];bad=[]
 for p in sorted(RAW.rglob('*')):
  if p.suffix.lower() not in ['.bmp','.jpg','.png']:continue
  try:
   content=p.read_bytes()
   with Image.open(p) as im:
    mode=im.mode;a=np.array(im.convert('RGB'));w,h=im.size
   m=c_mask(a);match=re.fullmatch(r'(\d{3})_(\d{8})_(\d{6})\((\d+)\)',p.stem)
   rows.append(dict(path=str(p.relative_to(RAW)),stem=p.stem,ext=p.suffix.lower(),width=w,height=h,mode=mode,bytes=len(content),file_sha256=sha(content),pixel_sha256=sha(a.tobytes()+str(a.shape).encode()),chroma_pixels=int(m.sum()),chroma_fraction=float(m.mean()),prefix=match[1] if match else '',date=match[2] if match else '',time=match[3] if match else '',source='raw_bmp' if 'X선이물검출기' in str(p) else 'label_sets' if '라벨링 6종 세트' in str(p) else 'other'))
  except Exception as e:bad.append(dict(path=str(p),error=str(e)))
  if len(rows)%500==0:print('images',len(rows),flush=True)
 images=pd.DataFrame(rows);images.to_csv(OUT/'image_inventory.csv',index=False)
# Candidate pools are deliberately limited to source BMP and curated JPG sets.
pool=images[images.source.isin(['raw_bmp','label_sets'])]
labelrows=[];boxrows=[];invalid=[];canonical={};pixel_conflicts=[]
for lp in sorted(LABEL.glob('*.txt')):
 stem=lp.stem;candidates=pool[pool.stem==stem];raw=candidates[candidates.source=='raw_bmp']
 if len(raw):selected=raw.sort_values('path').iloc[0]
 elif len(candidates):selected=candidates.sort_values('path').iloc[0]
 else:selected=None
 boxes=[]
 for line_no,line in enumerate(lp.read_text(encoding='utf-8-sig').splitlines(),1):
  if not line.strip():continue
  try:
   vals=list(map(float,line.split()));assert len(vals)==5 and all(math.isfinite(v) for v in vals)
   cls,x,y,bw,bh=vals;assert cls==int(cls) and cls>=0 and bw>0 and bh>0
   assert 0<=x-bw/2<=x+bw/2<=1+1e-7 and 0<=y-bh/2<=y+bh/2<=1+1e-7
   boxes.append(vals)
  except Exception:invalid.append(dict(stem=stem,line=line_no,value=line))
 record=dict(stem=stem,valid_boxes=len(boxes),label_sha256=sha(lp.read_bytes()),image_found=selected is not None,candidate_files=len(candidates),raw_pixel_variants=raw.pixel_sha256.nunique(),path=selected.path if selected is not None else '')
 if selected is None:labelrows.append(record);continue
 if raw.pixel_sha256.nunique()>1:pixel_conflicts.append(stem)
 a=np.array(Image.open(RAW/selected.path).convert('RGB'));h,w=a.shape[:2];g,m=clean(a);cm=c_mask(a)
 n,cc,stats,cent=cv2.connectedComponentsWithStats(cm,8)
 rects=[[int(x),int(y),int(x+ww),int(y+hh)] for x,y,ww,hh,area in stats[1:] if area>=4 and ww>=3 and hh>=3]
 canonical[stem]=(selected.path,boxes)
 # Perceptual hash: diagnostic candidate, not automatic deduplication.
 small=cv2.resize(g,(32,32)).astype(np.float32);dct=cv2.dct(small)[:8,:8];ph=(dct>np.median(dct.flatten()[1:])).flatten();ph[0]=False
 phash=''.join(f'{v:02x}' for v in np.packbits(ph))
 match=re.fullmatch(r'(\d{3})_(\d{8})_(\d{6})\((\d+)\)',stem)
 record.update(width=w,height=h,mode=selected['mode'],pixel_sha256=selected.pixel_sha256,chroma_pixels=int(cm.sum()),chroma_fraction=float(cm.mean()),chroma10_pixels=int(c_mask(a,10).sum()),chroma60_pixels=int(c_mask(a,60).sum()),color_components=len(rects),phash=phash,prefix=match[1] if match else '',date=match[2] if match else '',time=match[3] if match else '',gray_mean=float(g.mean()),gray_std=float(g.std()))
 absboxes=[]
 for j,(cl,x,y,bw,bh) in enumerate(boxes):
  x1=(x-bw/2)*w;y1=(y-bh/2)*h;x2=(x+bw/2)*w;y2=(y+bh/2)*h;ab=[x1,y1,x2,y2];absboxes.append(ab)
  xx1=max(0,int(np.floor(x1)));xx2=min(w,int(np.ceil(x2)));yy1=max(0,int(np.floor(y1)));yy2=min(h,int(np.ceil(y2)))
  pad=5;rx1=max(0,xx1-pad);rx2=min(w,xx2+pad);ry1=max(0,yy1-pad);ry2=min(h,yy2+pad)
  outer=g[ry1:ry2,rx1:rx2];ringmask=np.ones(outer.shape,bool);ringmask[yy1-ry1:yy2-ry1,xx1-rx1:xx2-rx1]=False
  ring=outer[ringmask];inside=g[yy1:yy2,xx1:xx2]
  diff=float(ring.mean()-inside.mean()) if ring.size and inside.size else float('nan')
  cnr=abs(diff)/(float(ring.std())+1e-6) if ring.size else float('nan')
  boxrows.append(dict(stem=stem,box_id=j,cls=int(cl),x=x,y=y,w=bw,h=bh,x1=x1,y1=y1,x2=x2,y2=y2,width_px=bw*w,height_px=bh*h,area_px=bw*w*bh*h,area_fraction=bw*bh,short_side_px=min(bw*w,bh*h),aspect_ratio=bw*w/(bh*h),image_edge_distance_px=min(x1,y1,w-x2,h-y2),signed_contrast=diff,cnr_proxy=cnr,raw_color_fraction=float(cm[yy1:yy2,xx1:xx2].mean()) if inside.size else 0,processed_mask_fraction=float(m[yy1:yy2,xx1:xx2].mean()) if inside.size else 0,best_marker_iou=max((iou(ab,r) for r in rects),default=0)))
  
 record['markers_iou05_matched_gt']=sum(any(iou(ab,r)>=.5 for r in rects) for ab in absboxes)
 record['markers_iou01_matched_gt']=sum(any(iou(ab,r)>=.1 for r in rects) for ab in absboxes)
 labelrows.append(record)
labels=pd.DataFrame(labelrows);b=pd.DataFrame(boxrows);labels.to_csv(OUT/'label_image_audit.csv',index=False);b.to_csv(OUT/'boxes.csv',index=False)
(OUT/'canonical_paths.json').write_text(json.dumps({k:v[0] for k,v in canonical.items()},ensure_ascii=False,indent=2))
# Duplicate tables, retaining all paths for investigation.
dups=images[images.duplicated('pixel_sha256',keep=False)].sort_values(['pixel_sha256','path']);dups.to_csv(OUT/'pixel_duplicates.csv',index=False)
matched=labels[labels.image_found].copy();unique=matched.drop_duplicates('pixel_sha256').copy()
# Canonical near-duplicate candidates, Hamming <=4; not ground truth.
near=[];rr=list(unique.itertuples())
for i,a in enumerate(rr):
 for bb in rr[i+1:]:
  if (a.width,a.height)!=(bb.width,bb.height):continue
  dist=(int(a.phash,16)^int(bb.phash,16)).bit_count()
  if dist<=4:near.append(dict(stem_a=a.stem,stem_b=bb.stem,phash_hamming=dist,same_date=a.date==bb.date,same_prefix=a.prefix==bb.prefix))
pd.DataFrame(near,columns=['stem_a','stem_b','phash_hamming','same_date','same_prefix']).to_csv(OUT/'near_duplicate_candidates.csv',index=False)
# subset intersections: these are educational nested samples, not official splits.
sets={p.name:{x.stem for x in p.glob('*.jpg')} for p in (DS/'라벨링 6종 세트').glob('images*')}
inter=pd.DataFrame({a:{bb:len(sa&sb) for bb,sb in sets.items()} for a,sa in sets.items()});inter.to_csv(OUT/'subset_intersections.csv')
summary=dict(image_files=len(images),unreadable_images=bad,all_extensions=images.ext.value_counts().to_dict(),all_pixel_unique=int(images.pixel_sha256.nunique()),raw_bmp_files=int((images.source=='raw_bmp').sum()),raw_bmp_unique_pixels=int(images[images.source=='raw_bmp'].pixel_sha256.nunique()),raw_bmp_unique_stems=int(images[images.source=='raw_bmp'].stem.nunique()),label_files=len(labels),matched_label_files=int(labels.image_found.sum()),missing_image_stems=labels[~labels.image_found].stem.tolist(),invalid_label_rows=invalid,boxes=int(len(b)),empty_label_files=int((labels.valid_boxes==0).sum()),canonical_unique_pixels=int(matched.pixel_sha256.nunique()),canonical_pixel_conflicts=pixel_conflicts,classes=b.cls.value_counts().to_dict(),boxes_per_image=labels.valid_boxes.value_counts().sort_index().to_dict(),dimensions=matched.groupby(['width','height']).size().to_dict(),prefix_counts=matched.prefix.value_counts().to_dict(),date_counts=matched.date.value_counts().sort_index().to_dict(),marked_images=int((matched.chroma_pixels>0).sum()),marked_images_threshold10=int((matched.chroma10_pixels>0).sum()),marked_images_threshold60=int((matched.chroma60_pixels>0).sum()),component_counts=matched.color_components.value_counts().sort_index().to_dict(),marker_gt_count_disagreement=int((matched.color_components!=matched.valid_boxes).sum()),near_duplicate_pairs=len(near),same_date_near_pairs=sum(x['same_date'] for x in near),small_area_lt1024=int((b.area_px<1024).sum()),short_side_lt10=int((b.short_side_px<10-1e-6).sum()),image_edge_le10=int((b.image_edge_distance_px<=10).sum()),any_mask_overlap=int((b.processed_mask_fraction>0).sum()),marker_iou05=int((b.best_marker_iou>=.5).sum()),marker_iou01=int((b.best_marker_iou>=.1).sum()),box_stats=b[['width_px','height_px','area_px','area_fraction','short_side_px','image_edge_distance_px','cnr_proxy','processed_mask_fraction']].describe(percentiles=[.05,.25,.5,.75,.95]).to_dict(),subset_sizes={k:len(v) for k,v in sets.items()},subset_union=len(set.union(*sets.values())))
summary['dimensions']={f'{k[0]}x{k[1]}':v for k,v in summary['dimensions'].items()}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)))
print(json.dumps({k:v for k,v in summary.items() if k not in ['box_stats','date_counts','missing_image_stems','canonical_pixel_conflicts']},ensure_ascii=False,indent=2),flush=True)
# Publication-style explanatory plots (English labels for portability).
plt.rcParams.update({'font.size':10})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
axs[0,0].hist(b.area_px,bins=35,color='#287f8e');axs[0,0].set(xlabel='Bounding-box area (original pixels)',ylabel='Object count',title='Object size')
axs[0,1].scatter(b.x,b.y,s=8,alpha=.3,c='#be5a3c');axs[0,1].set(xlim=(0,1),ylim=(1,0),xlabel='Normalized center x',ylabel='Normalized center y',title='Label centers')
counts=labels.valid_boxes.value_counts().sort_index();axs[1,0].bar(counts.index,counts.values,color='#526fba');axs[1,0].set(xlabel='Objects per label file',ylabel='Image count',title='Annotation count')
axs[1,1].hist(b.short_side_px,bins=30,color='#725d98');axs[1,1].set(xlabel='Short side (original pixels)',ylabel='Object count',title='Small-object sensitivity')
fig.savefig(OUT/'dataset_overview.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
counts=matched.groupby(['date','prefix']).size().unstack(fill_value=0);counts.plot.bar(stacked=True,ax=axs[0]);axs[0].set(title='Labeled images by date / filename prefix',ylabel='Count');axs[0].tick_params(axis='x',rotation=90,labelsize=7)
axs[1].hist(matched.chroma_fraction*100,bins=30,color='#a55367');axs[1].set(title='Color-marked pixels in labeled images',xlabel='Chromatic pixels (%)',ylabel='Image count')
fig.savefig(OUT/'collection_and_marks.png',dpi=160);plt.close(fig)
# Examples selected to cover regular, minimum object, count mismatch, and no-color data.
selection=[matched.iloc[0].stem,b.loc[b.area_px.idxmin(),'stem']]
selection+=matched[matched.color_components!=matched.valid_boxes].stem.head(2).tolist()
selection+=matched[matched.chroma_pixels==0].stem.head(1).tolist()
selection+=matched.tail(1).stem.tolist();selection=list(dict.fromkeys(selection))[:6]
fig,axs=plt.subplots(len(selection),3,figsize=(12,3.3*len(selection)),squeeze=False,layout='constrained')
for row,stem in enumerate(selection):
 path,bs=canonical[stem];a=np.array(Image.open(RAW/path).convert('RGB'));g,m=clean(a)
 for ax in axs[row]:ax.axis('off')
 axs[row,0].imshow(a);axs[row,0].set_title(stem+'\nOriginal pixels',fontsize=9)
 axs[row,1].imshow(m,cmap='gray',vmin=0,vmax=1);axs[row,1].set_title('Color mask + 1-pixel dilation',fontsize=9)
 axs[row,2].imshow(g,cmap='gray',vmin=0,vmax=255);axs[row,2].set_title('Illustrative inpainting + TXT boxes',fontsize=9)
 for _,x,y,bw,bh in bs:
  h,w=g.shape;axs[row,2].add_patch(Rectangle(((x-bw/2)*w,(y-bh/2)*h),bw*w,bh*h,fill=False,edgecolor='#00ffbb',lw=1))
fig.savefig(OUT/'annotation_examples.png',dpi=160);plt.close(fig)
(OUT/'example_stems.json').write_text(json.dumps(selection,indent=2))
print('Audit and figures complete',flush=True)
