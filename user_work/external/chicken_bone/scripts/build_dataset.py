"""Deterministic single-energy proxy dataset. Masks only create labels, never inputs."""
from pathlib import Path
import csv,json,hashlib,io
import numpy as np
import tifffile
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];SRC=R/'source/Submission/Dataset #1/40kV_40W_100ms_10avg';OUT=R/'dataset'
rows=[]
for split in ['train','val','test']:
 for cl in ['NoBone','RibBone']:
  for r in csv.DictReader((SRC/split/f'{cl}.csv').open()):
   c=int(r['Chicken_ID']);r.update(source_split=split,source_class=cl,split='train' if c<=9 else 'val' if c<=11 else 'test')
   r['image']=SRC/split/cl/f"{int(r['ID']):03d}.tiff";r['mask']=SRC/split/f'{cl}_segm'/r['image'].name
   r['stem']=f"c{c:02d}_{split}_{cl}_{int(r['ID']):03d}";rows.append(r)
# Global display window learned exclusively from training images, no mask use.
samples=[]
for r in rows:
 if r['split']=='train':
  a=tifffile.imread(r['image']);assert a.ndim==2 and np.isfinite(a).all()
  samples.append(a[::8,::8].ravel())
lo,hi=map(float,np.percentile(np.concatenate(samples),[0.1,99.9]));assert hi>lo
manifest=[];ann_by_split={};stats={};original_hashes={};output_hashes={}
for split in ['train','val','test']:
 for kind in ['images','labels']:(OUT/kind/split).mkdir(parents=True,exist_ok=True)
 coco={'images':[],'annotations':[],'categories':[{'id':0,'name':'Defect'}]};shorts=[];boxes=0;neg=0
 for r in [x for x in rows if x['split']==split]:
  a=tifffile.imread(r['image']);m=tifffile.imread(r['mask']);assert m.shape==(2,*a.shape),(r['mask'],m.shape)
  bone=m[1]>0;positive=r['source_class']=='RibBone';assert bool(bone.any())==positive
  # Each source sample has one physical bone ID: union disconnected mask islands.
  H,W=a.shape;w=352;h=round(H*w/W);sx,sy=w/W,h/H
  gray=np.rint(255*(1-np.clip((a-lo)/(hi-lo),0,1))).astype(np.uint8)
  im=Image.fromarray(gray).resize((w,h),Image.Resampling.LANCZOS).convert('RGB')
  dest=OUT/'images'/split/f"{r['stem']}.png";im.save(dest)
  iid=len(coco['images']);coco['images'].append({'id':iid,'file_name':dest.name,'width':w,'height':h})
  bbox=None;label=''
  if positive:
   ys,xs=np.nonzero(bone);x1,x2=float(xs.min()*sx),float((xs.max()+1)*sx);y1,y2=float(ys.min()*sy),float((ys.max()+1)*sy)
   bw,bh=x2-x1,y2-y1;bbox=[x1,y1,bw,bh];shorts.append(min(bw,bh));boxes+=1
   label=f'0 {(x1+bw/2)/w:.10f} {(y1+bh/2)/h:.10f} {bw/w:.10f} {bh/h:.10f}\n'
   coco['annotations'].append({'id':len(coco['annotations']),'image_id':iid,'category_id':0,'bbox':bbox,'area':bw*bh,'iscrowd':0})
  else:neg+=1
  (OUT/'labels'/split/f"{r['stem']}.txt").write_text(label)
  oh=hashlib.sha256(a.tobytes()).hexdigest();ph=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
  original_hashes.setdefault(oh,set()).add(split);output_hashes.setdefault(ph,set()).add(split)
  manifest.append({k:v for k,v in r.items() if k not in ['image','mask']}|{'source_image':str(r['image'].relative_to(R)),'source_mask':str(r['mask'].relative_to(R)),'output_image':str(dest.relative_to(OUT)),'width':w,'height':h,'bbox':bbox,'source_pixel_sha256':oh,'output_pixel_sha256':ph})
 (OUT/'annotations').mkdir(exist_ok=True);(OUT/'annotations'/f'{split}.json').write_text(json.dumps(coco,indent=2))
 stats[split]={'images':len(coco['images']),'positive':boxes,'negative':neg,'boxes':boxes,'short_side_min_median_max':list(map(float,[min(shorts),np.median(shorts),max(shorts)])),'chicken_ids':sorted({int(r['Chicken_ID']) for r in rows if r['split']==split}),'bone_ids':sorted({int(r['Bone_ID']) for r in rows if r['split']==split and int(r['Bone_ID'])})}
for key in ['chicken_ids','bone_ids']:
 for a,b in [('train','val'),('train','test'),('val','test')]:assert not set(stats[a][key])&set(stats[b][key])
assert all(len(v)==1 for v in original_hashes.values()),'source duplicate leakage'
assert all(len(v)==1 for v in output_hashes.values()),'output duplicate leakage'
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'data.yaml').write_text(f'path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Defect\n')
(OUT/'classes.txt').write_text('Defect\n')
summary={'source':'https://zenodo.org/records/10579608','license':'CC-BY-4.0','condition':'40kV_40W_100ms_10avg','resize':'width 352, aspect preserved, Lanczos; bbox uses exact per-axis ratios','window':{'lower':lo,'upper':hi,'method':'training-only pooled pixel 0.1/99.9 percentiles, sample every 8 pixels; invert attenuation'},'mask_rule':'channel 1 union -> one bbox per RibBone image; channel 0 is chicken, not target','stats':stats,'source_unique_pixels':len(original_hashes),'output_unique_pixels':len(output_hashes),'cross_split_source_duplicates':0,'cross_split_output_duplicates':0}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
# Separate visual audit: input stays annotation-free.
review=R/'review';review.mkdir(exist_ok=True)
for split in ['train','val','test']:
 selected=[r for r in manifest if r['split']==split and r['bbox']]
 selected=sorted(selected,key=lambda r:min(r['bbox'][2:]))
 selected=[selected[i] for i in np.linspace(0,len(selected)-1,5,dtype=int)]
 selected += [next(r for r in manifest if r['split']==split and r['bbox'] is None)]
 canvas=Image.new('RGB',(576*3,480*2),'white');d=ImageDraw.Draw(canvas)
 for i,r in enumerate(selected):
  im=Image.open(OUT/r['output_image']);q=ImageDraw.Draw(im)
  if r['bbox']:
   x,y,w,h=r['bbox'];q.rectangle((x,y,x+w,y+h),outline='red',width=2)
  im=im.resize((576,round(im.height*576/im.width)),Image.Resampling.NEAREST)
  ox=(i%3)*576;oy=(i//3)*480;canvas.paste(im,(ox,oy));d.text((ox+8,oy+460),r['stem'],fill='black')
 canvas.save(review/f'{split}_contact_sheet.jpg')
