from pathlib import Path
import json,hashlib,collections
import pandas as pd,numpy as np
from PIL import Image
R=Path('/home/viplab/contest/data/synthetic/kamp_synth_test_v1');O=Path('/home/viplab/contest/experiments/kamp_synth_v1_audit');I=pd.read_csv(R/'images.csv');B=pd.read_csv(R/'objects.csv');M=pd.read_csv('/home/viplab/contest/kamp_xray_v2/split_manifest.csv');groups={k:g for k,g in B.groupby('image_id',sort=False)}
errors=[];records=[];hashes=collections.defaultdict(list)
for n,row in enumerate(I.itertuples()):
 p=R/'images'/row.split/(row.image_id+'.png');l=R/'labels'/row.split/(row.image_id+'.txt');im=Image.open(p);im.load();a=np.array(im);h=hashlib.sha256(a.tobytes()).hexdigest();hashes[h].append(row.image_id)
 if im.mode!='L' or a.dtype!=np.uint8 or im.size!=(row.width,row.height):errors.append([row.image_id,'image format'])
 lines=l.read_text().splitlines() if l.exists() else []
 if len(lines)!=row.n_objects:errors.append([row.image_id,'label count',len(lines),row.n_objects])
 gg=groups.get(row.image_id);expected=[] if gg is None else gg[['x1','y1','x2','y2']].to_numpy();actual=[]
 for line in lines:
  z=np.array([float(t) for t in line.split()]);
  if len(z)!=5 or not np.isfinite(z).all() or z[0]!=0:errors.append([row.image_id,'invalid label']);continue
  c,x,y,w,hb=z
  if min(x,y,w,hb)<0 or max(x,y,w,hb)>1:errors.append([row.image_id,'label bounds'])
  actual.append([(x-w/2)*row.width,(y-hb/2)*row.height,(x+w/2)*row.width,(y+hb/2)*row.height])
 if len(actual)==len(expected) and len(actual):
  # order independent match
  err=np.abs(np.array(actual)[:,None,:]-expected[None,:,:]).max(axis=2).min(axis=1).max()
  if err>0.002:errors.append([row.image_id,'csv txt mismatch',float(err)])
 records.append(dict(image_id=row.image_id,split=row.split,set=row.set,width=im.width,height=im.height,mode=im.mode,min=int(a.min()),max=int(a.max()),mean=float(a.mean()),std=float(a.std()),sha256_pixels=hashlib.sha256(a.tobytes()).hexdigest(),label_exists=l.exists(),n_labels=len(lines)))
 if n%2000==0: print(n,flush=True)
D=pd.DataFrame(records);D.to_csv(O/'image_audit.csv',index=False)
summary={'images':len(I),'objects':len(B),'sets':I['set'].nunique(),'unique_sources':I.source_image.nunique(),'split_counts':I.groupby('split').size().to_dict(),'source_split_mismatches':int((I.source_image.map(M.set_index('image_id').split)!=I.split).sum()),'duplicate_image_ids':int(I.image_id.duplicated().sum()),'modes':D['mode'].value_counts().to_dict(),'dimensions':I.groupby(['width','height']).size().to_string(),'errors':errors,'exact_duplicate_groups':sum(len(v)>1 for v in hashes.values()),'unique_pixel_images':len(hashes),'missing_label_files':int((~D.label_exists).sum()),'all_image_files':len(list((R/'images').rglob('*.png'))),'all_label_files':len(list((R/'labels').rglob('*.txt'))),'donor_source_splits':B.donor.dropna().str.split('|').str[0].map(M.set_index('image_id').split).value_counts(dropna=False).to_dict()}
(O/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(O/'pixel_duplicate_groups.json').write_text(json.dumps([v for v in hashes.values() if len(v)>1],indent=2));I.groupby(['split','axis','set']).agg(images=('image_id','size'),objects=('n_objects','sum')).to_csv(O/'condition_counts.csv');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
