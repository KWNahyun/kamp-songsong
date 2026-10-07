from pathlib import Path
import csv,json,hashlib,collections,shutil
import cv2,numpy as np,yaml
import argparse
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
S=a.dataset.resolve();D=a.output.resolve();E=D
if D.exists():raise FileExistsError(D)
D.mkdir(parents=True)
rows=list(csv.DictReader((S/'split_manifest.csv').open())); groups=collections.defaultdict(set)
for r in rows: groups[r['group']].add(r['split'])
assert all(len(x)==1 for x in groups.values())
audit={'test_pixels_or_labels_read':False,'splits':{},'transforms':{},'pixel_duplicates_train_val':[]}; seen={}
for split in ['train','val']:
 for sub in ['images','labels']: (D/sub/split).mkdir(parents=True,exist_ok=True)
 coco={'images':[],'annotations':[],'categories':[{'id':0,'name':'defect'}]}; count=0
 for r in [x for x in rows if x['split']==split]:
  src=S/r['image_path']; im=cv2.imread(str(src)); assert im is not None
  h,w=im.shape[:2]; assert (w,h)==(int(r['width']),int(r['height']))
  ph=hashlib.sha256(im.tobytes()).hexdigest()
  if ph in seen and seen[ph]!=split: audit['pixel_duplicates_train_val'].append(r['image_id'])
  seen[ph]=split
  nw,nh=round(w*640/max(w,h)),round(h*640/max(w,h)); x,y=(640-nw)//2,(640-nh)//2
  canvas=np.full((640,640,3),114,np.uint8); canvas[y:y+nh,x:x+nw]=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
  cv2.imwrite(str(D/'images'/split/(r['image_id']+'.png')),canvas)
  iid=len(coco['images'])+1; coco['images'].append({'id':iid,'file_name':r['image_id']+'.png','width':640,'height':640})
  labs=[]
  for line in (S/r['label_path']).read_text().splitlines():
   c,cx,cy,bw,bh=map(float,line.split()); assert c==0 and bw>0 and bh>0
   a,b=(cx-bw/2)*w,(cy-bh/2)*h; assert a>=-1e-3 and b>=-1e-3 and a+bw*w<=w+1e-3 and b+bh*h<=h+1e-3
   box=[a*nw/w+x,b*nh/h+y,bw*nw,bh*nh]; count+=1
   coco['annotations'].append({'id':count,'image_id':iid,'category_id':0,'bbox':box,'area':box[2]*box[3],'iscrowd':0})
   labs.append(f'0 {(cx*nw+x)/640:.10f} {(cy*nh+y)/640:.10f} {bw*nw/640:.10f} {bh*nh/640:.10f}')
  assert len(labs)==int(r['n_defects'])
  (D/'labels'/split/(r['image_id']+'.txt')).write_text('\n'.join(labs)+'\n')
  audit['transforms'][r['image_id']]={'split':split,'width':w,'height':h,'sx':nw/w,'sy':nh/h,'pad_x':x,'pad_y':y}
 audit['splits'][split]={'images':len(coco['images']),'boxes':count}
 (D/'annotations').mkdir(exist_ok=True); (D/'annotations'/f'{split}.json').write_text(json.dumps(coco))
assert not audit['pixel_duplicates_train_val']
(D/'data.yaml').write_text(yaml.safe_dump({'path':str(D),'train':'images/train','val':'images/val','names':{0:'defect'}}))
(E/'data_audit.json').write_text(json.dumps(audit,indent=2)); (E/'preflight.json').write_text(json.dumps({'passed':True,'scope':'data geometry, labels, group split, train/val exact duplicates'}))
