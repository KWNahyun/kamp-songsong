from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];D=R/'dataset';report={};hashes={};checks=[]
for split in ['train','val','test']:
 c=json.loads((D/'annotations'/f'{split}.json').read_text());ann={a['image_id']:a for a in c['annotations']}
 for im in c['images']:
  p=D/'images'/split/im['file_name'];a=np.asarray(Image.open(p));assert a.shape==(im['height'],im['width'],3)
  assert np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,1],a[:,:,2])
  h=hashlib.sha256(a.tobytes()).hexdigest();assert h not in hashes;hashes[h]=split
  lines=(D/'labels'/split/(p.stem+'.txt')).read_text().splitlines()
  assert len(lines)==int(im['id'] in ann)
  if lines:
   cl,cx,cy,w,ht=map(float,lines[0].split());assert cl==0 and 0<w<=1 and 0<ht<=1
   box=np.array([(cx-w/2)*im['width'],(cy-ht/2)*im['height'],w*im['width'],ht*im['height']]);assert np.max(abs(box-np.array(ann[im['id']]['bbox'])))<1e-5
   assert box[0]>=0 and box[1]>=0 and box[0]+box[2]<=im['width']+1e-5 and box[1]+box[3]<=im['height']+1e-5
  checks.append({'file':str(p.relative_to(D)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 report[split]={'images':len(c['images']),'boxes':len(ann),'negative':len(c['images'])-len(ann)}
assert len(hashes)==254
(D/'validation.json').write_text(json.dumps({'passed':True,'checks':['254 images decoded','all RGB channels identical','YOLO coordinates in bounds','YOLO and COCO boxes match','all image pixel hashes unique','positive/negative label presence consistent'],'splits':report},indent=2))
(D/'image_checksums.json').write_text(json.dumps(checks,indent=2));print(report)
