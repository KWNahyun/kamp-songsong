from pathlib import Path
import csv,json,hashlib,collections,shutil
import cv2,numpy as np,yaml
R=Path('/home/viplab/contest'); E=Path(__file__).parent; S=R/'kamp_xray_v2'; D=E/'data640'; P=R/'experiments/kamp_pilot_v1'
for d in ['logs','configs','runs']: (E/d).mkdir(parents=True,exist_ok=True)
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
# Reuse only verified pre-training initialization, never a fitted KAMP checkpoint.
base=yaml.safe_load((P/'dfine_s.yml').read_text())
for split,key in [('train','train_dataloader'),('val','val_dataloader')]:
 base[key]['dataset']['img_folder']=str(D/'images'/split); base[key]['dataset']['ann_file']=str(D/'annotations'/f'{split}.json')
base['print_freq']=10
jobs=[]
for seed in [20260929,20260930,20261001]:
 for variant in ['yolov8s','yolov8s_p2','dfine_s','dfine_s_p2']:
  name=f'{variant}_seed{seed}'; p2=variant.endswith('p2'); family='yolo' if variant.startswith('yolo') else 'dfine'
  cmd=[str(R/'.detector-venv/bin/python')]
  if family=='yolo': cmd += [str(E/'train_yolo.py'),'--seed',str(seed),'--name',name]+(['--p2'] if p2 else [])
  else:
   cfg=json.loads(json.dumps(base)); cfg['output_dir']=str(E/'runs'/name)
   if p2:
    cfg['HGNetv2']['return_idx']=[0,1,2,3]; cfg['DFINETransformer']={'feat_channels':[256]*4,'feat_strides':[8,16,32,4],'num_levels':4,'num_points':[3,6,3,3]}
   cp=E/'configs'/f'{name}.yml';cp.write_text(yaml.safe_dump(cfg))
   cmd += [str(P/'train_dfine.py')]+(['--p2'] if p2 else [])+['-c',str(cp),'-t',str(P/f'{variant}_init.pth'),'--device','cuda:0','--seed',str(seed)]
  jobs.append({'name':name,'family':family,'seed':seed,'command':cmd,'status':'pending'})
(E/'run_manifest.json').write_text(json.dumps({'jobs':jobs,'sha256':{},'epochs':30,'input':640,'batch':8,'test_used':False,'initialization':'existing COCO-derived init, no KAMP-trained weights'},indent=2))
print(audit['splits']); print('Prepared',len(jobs),'jobs')
