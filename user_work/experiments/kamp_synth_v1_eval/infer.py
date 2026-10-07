from pathlib import Path
import os,sys,json,time,hashlib,argparse
import numpy as np,pandas as pd,cv2,torch
from torch.utils.data import Dataset,DataLoader
from torchvision.ops import nms
R=Path('/home/viplab/contest');E=Path(__file__).parent;S=R/'data/synthetic/kamp_synth_test_v1'
sys.path[:0]=[str(R/'experiments/kamp_pilot_v1'),str(R/'models/D-FINE'),str(R/'experiments/kamp_v2_uq')]
class Images(Dataset):
 def __init__(self,df):self.df=df.reset_index(drop=True)
 def __len__(self):return len(self.df)
 def __getitem__(self,k):
  r=self.df.iloc[k];a=cv2.imread(str(S/'images'/r['split']/(r.image_id+'.png')),cv2.IMREAD_GRAYSCALE);h,w=a.shape;nw,nh=round(w*640/max(w,h)),round(h*640/max(w,h));px,py=(640-nw)//2,(640-nh)//2;c=np.full((640,640,3),114,np.uint8);c[py:py+nh,px:px+nw]=cv2.cvtColor(cv2.resize(a,(nw,nh),interpolation=cv2.INTER_LINEAR),cv2.COLOR_GRAY2BGR)
  return torch.from_numpy(c).permute(2,0,1),np.array([w,h,nw/w,nh/h,px,py]),k

def main():
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--limit',type=int,default=0);p.add_argument('--batch',type=int,default=8);p.add_argument('--tag',default='runs');a=p.parse_args();j=next(x for x in json.loads((E/'protocol.json').read_text())['jobs'] if x['name']==a.name);out=E/a.tag/a.name;out.mkdir(parents=True,exist_ok=True)
 assert hashlib.sha256(Path(j['checkpoint']).read_bytes()).hexdigest()==j['sha256'];torch.set_num_threads(3);cv2.setNumThreads(0)
 df=pd.read_csv(S/'images.csv')
 if a.limit:df=df.head(a.limit)
 df=df.reset_index(drop=True);df.to_csv(out/'images.csv',index=False)
 if a.name.startswith('yolo'):
  from ultralytics import YOLO
  model=YOLO(j['checkpoint']);cfg=None
 else:
  if j['variant']=='dfine_s_p2':import dfine_p2;dfine_p2.install()
  if j['variant']=='dfine_mal_UQ':
   from selector_patch import install_model
   install_model('unary')
  from src.core import YAMLConfig
  cfg=YAMLConfig(j['config']);model=cfg.model;ck=torch.load(j['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model']);model.cuda().eval()
 loader=DataLoader(Images(df),batch_size=a.batch,num_workers=2,pin_memory=True,shuffle=False)
 start=time.time();processed=0;files=[]
 for step,(x,ts,ids) in enumerate(loader):
  part=out/f'batch_{step:05d}.npz'
  if part.exists():processed+=len(ids);files.append(part.name);continue
  x=x.cuda(non_blocking=True).float()/255
  with torch.inference_mode():
   if cfg is None:
    yy=model.predict(x,imgsz=640,device=0,conf=.001,iou=.7,max_det=300,rect=False,verbose=False);qs=[(z.boxes.xyxy.cpu().numpy(),z.boxes.conf.cpu().numpy(),None,None) for z in yy]
   else:
    yy=model(x);b=yy['pred_boxes'];b=torch.cat((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)*640;ss=yy['pred_logits'].sigmoid().squeeze(-1);base=yy.get('pred_logits_base',yy['pred_logits']).sigmoid().squeeze(-1);q=yy.get('pred_quality_logits');q=q.sigmoid().squeeze(-1) if q is not None else torch.full_like(ss,float('nan'))
    qs=[(b[k].cpu().numpy(),ss[k].cpu().numpy(),base[k].cpu().numpy(),q[k].cpu().numpy()) for k in range(len(ids))]
  boxes=np.zeros((len(ids),300,4),np.float32);scores=np.zeros((len(ids),300),np.float32);base=np.zeros_like(scores);quality=np.full_like(scores,np.nan);valid=np.zeros_like(scores,bool);keeps=np.zeros_like(scores,bool);basekeep=np.zeros_like(scores,bool)
  for k,(bb,sc,bs,qu) in enumerate(qs):
   w,h,sx,sy,px,py=ts[k].numpy();bb=bb.astype(np.float64);bb[:,[0,2]]=np.clip((bb[:,[0,2]]-px)/sx,0,w);bb[:,[1,3]]=np.clip((bb[:,[1,3]]-py)/sy,0,h);v=(bb[:,2]>bb[:,0])&(bb[:,3]>bb[:,1])
   n=len(bb);boxes[k,:n]=bb;scores[k,:n]=sc;base[k,:n]=sc if bs is None else bs
   if qu is not None:quality[k,:n]=qu
   valid[k,:n]=v
   for target,sv in [(keeps,scores[k,:n]),(basekeep,base[k,:n])]:
    idx=np.where(v & (sv>=.001))[0]
    if len(idx): kk=nms(torch.tensor(bb[idx],dtype=torch.float64),torch.tensor(sv[idx],dtype=torch.float64),.7).numpy();target[k,idx[kk]]=True
  np.savez_compressed(part,index=ids.numpy(),boxes=boxes,scores=scores,base=base,quality=quality,valid=valid,keep=keeps,basekeep=basekeep)
  files.append(part.name);processed+=len(ids)
  if step%20==0:
   torch.cuda.synchronize();status=dict(name=a.name,processed=processed,total=len(df),seconds=time.time()-start,images_per_second=processed/max(time.time()-start,.001),peak_vram_gib=torch.cuda.max_memory_allocated()/2**30);(out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 torch.cuda.synchronize();result=dict(name=a.name,images=processed,seconds=time.time()-start,images_per_second=processed/(time.time()-start),peak_vram_gib=torch.cuda.max_memory_allocated()/2**30,checkpoint_sha256=j['sha256'],batch=a.batch,complete=True,YOLO_preNMS_saved=False,DFINE_regular_queries_saved=cfg is not None,postprocessing='score>=0.001, clip original coordinates, NMS0.7',files=files)
 (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
if __name__=='__main__':main()
