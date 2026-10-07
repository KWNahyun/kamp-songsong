"""Same COCO initialization for shared layers, independently added P2 branches."""
import os,sys,copy,json
from pathlib import Path
import torch,yaml
ROOT=Path('/home/viplab/contest'); EXP=ROOT/'experiments/kamp_pilot_v1'
os.environ['YOLO_CONFIG_DIR']=str(EXP/'ultralytics_settings')
sys.path.insert(0,str(ROOT/'models/D-FINE'))
SEED=20260929

def dfine():
    from src.core import YAMLConfig
    torch.manual_seed(SEED)
    base=YAMLConfig(str(EXP/'dfine_s.yml')).model
    source=torch.load(ROOT/'models/weights/dfine_s_coco.pth',map_location='cpu',weights_only=False)
    pretrained=source['ema']['module'] if 'ema' in source else source['model']
    shared={k:v for k,v in pretrained.items() if k in base.state_dict() and base.state_dict()[k].shape==v.shape}
    base.load_state_dict(shared,strict=False)
    state=copy.deepcopy(base.state_dict())
    torch.save({'model':state},EXP/'dfine_s_init.pth')
    import dfine_p2
    dfine_p2.install()
    torch.manual_seed(SEED)
    p2=YAMLConfig(str(EXP/'dfine_s_p2.yml')).model
    dest=p2.state_dict();copied=[];expanded=[]
    for key,value in state.items():
        if key not in dest:continue
        if dest[key].shape==value.shape:
            dest[key]=value.clone();copied.append(key)
        elif any(t in key for t in ['sampling_offsets','attention_weights']):
            # Preserve old per-head sampling points; initialize only P2 positions.
            factor=2 if 'sampling_offsets' in key else 1
            old=value.reshape(8,12,factor,*value.shape[1:])
            new=dest[key].reshape(8,15,factor,*dest[key].shape[1:])
            new[:,:12]=old
            dest[key]=new.reshape_as(dest[key]);expanded.append(key)
    p2.load_state_dict(dest)
    torch.save({'model':p2.state_dict()},EXP/'dfine_s_p2_init.pth')
    assert all(torch.equal(base.state_dict()[k],p2.state_dict()[k]) for k in copied)
    # Forward + backward verifies all heads and the new P2 path are trainable.
    for name,model in [('dfine_s',base),('dfine_s_p2',p2)]:
        model.cuda().eval()
        with torch.no_grad():
            out=model(torch.zeros(1,3,640,640,device='cuda'))
        assert out['pred_boxes'].shape==(1,300,4) and torch.isfinite(out['pred_boxes']).all()
        if name.endswith('p2'):
            model.train()
            features=model.encoder(model.backbone(torch.randn(2,3,128,128,device='cuda')))
            assert [f.shape[-1] for f in features]==[16,8,4,32]
            sum(f.square().mean() for f in features).backward()
            assert model.encoder.p2_project[0].weight.grad.abs().sum()>0
            model.zero_grad()
        model.cpu()
    return {'pretrained_tensors':len(shared),'identical_shared_tensors':len(copied),'expanded_attention_tensors':expanded,'parameters_base':sum(p.numel() for p in base.parameters()),'parameters_p2':sum(p.numel() for p in p2.parameters())}

def yolo():
    import ultralytics
    from ultralytics import YOLO
    cfg=yaml.safe_load((Path(ultralytics.__file__).parent/'cfg/models/v8/yolov8.yaml').read_text())
    cfg['nc']=1;cfg['scale']='s'
    (EXP/'yolov8s_base.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    p2cfg=copy.deepcopy(cfg)
    # Keep original model.0..21 unchanged. Move Detect from 22 to 25.
    p2cfg['head']=p2cfg['head'][:-1]+[[15,1,'nn.Upsample',[None,2,'nearest']],[[-1,2],1,'Concat',[1]],[-1,3,'C2f',[128]],[[15,18,21,24],1,'Detect',['nc']]]
    (EXP/'yolov8s_p2.yaml').write_text(yaml.safe_dump(p2cfg,sort_keys=False))
    torch.manual_seed(SEED)
    base=YOLO(str(EXP/'yolov8s_base.yaml')).load(str(ROOT/'models/weights/yolov8s.pt'))
    base.save(str(EXP/'yolov8s_init.pt'))
    # Reload the saved initialization so both start from identical stored precision.
    base=YOLO(str(EXP/'yolov8s_init.pt'))
    torch.manual_seed(SEED)
    p2=YOLO(str(EXP/'yolov8s_p2.yaml'))
    dest=p2.model.state_dict();copied=[]
    for key,value in base.model.state_dict().items():
        newkey=key.replace('model.22.','model.25.') if key.startswith('model.22.') else key
        if newkey in dest and dest[newkey].shape==value.shape:
            dest[newkey]=value.clone();copied.append((key,newkey))
    p2.model.load_state_dict(dest)
    p2.save(str(EXP/'yolov8s_p2_init.pt'))
    p2=YOLO(str(EXP/'yolov8s_p2_init.pt'))
    assert all(torch.equal(base.model.state_dict()[a],p2.model.state_dict()[b]) for a,b in copied)
    assert p2.model.stride.tolist()==[8.,16.,32.,4.]
    return {'identical_shared_tensors':len(copied),'strides':p2.model.stride.tolist(),'parameters_base':sum(p.numel() for p in base.model.parameters()),'parameters_p2':sum(p.numel() for p in p2.model.parameters())}

if __name__=='__main__':
    torch.set_num_threads(4)
    details={'dfine':dfine(),'yolo':yolo()}
    (EXP/'initialization_audit.json').write_text(json.dumps(details,indent=2))
    print(details)
