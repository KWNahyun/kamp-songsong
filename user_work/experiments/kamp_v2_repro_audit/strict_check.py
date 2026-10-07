from pathlib import Path
import sys,torch,json,traceback
R=Path('/home/viplab/contest');E=Path(__file__).parent;sys.path[:0]=[str(R/'models/D-FINE'),str(R/'experiments/kamp_ablation_v3')]
from src.core import YAMLConfig
from loss_patch import install
install(0,True);torch.set_num_threads(4);torch.manual_seed(20260929);torch.use_deterministic_algorithms(True)
cfg=YAMLConfig(str(R/'experiments/kamp_v2_mal/configs/dfine_M_seed20260929.yml'));m=cfg.model.cuda().train();m.load_state_dict(torch.load(R/'experiments/kamp_pilot_v1/dfine_s_init.pth',map_location='cpu',weights_only=False)['model']);im,t=next(iter(cfg.train_dataloader));im=im.cuda();t=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in a.items()} for a in t]
try:
 loss=sum(cfg.criterion(m(im,t),t).values());loss.backward();result={'raised':False,'strict_backward_completed':True}
except RuntimeError as e:
 result={'raised':True,'error':str(e),'traceback':traceback.format_exc(),'scope':'diagnostic actual D-FINE training forward/backward; no optimizer update'}
(E/'strict_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
