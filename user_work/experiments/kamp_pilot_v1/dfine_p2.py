"""Minimal additive P2: preserve pretrained P3/P4/P5 paths and append stride-4.

P2 is projected C2 concatenated with upsampled P3 and fused by 1x1/3x3 conv.
No extra bottom-up feedback into existing levels. Register before config.model.
"""
import torch
from torch import nn
from torch.nn import functional as F

def install():
    from src.zoo.dfine.hybrid_encoder import HybridEncoder
    from src.zoo.dfine.dfine_decoder import DFINETransformer
    original_init=HybridEncoder.__init__
    original_forward=HybridEncoder.forward

    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        d=self.hidden_dim
        self.p2_project=nn.Sequential(nn.Conv2d(64,d,1,bias=False),nn.BatchNorm2d(d),nn.SiLU())
        self.p2_fuse=nn.Sequential(nn.Conv2d(2*d,d,1,bias=False),nn.BatchNorm2d(d),nn.SiLU(),nn.Conv2d(d,d,3,padding=1,bias=False),nn.BatchNorm2d(d),nn.SiLU())

    def forward(self,feats):
        if len(feats)==3:
            return original_forward(self,feats)
        assert len(feats)==4
        existing=original_forward(self,feats[1:])
        shallow=self.p2_project(feats[0])
        p2=self.p2_fuse(torch.cat([shallow,F.interpolate(existing[0],size=shallow.shape[-2:],mode='nearest')],1))
        return existing+[p2]

    def anchors(self,spatial_shapes=None,grid_size=.05,dtype=torch.float32,device='cpu'):
        if spatial_shapes is None:
            h,w=self.eval_spatial_size
            spatial_shapes=[[int(h/s),int(w/s)] for s in self.feat_strides]
        levels=[]
        for stride,(h,w) in zip(self.feat_strides,spatial_shapes):
            yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
            xy=(torch.stack([xx,yy],-1).unsqueeze(0)+.5)/torch.tensor([w,h],dtype=dtype)
            # Original P3/P4/P5 priors .05/.10/.20; P2 must be .025, not .40.
            wh=torch.ones_like(xy)*grid_size*(stride/8)
            levels.append(torch.cat([xy,wh],-1).reshape(1,h*w,4))
        a=torch.cat(levels,1).to(device)
        valid=((a>self.eps)*(a<1-self.eps)).all(-1,keepdim=True)
        logits=torch.log(a/(1-a))
        return torch.where(valid,logits,torch.inf),valid

    HybridEncoder.__init__=init
    HybridEncoder.forward=forward
    DFINETransformer._generate_anchors=anchors
