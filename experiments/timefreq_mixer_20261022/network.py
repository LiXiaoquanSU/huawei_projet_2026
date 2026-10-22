"""Independent local time-frequency classifiers with preserved amplitude polyphase bins."""
import importlib.util,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from model import StructuredPreprocessor

def pack(t,fine):
    a=t[:,:768].reshape(-1,6,32,4,150)
    if fine:a=a.permute(0,1,3,2,4).reshape(-1,24,32,150)
    else:a=a.mean(3)
    return torch.cat([a,t[:,768:960].reshape(-1,6,32,150)],1)

class LocalBlock(nn.Module):
    def __init__(self,width,dilation,axial):
        super().__init__();self.axial=axial
        if axial:
            self.frequency=nn.Conv2d(width,width,(5,1),padding=(2,0),groups=width)
            self.time=nn.Conv2d(width,width,(1,7),padding=(0,3*dilation),dilation=(1,dilation),groups=width)
        else:self.spatial=nn.Conv2d(width,width,(5,7),padding=(2,3*dilation),dilation=(1,dilation),groups=width)
        self.norm=nn.LayerNorm(width)
        self.mix=nn.Sequential(nn.Linear(width,width*3),nn.GELU(),nn.Dropout(.2),nn.Linear(width*3,width))
        self.scale=nn.Parameter(torch.full((width,),.01))
    def forward(self,x):
        y=self.time(self.frequency(x)) if self.axial else self.spatial(x)
        y=self.mix(self.norm(y.permute(0,2,3,1)))*self.scale
        return x+y.permute(0,3,1,2)

class TimeFrequencyModel(nn.Module):
    def __init__(self,variant):
        super().__init__();self.fine=variant!='conv_coarse';self.preprocess=StructuredPreprocessor(128)
        width=96;channels=30 if self.fine else 12
        self.stem=nn.Sequential(nn.Conv2d(channels+3,width,(2,5),stride=(2,5)),nn.GroupNorm(8,width),nn.GELU())
        self.blocks=nn.Sequential(*[LocalBlock(width,d,variant=='axial_fine') for d in (1,2,4,1,2,4)])
        self.statistics=nn.Sequential(nn.LayerNorm(17),nn.Linear(17,32),nn.GELU())
        self.head=nn.Sequential(nn.Linear(width*3+32,128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,5))
    def forward_features(self,t,summary):
        x=pack(t,self.fine)
        if self.training:
            keep=(torch.rand(x.shape[0],1,32,1,device=x.device)>.05).to(x.dtype)
            x=(x+torch.randn_like(x)*.02)*keep
        timing=t[:,960:,None,:].expand(-1,-1,32,-1)
        x=self.blocks(self.stem(torch.cat([x,timing],1)))
        mean=x.mean((2,3));std=((x-mean[:,:,None,None]).square().mean((2,3))+1e-5).sqrt()
        z=torch.cat([mean,std,x.amax((2,3)),self.statistics(summary)],1)
        return self.head(z).log_softmax(-1)
    def forward_tasks(self,t,summary):return {'logits':self.forward_features(t,summary)}
    def forward(self,frames,intervals):return self.forward_features(*self.preprocess(frames,intervals))

def build(variant):
    if variant=='mixer':
        spec=importlib.util.spec_from_file_location('mixer_reference',ROOT/'experiments/fresh_backbones_20261022/network.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.build('mixer')
    if variant not in ('conv_coarse','conv_fine','axial_fine'):raise ValueError(variant)
    return TimeFrequencyModel(variant)
