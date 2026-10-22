"""Independent single-backbone classifiers; no original experts or ordinal heads."""
import importlib.util,sys,math
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from model import StructuredPreprocessor

def pool(x,dim):
    mean=x.mean(dim);std=((x-mean.unsqueeze(dim)).square().mean(dim)+1e-5).sqrt()
    return torch.cat([mean,std,x.amax(dim)],-1)

def signals(t):
    a=t[:,:768].reshape(-1,6,32,4,150).mean(3)
    p=t[:,768:960].reshape(-1,6,32,150)
    return torch.cat([a,p],1)

class Attention(nn.Module):
    def __init__(self,width):
        super().__init__();self.width=width;self.n1=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.n2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,width*3),nn.GELU(),nn.Dropout(.2),nn.Linear(width*3,width),nn.Dropout(.2))
    def forward(self,x):
        b,n,c=x.shape;q,k,v=self.qkv(self.n1(x)).reshape(b,n,3,4,c//4).permute(2,0,3,1,4).unbind(0)
        z=((q@k.transpose(-1,-2)/math.sqrt(self.width//4)).softmax(-1)@v).transpose(1,2).reshape(b,n,c)
        x=x+self.out(z);return x+self.ff(self.n2(x))

class BandPatch(nn.Module):
    def __init__(self):
        super().__init__();self.patch=nn.Linear(48*15,96);self.position=nn.Parameter(torch.zeros(1,10,96));self.clock=nn.Linear(3,96)
        self.blocks=nn.Sequential(*[Attention(96) for _ in range(3)])
        self.band=nn.Sequential(nn.Linear(288,128),nn.LayerNorm(128),nn.GELU())
        self.project=nn.Sequential(nn.Linear(384,256),nn.LayerNorm(256),nn.GELU())
    def forward(self,x,timing):
        b=x.shape[0]
        x=x.reshape(b,12,8,4,10,15).permute(0,2,4,1,3,5).reshape(b*8,10,720)
        clock=timing.reshape(b,3,10,15).mean(-1).transpose(1,2)
        clock=clock[:,None].expand(-1,8,-1,-1).reshape(b*8,10,3)
        x=self.blocks(self.patch(x)+self.position+self.clock(clock))
        bands=self.band(pool(x,1)).reshape(b,8,128)
        return self.project(pool(bands,1))

class MixerBlock(nn.Module):
    def __init__(self):
        super().__init__();self.n1=nn.LayerNorm(128);self.n2=nn.LayerNorm(128)
        self.time=nn.Sequential(nn.Linear(30,60),nn.GELU(),nn.Dropout(.2),nn.Linear(60,30))
        self.feature=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,128))
    def forward(self,x):
        x=x+self.time(self.n1(x).transpose(1,2)).transpose(1,2)
        return x+self.feature(self.n2(x))

class Mixer(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(384*5+3,128);self.blocks=nn.Sequential(*[MixerBlock() for _ in range(4)])
        self.project=nn.Sequential(nn.Linear(384,256),nn.LayerNorm(256),nn.GELU())
    def forward(self,x,timing):
        b=x.shape[0];x=x.reshape(b,384,30,5).permute(0,2,1,3).reshape(b,30,1920)
        clock=timing.reshape(b,3,30,5).mean(-1).transpose(1,2)
        return self.project(pool(self.blocks(self.input(torch.cat([x,clock],-1))),1))

class Recurrent(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Sequential(nn.Linear(387,128),nn.LayerNorm(128),nn.GELU())
        self.gru=nn.GRU(128,128,num_layers=2,batch_first=True,bidirectional=True,dropout=.2)
        self.project=nn.Sequential(nn.Linear(768,256),nn.LayerNorm(256),nn.GELU())
    def forward(self,x,timing):
        x=torch.cat([x.flatten(1,2),timing],1).transpose(1,2)
        x,_=self.gru(self.input(x));return self.project(pool(x,1))

class FreshModel(nn.Module):
    def __init__(self,variant):
        super().__init__();self.preprocess=StructuredPreprocessor(128)
        self.backbone={'band_patch':BandPatch,'mixer':Mixer,'bigru':Recurrent}[variant]()
        self.summary=nn.Sequential(nn.LayerNorm(17),nn.Linear(17,32),nn.GELU())
        self.head=nn.Sequential(nn.Linear(288,128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,5))
    def forward_features(self,t,summary):
        x=signals(t)
        if self.training:
            keep=(torch.rand(x.shape[0],1,32,1,device=x.device)>.05).to(x.dtype)
            x=(x+torch.randn_like(x)*.02)*keep
        z=self.backbone(x,t[:,960:])
        return self.head(torch.cat([z,self.summary(summary)],1)).log_softmax(-1)
    def forward_tasks(self,t,summary):return {'logits':self.forward_features(t,summary)}
    def forward(self,frames,intervals):return self.forward_features(*self.preprocess(frames,intervals))

def build(variant):
    if variant=='context_control':
        spec=importlib.util.spec_from_file_location('context_reference',ROOT/'experiments/global_local_20261021/network.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.build('context_events')
    return FreshModel(variant)
