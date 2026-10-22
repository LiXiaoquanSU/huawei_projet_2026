"""Aligned antenna-pair evidence and continuous-time attention ablations."""
import importlib.util
import math
import sys
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('global_local_reference',ROOT/'experiments/global_local_20261021/network.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class PairEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(18,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,256))
        self.position=nn.Linear(5,256)
        self.query=old.CrossExchange()
        self.scale=nn.Parameter(torch.tensor(-1.0))
        # Identity is retained; no assumption of antenna permutation invariance.
        self.register_buffer('pair_id',torch.eye(3).reshape(1,3,1,1,3))
        self.register_buffer('frequency',torch.linspace(-1,1,16).reshape(1,1,16,1,1))

    def tokens(self,a,p,t):
        b=a.shape[0]
        a=a.reshape(b,3,32,4,150).mean(3)
        signals=[]
        for k,(u,v) in enumerate(((0,1),(0,2),(1,2))):
            x,y=a[:,u],a[:,v]
            signals.append(torch.stack((x,y,p[:,2*k],p[:,2*k+1],x-y,x*y),1))
        # B,pair,feature,frequency,segment,time -> B,pair,frequency,segment,feature,time
        x=torch.stack(signals,1).reshape(b,3,6,16,2,5,30).mean(4).permute(0,1,3,4,2,5)
        mean=x.mean(-1);std=((x-mean[...,None]).square().mean(-1)+1e-5).sqrt()
        change=(x[...,1:]-x[...,:-1]).abs().mean(-1)
        z=self.encode(torch.cat((mean,std,change),-1))
        clock=t[:,0].reshape(b,5,30).mean(-1).reshape(b,1,1,5,1)
        pos=torch.cat((self.pair_id.expand(b,3,16,5,3),self.frequency.expand(b,3,16,5,1),clock.expand(b,3,16,5,1)),-1)
        return (z+self.position(pos)).reshape(b,240,256)

    def forward(self,q,a,p,t):
        # Residual gate allows the original temporal evidence to remain useful.
        attended=self.query(q,self.tokens(a,p,t))
        return q+self.scale.sigmoid()*(attended-q)

class TimedAttention(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(256);self.qkv=nn.Linear(256,768);self.out=nn.Linear(256,256)
        self.log_rates=nn.Parameter(torch.tensor([.1,.2,.5,1.,2.,4.,8.,16.]).log())
        self.ff=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1024),nn.GELU(),nn.Dropout(.1),nn.Linear(1024,256),nn.Dropout(.1))
    def forward(self,x,clock):
        b,n,c=x.shape;q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,8,32).permute(2,0,3,1,4)
        elapsed=(clock[:,:,None]-clock[:,None,:]).abs()
        bias=-elapsed[:,None]*self.log_rates.exp()[None,:,None,None]
        att=(q@k.transpose(-1,-2)/math.sqrt(32)+bias).softmax(-1)
        x=x+self.out((att@v).transpose(1,2).reshape(b,n,c))
        return x+self.ff(x)

class EvidenceTemporal(old.CrossEvents):
    def __init__(self,original,aligned,timed):
        super().__init__(original);self.aligned=aligned;self.timed=timed
        if aligned:self.pairs=PairEvidence()
        if timed:
            # Same 75 tokens, actual elapsed-time inductive bias, no fabricated high-rate observations.
            self.global_time.blocks=nn.ModuleList([TimedAttention() for _ in range(3)])
    def forward(self,a,p,t):
        tokens,global_features=self.local_features(a,p,t)
        small=a.reshape(-1,3,32,4,150).mean(3)
        dynamic=torch.cat((small,p),1).reshape(-1,288,150)
        x=self.global_time.input(dynamic).transpose(1,2)
        clock=t.reshape(-1,3,75,2).mean(-1).transpose(1,2)
        angle=clock[:,:,:1]*self.global_time.frequencies[None,None,:]*(2*math.pi)
        x=x+self.global_time.time(torch.cat((angle.sin(),angle.cos(),clock[:,:,1:]),-1))
        if self.timed:
            for block in self.global_time.blocks:x=block(x,clock[:,:,0])
        else:x=self.global_time.blocks(x)
        tokens=self.segment(tokens)+self.clock(old.time_features(t))
        if self.aligned:tokens=self.pairs(tokens,a,p,t)
        tokens=self.context(self.exchange(tokens,x))
        return self.project(torch.cat((global_features,old.pool(tokens,1)),1))

def build(variant):
    if variant=='cross_events':return old.build(variant)
    model=old.StructuredModel('spectral','physical')
    assert variant in ('aligned_events','timed_events','aligned_timed')
    model.temporal=EvidenceTemporal(model.temporal,variant!='timed_events',variant!='aligned_events')
    return model
