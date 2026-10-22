"""Exchange local event evidence with temporal/frequency context before pooling."""
import importlib.util,math,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from model import StructuredModel,AttentionBlock
spec=importlib.util.spec_from_file_location('events_reference',ROOT/'experiments/structure_events_20261021/network.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def pool(x,dim):
    m=x.mean(dim);s=((x-m.unsqueeze(dim)).square().mean(dim)+1e-5).sqrt()
    return torch.cat([m,s,x.amax(dim)],-1)

def event_stats(e):
    m=e.mean(-1);s=((e-m[...,None]).square().mean(-1)+1e-5).sqrt()
    return torch.cat([m,s,e.amax(-1),(e[...,1:]-e[...,:-1]).abs().mean(-1)],-1)

def time_features(t):
    clock=t.reshape(-1,3,5,30).mean(-1).transpose(1,2)
    angle=clock[:,:,:1]*clock.new_tensor([.5,1.,2.])[None,None,:]*(2*math.pi)
    return torch.cat([clock,angle.sin(),angle.cos()],-1)

class LocalEvents(old.EventEncoder):
    def local_features(self,a,p,t):
        a=a.reshape(-1,3,32,4,150).mean(3);signal=torch.cat([a,p],1).reshape(-1,288,150)
        x=self.input(torch.cat([signal,t],1));z=torch.cat([layer(x) for layer in self.local],1)
        e=self.events(z).sigmoid().reshape(-1,32,5,30)
        # [B,5,128] shared event representation per chronological segment.
        tokens=event_stats(e.transpose(1,2)).reshape(-1,5,128)
        global_features=pool(z.transpose(1,2),1)
        return tokens,global_features

class ContextEvents(LocalEvents):
    def __init__(self):
        super().__init__();self.segment=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.SiLU());self.clock=nn.Linear(9,128)
        self.context=nn.Sequential(AttentionBlock(128),AttentionBlock(128))
        self.project=nn.Sequential(nn.Linear(1152+384,256),nn.LayerNorm(256),nn.SiLU())
    def forward(self,a,p,t):
        tokens,global_features=self.local_features(a,p,t);tokens=self.context(self.segment(tokens)+self.clock(time_features(t)))
        return self.project(torch.cat([global_features,pool(tokens,1)],1))

class CrossExchange(nn.Module):
    def __init__(self):
        super().__init__();self.qnorm=nn.LayerNorm(256);self.knorm=nn.LayerNorm(256);self.q=nn.Linear(256,256);self.kv=nn.Linear(256,512);self.out=nn.Linear(256,256)
    def forward(self,local,global_tokens):
        b,n,c=local.shape;m=global_tokens.shape[1];q=self.q(self.qnorm(local)).reshape(b,n,8,32).transpose(1,2)
        kv=self.kv(self.knorm(global_tokens)).reshape(b,m,2,8,32).permute(2,0,3,1,4);k,v=kv[0],kv[1]
        weights=(q@k.transpose(-1,-2)/math.sqrt(32)).softmax(-1)
        return local+self.out((weights@v).transpose(1,2).reshape(b,n,c))

class CrossEvents(LocalEvents):
    def __init__(self,original):
        super().__init__();self.global_time=original;del self.global_time.project
        self.segment=nn.Sequential(nn.Linear(128,256),nn.LayerNorm(256),nn.SiLU());self.clock=nn.Linear(9,256)
        self.exchange=CrossExchange();self.context=AttentionBlock(256)
        self.project=nn.Sequential(nn.Linear(1152+768,256),nn.LayerNorm(256),nn.SiLU())
    def forward(self,a,p,t):
        tokens,global_features=self.local_features(a,p,t)
        small=a.reshape(-1,3,32,4,150).mean(3);dynamic=torch.cat([small,p],1).reshape(-1,288,150)
        x=self.global_time.input(dynamic).transpose(1,2);clock=t.reshape(-1,3,75,2).mean(-1).transpose(1,2)
        angle=clock[:,:,:1]*self.global_time.frequencies[None,None,:]*(2*math.pi)
        x=x+self.global_time.time(torch.cat([angle.sin(),angle.cos(),clock[:,:,1:]],-1));x=self.global_time.blocks(x)
        tokens=self.segment(tokens)+self.clock(time_features(t));tokens=self.context(self.exchange(tokens,x))
        return self.project(torch.cat([global_features,pool(tokens,1)],1))

class BandEvents(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Sequential(nn.Conv1d(75,64,1),nn.GroupNorm(8,64),nn.SiLU())
        self.local=nn.ModuleList([nn.Sequential(nn.Conv1d(64,64,k,padding=k//2,groups=64),nn.Conv1d(64,64,1),nn.GroupNorm(8,64),nn.SiLU()) for k in (3,9,21)])
        self.events=nn.Conv1d(192,16,1);self.segment=nn.Sequential(nn.Linear(64,128),nn.LayerNorm(128),nn.SiLU());self.clock=nn.Linear(9,128);self.band=nn.Linear(1,128)
        self.context=nn.Sequential(AttentionBlock(128),AttentionBlock(128));self.project=nn.Sequential(nn.Linear(576+384,256),nn.LayerNorm(256),nn.SiLU())
        self.register_buffer('band_coordinates',torch.tensor([-.75,-.25,.25,.75]).reshape(1,4,1,1))
    def forward(self,a,p,t):
        a=a.reshape(-1,3,32,4,150).mean(3);v=torch.cat([a,p],1);b=v.shape[0]
        v=v.reshape(b,9,4,8,150).permute(0,2,1,3,4).reshape(-1,72,150);timing=t[:,None].expand(-1,4,-1,-1).reshape(-1,3,150)
        x=self.input(torch.cat([v,timing],1));z=torch.cat([layer(x) for layer in self.local],1)
        e=self.events(z).sigmoid().reshape(b,4,16,5,30).transpose(2,3)
        tokens=self.segment(event_stats(e))+self.clock(time_features(t))[:,None]+self.band(self.band_coordinates)
        tokens=self.context(tokens.reshape(b,20,128))
        z=z.reshape(b,4,192,150).permute(0,1,3,2).reshape(b,600,192)
        return self.project(torch.cat([pool(z,1),pool(tokens,1)],1))

class GlobalLocalModel(StructuredModel):
    def __init__(self,variant):
        super().__init__('spectral','physical');self.variant=variant
        if variant=='context_events':self.temporal=ContextEvents()
        elif variant=='cross_events':self.temporal=CrossEvents(self.temporal)
        elif variant=='band_events':self.temporal=BandEvents()
        else:raise ValueError(variant)

def build(variant):
    if variant in ('baseline','events'):return old.build(variant)
    return GlobalLocalModel(variant)
