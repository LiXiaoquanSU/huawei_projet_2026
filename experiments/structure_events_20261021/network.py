"""Local event statistics and raw-frequency lag relations; no person-level labels."""
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from model import StructuredModel,Residual2D

class EventEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.Conv1d(291,128,1),nn.GroupNorm(8,128),nn.SiLU())
        self.local=nn.ModuleList([nn.Sequential(nn.Conv1d(128,128,k,padding=k//2,groups=128),nn.Conv1d(128,128,1),nn.GroupNorm(8,128),nn.SiLU()) for k in (3,9,21)])
        self.events=nn.Conv1d(384,32,1)
        self.project=nn.Sequential(nn.Linear(32*5*4+384*3,256),nn.LayerNorm(256),nn.SiLU())
    def forward(self,a,p,t):
        a=a.reshape(-1,3,32,4,150).mean(3)
        signal=torch.cat([a,p],1).reshape(-1,288,150)
        x=self.input(torch.cat([signal,t],1))
        z=torch.cat([layer(x) for layer in self.local],1)
        e=self.events(z).sigmoid().reshape(-1,32,5,30)
        m=e.mean(-1);s=((e-m[:,:,:,None]).square().mean(-1)+1e-5).sqrt()
        mx=e.amax(-1);change=(e[:,:,:,1:]-e[:,:,:,:-1]).abs().mean(-1)
        mean=z.mean(-1);std=((z-mean[:,:,None]).square().mean(-1)+1e-5).sqrt()
        return self.project(torch.cat([m.flatten(1),s.flatten(1),mx.flatten(1),change.flatten(1),mean,std,z.amax(-1)],1))

class LagRelations(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(Residual2D(27,32),Residual2D(32,64),Residual2D(64,96))
        self.project=nn.Sequential(nn.Linear(96*3,256),nn.LayerNorm(256),nn.SiLU())
    def matrices(self,t):
        amp=t[:,384:768].reshape(-1,3,32,4,150).mean(3)
        phase=t[:,768:960].reshape(-1,6,32,150)
        v=torch.cat([amp,phase],1);v=v-v.mean(-1,keepdim=True)
        v=v/(v.square().mean(-1,keepdim=True)+1e-4).sqrt()
        clock=t[:,960];maps=[]
        for lag in (0,4,16):
            if lag:
                a,b=v[:,:,:,:-lag],v[:,:,:,lag:]
                valid=((clock[:,lag:]-clock[:,:-lag])<(.04*lag+.01)).to(v.dtype)
            else:a=b=v;valid=torch.ones_like(clock)
            denom=valid.sum(-1).clamp_min(1)
            maps.append(((a*valid[:,None,None,:])@b.transpose(-1,-2)/denom[:,None,None,None]).clamp(-1,1))
        return torch.cat(maps,1)
    def forward(self,t):
        x=self.layers(self.matrices(t));m=x.mean((2,3));s=((x-m[:,:,None,None]).square().mean((2,3))+1e-5).sqrt()
        return self.project(torch.cat([m,s,x.amax((2,3))],1))

class ResidualEvents(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original;self.events=EventEncoder();self.gate=nn.Parameter(torch.full((256,),-2.))
    def forward(self,a,p,t):
        return self.original(a,p,t)+self.gate.sigmoid()*self.events(a,p,t)

class EventRelationModel(StructuredModel):
    def __init__(self,variant):
        super().__init__('spectral','physical');self.variant=variant
        if variant in ('events','combined'):self.temporal=EventEncoder()
        if variant=='event_residual':self.temporal=ResidualEvents(self.temporal)
        if variant in ('relations','combined'):
            self.relations=LagRelations();self.relation_gate=nn.Parameter(torch.full((256,),-1.5))
    def refine_latent(self,z,t,summary):
        if self.variant in ('relations','combined'):z=z+self.relation_gate.sigmoid()*self.relations(t)
        return z

def build(variant):return StructuredModel('spectral','physical') if variant=='baseline' else EventRelationModel(variant)
