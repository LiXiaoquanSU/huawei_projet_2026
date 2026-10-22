"""Cross-receiver/band relational dynamics, not person detection or source separation."""
import importlib.util,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('context_reference',ROOT/'experiments/global_local_20261021/network.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class RelationDynamics(nn.Module):
    def __init__(self,contrast):
        super().__init__();self.contrast=contrast
        self.node=nn.Sequential(nn.Linear(72*2+2,64),nn.LayerNorm(64),nn.SiLU())
        self.nodes=old.AttentionBlock(64)
        self.segment=nn.Sequential(nn.Linear(192,128),nn.LayerNorm(128),nn.SiLU())
        self.clock=nn.Linear(9,128);self.sequence=nn.Sequential(old.AttentionBlock(128),old.AttentionBlock(128))
        self.project=nn.Sequential(nn.Linear(384,256),nn.LayerNorm(256),nn.SiLU())

    @staticmethod
    def correlation(x):
        x=x-x.mean(-1,keepdim=True)
        energy=x.square().mean(-1,keepdim=True)
        x=x/(energy+1e-4).sqrt()
        return (x@x.transpose(-1,-2)/x.shape[-1]).clamp(-1,1),energy

    def forward(self,a,p,t):
        # 9 measured channels x 8 frequency regions = 72 nodes.
        a=a.reshape(-1,3,8,16,150).mean(3);p=p.reshape(-1,6,8,4,150).mean(3)
        x=torch.cat([a,p],1).reshape(-1,72,150)
        whole,_=self.correlation(x)
        local=x.reshape(-1,72,5,30).transpose(1,2)
        corr,energy=self.correlation(local)
        # Contrast is relative to this sample only; it is not an empty-room reference.
        reference=whole[:,None].expand(-1,5,-1,-1)
        relation=corr-reference if self.contrast else corr
        descriptor=torch.cat([relation,reference,torch.log1p(energy),local.mean(-1,keepdim=True)],-1)
        z=self.nodes(self.node(descriptor).reshape(-1,72,64))
        z=self.segment(old.pool(z,1)).reshape(-1,5,128)
        z=self.sequence(z+self.clock(old.time_features(t)))
        return self.project(old.pool(z,1))

class RelationContext(nn.Module):
    def __init__(self,original,contrast):
        super().__init__();self.original=original;self.relation=RelationDynamics(contrast)
        self.gate=nn.Sequential(nn.Linear(512,128),nn.SiLU(),nn.Linear(128,256))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2.)
    def forward(self,a,p,t):
        base=self.original(a,p,t);extra=self.relation(a,p,t)
        return base+self.gate(torch.cat([base,extra],1)).sigmoid()*extra

def build(variant):
    if variant in ('context_events','cross_events'):return old.build(variant)
    model=old.build('context_events')
    if variant=='relation_only':model.temporal=RelationDynamics(True)
    elif variant in ('relation_context','contrast_context'):
        model.temporal=RelationContext(model.temporal,variant=='contrast_context')
    else:raise ValueError(variant)
    return model
