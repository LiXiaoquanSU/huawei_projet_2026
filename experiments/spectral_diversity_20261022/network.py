"""Compact spectral diversity summaries instead of learning room-shaped relation maps."""
import importlib.util,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('context_reference',ROOT/'experiments/global_local_20261021/network.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class SpectralDiversity(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Sequential(nn.Linear(180,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,256),nn.LayerNorm(256),nn.SiLU())

    @staticmethod
    def moments(x):
        x=x-x.mean(-1,keepdim=True)
        c=x@x.transpose(-1,-2)/x.shape[-1]
        n=x.shape[-2];eye=torch.eye(n,device=x.device,dtype=x.dtype)
        power=(c*eye).sum((-1,-2))
        c=c/(power[...,None,None]+1e-5)
        c2=c@c
        # Tr(C^k)=sum(lambda_i^k). No eigendecomposition is needed for ONNX.
        m2=c.square().sum((-1,-2));m3=(c2*c).sum((-1,-2));m4=c2.square().sum((-1,-2))
        trace=(c*eye).sum((-1,-2))
        return torch.stack([torch.log1p(power),m2,m3,m4,(trace.square()/(m2+1e-4)).clamp_max(n)/n],-1)

    def forward(self,t):
        a=t[:,384:768].reshape(-1,3,8,16,150).mean(3).reshape(-1,24,150)
        p=t[:,768:960].reshape(-1,6,8,4,150).mean(3).reshape(-1,48,150)
        result=[]
        for x in (a,p):
            # Raw fluctuations, slower averages, and faster frame differences.
            streams=(x,x.reshape(-1,x.shape[1],50,3).mean(-1),x[:,:,1:]-x[:,:,:-1])
            for z in streams:
                result.append(self.moments(z))
                # Five nonoverlapping windows; discard at most4 residual points.
                length=z.shape[-1]//5
                local=z[:,:,:length*5].reshape(-1,z.shape[1],5,length).transpose(1,2)
                result.append(self.moments(local).flatten(1))
        return self.project(torch.cat(result,1))

class DiversityModel(old.GlobalLocalModel):
    def __init__(self,variant):
        super().__init__('context_events' if variant=='diversity_context' else 'cross_events')
        self.diversity=SpectralDiversity();self.diversity_gate=nn.Parameter(torch.full((256,),-2.))
    def refine_latent(self,z,t,summary):
        return z+self.diversity_gate.sigmoid()*self.diversity(t)

def build(variant):
    if variant in ('context_events','cross_events'):return old.build(variant)
    if variant not in ('diversity_context','diversity_cross'):raise ValueError(variant)
    return DiversityModel(variant)
