"""Small evidence adapters on the unchanged original-width four-route backbone."""
import sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from model import StructuredModel

VARIANTS = ['baseline', 'capacity_control', 'static', 'slow', 'fast', 'three',
            'spectrum', 'three_spectrum']

class SeriesEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(96,64,5,padding=2), nn.GroupNorm(8,64), nn.SiLU(),
                                 nn.Conv1d(64,64,5,padding=2), nn.GroupNorm(8,64), nn.SiLU())
        self.out = nn.Sequential(nn.Linear(128,64), nn.LayerNorm(64), nn.SiLU())
    def forward(self, x):
        x=self.net(x); mean=x.mean(-1)
        std=(x.sub(mean[:,:,None]).square().mean(-1)+1e-5).sqrt()
        return self.out(torch.cat([mean,std],1))

class MotionModel(StructuredModel):
    def __init__(self, variant):
        super().__init__('spectral','physical')
        self.variant=variant
        branches = ['static','slow','fast'] if variant in ['three','three_spectrum'] else [variant]
        if variant=='three_spectrum': branches+=['spectrum']
        self.branches=branches if variant!='baseline' else []
        self.adapters=nn.ModuleDict()
        for name in self.branches:
            if name in ['slow','fast']: self.adapters[name]=SeriesEncoder()
            elif name=='spectrum': self.adapters[name]=nn.Sequential(nn.Linear(8*13*3,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU())
            else: self.adapters[name]=nn.Sequential(nn.Linear(305,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU())
        if self.branches:
            self.fuse=nn.Linear(64*len(self.branches),256)
            nn.init.normal_(self.fuse.weight,std=.001);nn.init.zeros_(self.fuse.bias)
        self.register_buffer('motion_frequencies',torch.tensor([.5,1.,1.5,2.,3.,4.,6.,8.,10.,12.,16.,18.,20.]))

    def refine_latent(self,z,t,summary):
        if not self.branches:return z
        amp=t[:,:384].reshape(-1,3,32,4,150).mean(3).reshape(-1,96,150)
        dynamic=t[:,384:768].reshape(-1,3,32,4,150).mean(3).reshape(-1,96,150)
        clock=t[:,-3,:]
        values=[]
        if any(n in self.branches for n in ['slow','fast','spectrum']):
            step=torch.cat([torch.zeros_like(clock[:,:1]),clock[:,1:]-clock[:,:-1]],1)
            segment=(step>.1).to(t.dtype).cumsum(1)
            delta=(clock[:,:,None]-clock[:,None,:]).abs()
            weights=torch.exp(-delta/.12)*(delta<.6).to(t.dtype)
            weights=weights*(segment[:,:,None]==segment[:,None,:]).to(t.dtype)
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
            slow=dynamic@weights.transpose(1,2)
            fast=dynamic-slow
        for name in self.branches:
            if name in ['static','capacity_control']:
                x=amp if name=='static' else dynamic
                mean=x.mean(-1);std=(x.sub(mean[:,:,None]).square().mean(-1)+1e-5).sqrt()
                vector=torch.cat([mean,std,x.amax(-1),summary],1)
            elif name in ['slow','fast']:
                vector=slow if name=='slow' else fast
            else:
                # A nonuniform temporal spectrum of normalized amplitude variation.
                # It is not a calibrated body-velocity profile or a count of people.
                time=clock-clock[:,:1]
                span=(time[:,-1:]).clamp_min(.1)
                angle=time[:,:,None]*self.motion_frequencies[None,None,:]*6.28318530718
                quadrature=step.clamp(.001,.1)*(step<=.1).to(t.dtype)
                spectra=[]
                for center,width in [(.5,.5),(.25,.25),(.75,.25)]:
                    win=(1-(time/span-center).abs()/width).clamp_min(0)*quadrature
                    win=win/win.sum(1,keepdim=True).clamp_min(1e-6)
                    x=dynamic-(dynamic*win[:,None]).sum(-1,keepdim=True)
                    real=x@(angle.cos()*win[:,:,None]);imag=x@(angle.sin()*win[:,:,None])
                    power=(real.square()+imag.square()).reshape(-1,3,8,4,13).mean((1,3))
                    spectra.append(torch.log1p(power).flatten(1))
                vector=torch.cat(spectra,1)
            values.append(self.adapters[name](vector))
        return z+self.fuse(torch.cat(values,1))

def fresh(variant,seed):
    torch.manual_seed(seed)
    return MotionModel(variant)

class Submission(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames,intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_features(t.float(),s.float())[0]
