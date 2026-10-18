"""Relative-antenna phase with per-window phase alignment, real-valued ONNX ops."""
import torch
from torch import nn
from roomsense_model import TemporalBlock
from improved_models import standardize


class RelativePreprocessor(nn.Module):
    def forward(self,frames,intervals):
        b=frames.shape[0]
        r,i=frames[:,:,0,:,:,0],frames[:,:,0,:,:,1]
        power=r.square()+i.square()
        valid=(power>1e-8).to(frames.dtype)
        a=.5*torch.log1p(power)
        count=valid.reshape(b,600,3,32,32).sum(-1).clamp_min(1)
        amp=(a*valid).reshape(b,600,3,32,32).sum(-1)/count
        # Remove per-frame common gain across frequencies and per-bin static response.
        amp=amp-amp.mean(dim=-1,keepdim=True)
        amp=standardize(amp.reshape(b,600,96).transpose(1,2))
        r=r/torch.sqrt(power+1e-4);i=i/torch.sqrt(power+1e-4)
        phase=[]
        for u,v in [(0,1),(0,2),(1,2)]:
            mask=valid[:,:,u]*valid[:,:,v]
            denom=mask.reshape(b,600,32,32).sum(-1).clamp_min(1)
            c=((r[:,:,u]*r[:,:,v]+i[:,:,u]*i[:,:,v])*mask).reshape(b,600,32,32).sum(-1)/denom
            s=((i[:,:,u]*r[:,:,v]-r[:,:,u]*i[:,:,v])*mask).reshape(b,600,32,32).sum(-1)/denom
            mc,ms=c.mean(dim=1,keepdim=True),s.mean(dim=1,keepdim=True)
            norm=torch.sqrt(mc.square()+ms.square()+1e-4)
            aligned_c=(c*mc+s*ms)/norm
            aligned_s=(s*mc-c*ms)/norm
            phase.extend([aligned_c,aligned_s])
        phase=torch.stack(phase,dim=2).reshape(b,600,192).transpose(1,2)
        phase=standardize(phase)
        dt=intervals[:,:599].clamp_min(0)/1e6
        time=torch.stack([torch.log1p(dt/.005).clamp_max(8),(dt>.02).to(frames.dtype),(dt>0).to(frames.dtype)],dim=1)
        time=torch.cat([torch.zeros_like(time[:,:,:1]),time],dim=-1)
        return torch.cat([amp,phase,time],dim=1),torch.zeros_like(intervals[:,:17])


class PhaseModel(nn.Module):
    def __init__(self,mode='phase'):
        super().__init__()
        self.mode=mode;self.preprocess=RelativePreprocessor()
        self.register_buffer('frequencies',torch.tensor([.5,1.,1.5,2.,3.,4.,6.,8.,12.,16.,24.,32.,48.,64.,80.]))
        if mode=='phase_spectrum':
            self.head=nn.Sequential(nn.Linear(311,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.4),
                                    nn.Linear(256,128),nn.SiLU(),nn.Dropout(.3),nn.Linear(128,5))
        else:
            self.stem=nn.Sequential(nn.Conv1d(291,128,5,stride=3,padding=2),nn.GroupNorm(8,128),nn.SiLU())
            self.blocks=nn.Sequential(*(TemporalBlock(128,d) for d in [1,2,4,8,16]))
            self.head=nn.Sequential(nn.Linear(384,128),nn.SiLU(),nn.Dropout(.4),nn.Linear(128,5))

    def spectral(self,x,t):
        b=x.shape[0]
        dt=.005*(torch.exp(t[:,288])-1)
        time=torch.cumsum(dt,dim=-1)
        angle=6.283185307179586*time[:,:,None]*self.frequencies[None,None,:]
        re=torch.bmm(x,torch.cos(angle))/600;im=torch.bmm(x,torch.sin(angle))/600
        power=(re.square()+im.square()).reshape(b,9,32,15).mean(2)
        power=power/(power.sum(-1,keepdim=True)+1e-6)
        terms=[power.reshape(b,135),-(power*torch.log(power+1e-6)).sum(-1)/2.7080502]
        for lag in [1,2,4,8,16,32,64]:
            c=(x[:,:,:-lag]*x[:,:,lag:]).mean(-1).reshape(b,9,32)
            terms.extend([c.mean(-1),torch.sqrt((c-c.mean(-1,keepdim=True)).square().mean(-1)+1e-5)])
        for j in range(10):
            z=x[:,j*32:(j+1)*32] if j<9 else x
            cov=torch.bmm(z,z.transpose(1,2))/600
            diagonal=cov.diagonal(dim1=1,dim2=2).sum(-1).clamp_min(1e-3)
            terms.extend([(cov.square().sum((1,2))/diagonal.square())[:,None],
                          ((torch.bmm(cov,cov)*cov.transpose(1,2)).sum((1,2))/diagonal.pow(3))[:,None]])
        terms.extend([x.pow(3).mean(-1).reshape(b,9,32).abs().mean(-1),
                      x.pow(4).mean(-1).reshape(b,9,32).mean(-1)/10])
        rx=x[:,:96].reshape(b,3,32,600)
        terms.append(torch.stack([(rx[:,u]*rx[:,v]).mean((1,2)) for u,v in [(0,1),(0,2),(1,2)]],dim=1))
        return torch.cat(terms,dim=1)

    def forward_features(self,t,s):
        x=t[:,:288]
        if self.training:
            keep=(torch.rand(x.shape[0],9,32,1,device=x.device)>.1).to(x.dtype).reshape(-1,288,1)
            x=x*keep
            if self.mode!='phase_spectrum':
                reverse=torch.rand(x.shape[0],1,1,device=x.device)<.5
                x=torch.where(reverse,x.flip(-1),x)
        if self.mode=='phase_spectrum':return self.head(self.spectral(x,t))
        x=self.blocks(self.stem(torch.cat([x,t[:,288:]],dim=1)))
        mean=x.mean(-1);std=torch.sqrt((x-mean[:,:,None]).square().mean(-1)+1e-5)
        return self.head(torch.cat([mean,std,x.max(-1).values],dim=1))

    def forward(self,frames,intervals):return self.forward_features(*self.preprocess(frames,intervals))
