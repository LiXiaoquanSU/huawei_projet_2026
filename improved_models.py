"""Domain-robust candidates; fixed transforms are also part of ONNX inference."""
import torch
from torch import nn
from roomsense_model import CSIPreprocessor, TemporalBlock


def standardize(x):
    x=x-x.mean(dim=-1,keepdim=True)
    return (x/torch.sqrt(x.square().mean(dim=-1,keepdim=True)+1e-5)).clamp(-6,6)


class ImprovedModel(nn.Module):
    def __init__(self, mode='whiten', width=96):
        super().__init__()
        self.mode=mode
        self.preprocess=CSIPreprocessor()
        if mode=='spectrum':
            self.register_buffer('frequencies',torch.tensor([.5,1.,1.5,2.,3.,4.,6.,8.,12.,16.,24.,32.,48.,64.,80.]))
            self.head=nn.Sequential(nn.Linear(107,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.35),
                                    nn.Linear(128,64),nn.SiLU(),nn.Dropout(.25),nn.Linear(64,5))
        else:
            self.stem=nn.Sequential(nn.Conv1d(195,width,5,stride=3,padding=2),nn.GroupNorm(8,width),nn.SiLU())
            self.blocks=nn.Sequential(*(TemporalBlock(width,d) for d in [1,2,4,8,16]))
            self.head=nn.Sequential(nn.Linear(width*3,96),nn.SiLU(),nn.Dropout(.4),nn.Linear(96,5))

    def signal(self,t):
        x=t[:,:96]
        if self.mode=='decorrelated':
            x=x.reshape(-1,3,32,600)
            x=x-x.mean(dim=2,keepdim=True)
            x=x.reshape(-1,96,600)
        x=standardize(x)
        if self.training:
            # Independent frequency masking/gain, plus time reversal; labels unchanged.
            keep=(torch.rand(x.shape[0],96,1,device=x.device)>.15).to(x.dtype)
            gain=torch.exp(torch.randn(x.shape[0],96,1,device=x.device)*.25)
            x=x*keep*gain
            reverse=(torch.rand(x.shape[0],1,1,device=x.device)<.5)
            x=torch.where(reverse,x.flip(-1),x)
        return x

    def spectral_features(self,x,t):
        batch=x.shape[0]
        # ATI channel was aligned to the receiving frame by CSIPreprocessor.
        dt=.005*(torch.exp(t[:,192])-1)
        time=torch.cumsum(dt,dim=-1)
        angle=6.283185307179586*time[:,:,None]*self.frequencies[None,None,:]
        re=torch.bmm(x,torch.cos(angle))/600
        im=torch.bmm(x,torch.sin(angle))/600
        power=(re.square()+im.square()).reshape(batch,3,32,15).mean(dim=2)
        power=power/(power.sum(dim=-1,keepdim=True)+1e-6)
        spec=power.reshape(batch,45)
        entropy=-(power*torch.log(power+1e-6)).sum(dim=-1)/2.7080502
        correlations=[]
        for lag in [1,2,4,8,16,32,64]:
            c=(x[:,:,:-lag]*x[:,:,lag:]).mean(dim=-1).reshape(batch,3,32)
            correlations += [c.mean(dim=-1), torch.sqrt((c-c.mean(dim=-1,keepdim=True)).square().mean(dim=-1)+1e-5)]
        concentrations=[]
        for z in [x[:,:32],x[:,32:64],x[:,64:],x]:
            cov=torch.bmm(z,z.transpose(1,2))/600
            diagonal=cov.diagonal(dim1=1,dim2=2).sum(dim=-1).clamp_min(1e-3)
            concentrations.append((cov.square().sum(dim=(1,2))/diagonal.square())[:,None])
            concentrations.append(((torch.bmm(cov,cov)*cov.transpose(1,2)).sum(dim=(1,2))/diagonal.pow(3))[:,None])
        skew=x.pow(3).mean(dim=-1).reshape(batch,3,32).abs().mean(dim=-1)
        kurt=x.pow(4).mean(dim=-1).reshape(batch,3,32).mean(dim=-1)/10
        rx=x.reshape(batch,3,32,600)
        cross=torch.stack([(rx[:,a]*rx[:,b]).mean(dim=(1,2)) for a,b in [(0,1),(0,2),(1,2)]],dim=1)
        return torch.cat([spec,entropy,*correlations,*concentrations,skew,kurt,cross],dim=1)

    def forward_features(self,t,s):
        x=self.signal(t)
        if self.mode=='spectrum':
            return self.head(self.spectral_features(x,t))
        # Difference features are re-derived from normalized signals; no static summary.
        diff=torch.cat([torch.zeros_like(x[:,:,:1]),x[:,:,1:]-x[:,:,:-1]],dim=-1)
        x=torch.cat([x,diff,t[:,192:]],dim=1)
        x=self.blocks(self.stem(x))
        mean=x.mean(dim=-1)
        std=torch.sqrt((x-mean[:,:,None]).square().mean(dim=-1)+1e-5)
        return self.head(torch.cat([mean,std,x.max(dim=-1).values],dim=1))

    def forward(self,frames,intervals):
        return self.forward_features(*self.preprocess(frames,intervals))
