"""Temporal covariance images: no absolute amplitude or receiver identity."""
import torch
from torch import nn
from torch.nn import functional as F

class CovarianceFeatures(nn.Module):
    def kernel(self,x):
        x=x-x.mean(-1,keepdim=True)
        x=x/(x.square().mean((-1,-2),keepdim=True)+1e-5).sqrt()
        k=x.transpose(-1,-2)@x/x.shape[-2]
        return k/(k.square().mean((-1,-2),keepdim=True)+1e-5).sqrt()
    def pool(self,k):
        m=k.mean(1);v=((k-m[:,None]).square().mean(1)+1e-5).sqrt()
        return torch.stack([m,v],1)
    def forward(self,t):
        b=t.shape[0]
        a=t[:,384:768].reshape(b,3,32,4,30,5).mean((3,5))
        p=t[:,768:960].reshape(b,3,64,30,5).mean(-1)
        out=[]
        for x in (a,p):
            out.append(self.pool(self.kernel(x)))
            dx=F.pad(x[...,1:]-x[...,:-1],(0,1))
            out.append(self.pool(self.kernel(dx)))
        return torch.cat(out,1)

class CovarianceNet(nn.Module):
    def __init__(self):
        super().__init__();self.body=nn.Sequential(nn.Conv2d(8,24,3,padding=1),nn.GroupNorm(6,24),nn.SiLU(),nn.Conv2d(24,48,3,stride=2,padding=1),nn.GroupNorm(8,48),nn.SiLU(),nn.Conv2d(48,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Dropout(.2),nn.Linear(128,5))
    def forward(self,x):
        x=self.body(x);m=x.mean((2,3));v=((x-m[:,:,None,None]).square().mean((2,3))+1e-5).sqrt()
        return self.head(torch.cat([m,v],1)).log_softmax(-1)
