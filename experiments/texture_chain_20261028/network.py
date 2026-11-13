"""Preserve local frequency/time textures before symmetric receiver aggregation."""
import torch
from torch import nn
from torch.nn import functional as F

def spread(x,dim):return ((x-x.mean(dim,keepdim=True)).square().mean(dim)+1e-5).sqrt()

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.main=nn.Sequential(nn.Conv2d(cin,cout,3,stride=2,padding=1),nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(8,cout))
        self.skip=nn.Conv2d(cin,cout,1,stride=2)
    def forward(self,x):return F.silu(self.main(x)+self.skip(x))

class Texture(nn.Module):
    def __init__(self,channels):
        super().__init__();self.conv=nn.Sequential(Block(channels,64),Block(64,128),Block(128,256))
        self.project=nn.Sequential(nn.Linear(2048,384),nn.LayerNorm(384),nn.SiLU())
    def forward(self,x):
        x=self.conv(x);return self.project(torch.cat([x.mean(-1),spread(x,-1)],1).flatten(1))

class EvidenceChain(nn.Module):
    def __init__(self,variant='full'):
        super().__init__();assert variant=='full'
        self.amp=Texture(3);self.phase=Texture(2)
        self.summary=nn.Sequential(nn.Linear(13,128),nn.LayerNorm(128),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(1664,768),nn.LayerNorm(768),nn.SiLU(),nn.Dropout(.3),nn.Linear(768,256),nn.SiLU(),nn.Linear(256,5))
    def tasks(self,t,s):
        b=t.shape[0];a=t[:,:768].reshape(-1,6,128,150)
        d=a[:,3:].reshape(-1,1,128,150);smooth=F.avg_pool2d(F.pad(d,(2,2,0,0),mode='replicate'),(1,5),stride=1)
        amp=torch.cat([a[:,:3].reshape(-1,1,128,150),d,d-smooth],1)
        amp=self.amp(F.avg_pool2d(amp,(4,3))).reshape(b,3,384)
        p=t[:,768:960].reshape(-1,3,2,32,150);reverse=torch.stack([p[:,:,0],-p[:,:,1]],2)
        p=self.phase(F.avg_pool2d(torch.cat([p,reverse],1).reshape(-1,2,32,150),(1,3))).reshape(b,6,384)
        scale=s[:,:12].reshape(-1,4,3);summary=self.summary(torch.cat([scale.mean(-1),spread(scale,-1),s[:,12:17]],-1))
        x=torch.cat([amp.mean(1),spread(amp,1),p.mean(1),spread(p,1),summary],-1)
        z=self.head(x).log_softmax(-1)
        return dict(logits=z,modulation=summary,local_probabilities=z.exp()[:,:,None].expand(-1,-1,2))
    def forward(self,t,s):return self.tasks(t,s)['logits']
