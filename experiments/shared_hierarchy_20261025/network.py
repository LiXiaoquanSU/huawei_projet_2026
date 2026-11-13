"""Shared receiver/pair encoders with multiscale time analysis and count hierarchy.

No graph message passing, receiver IDs, old-model predictions or protection rules.
Input features retain the audited original CSI preprocessing.
"""
import torch
from torch import nn
from torch.nn import functional as F

class SequenceEncoder(nn.Module):
    def __init__(self,channels):
        super().__init__();layers=[]
        for width in (24,48,96):
            layers += [nn.Conv2d(channels,width,3,2,1,bias=False),nn.GroupNorm(8,width),nn.SiLU()];channels=width
        self.spatial=nn.Sequential(*layers)
        self.project=nn.Conv1d(192,96,1)
        self.temporal=nn.ModuleList([nn.Sequential(nn.Conv1d(96,96,3,padding=d,dilation=d,groups=96),nn.Conv1d(96,96,1),nn.GroupNorm(8,96),nn.SiLU()) for d in (1,2,4)])
        self.output=nn.Sequential(nn.Linear(288,128),nn.LayerNorm(128),nn.SiLU())
    def forward(self,x):
        x=self.spatial(x);m=x.mean(2);v=((x-m[:,:,None]).square().mean(2)+1e-5).sqrt()
        x=self.project(torch.cat([m,v],1))
        for block in self.temporal:x=x+block(x)
        m=x.mean(-1);v=((x-m[:,:,None]).square().mean(-1)+1e-5).sqrt()
        return self.output(torch.cat([m,v,x.amax(-1)],1))

class SetPool(nn.Module):
    def __init__(self):
        super().__init__();self.attention=nn.Linear(128,1)
    def forward(self,x):
        m=x.mean(1);v=((x-m[:,None]).square().mean(1)+1e-5).sqrt()
        a=(x*self.attention(x).softmax(1)).sum(1)
        return torch.cat([m,v,a],-1)

class SharedHierarchy(nn.Module):
    def __init__(self):
        super().__init__();self.nodes=SequenceEncoder(4);self.pairs=SequenceEncoder(2)
        self.node_pool=SetPool();self.pair_pool=SetPool()
        self.background=nn.Sequential(nn.Linear(2,32),nn.SiLU(),nn.Linear(32,128),nn.Tanh())
        self.fusion=nn.Sequential(nn.Linear(768+9,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.2),nn.Linear(256,128),nn.LayerNorm(128),nn.SiLU())
        self.presence=nn.Linear(128,1);self.count=nn.Linear(128,4);self.ordinal=nn.Linear(128,3)
    def tasks(self,t,s):
        b=t.shape[0];a=t[:,:768].reshape(-1,6,128,150)
        dyn=a[:,3:].reshape(-1,1,128,150)
        slow=F.avg_pool2d(F.pad(dyn,(2,2,0,0),mode='replicate'),(1,5),stride=1)
        slower=F.avg_pool2d(F.pad(dyn,(8,8,0,0),mode='replicate'),(1,17),stride=1)
        # Four evidence scales, not supervised activity categories.
        node=torch.cat([a[:,:3].reshape(-1,1,128,150),dyn-slow,slow-slower,slower],1)
        h=self.nodes(node).reshape(b,3,128)
        background=s[:,:6].reshape(-1,2,3).transpose(1,2)
        h=h*(1+.1*self.background(background))
        p=t[:,768:960].reshape(-1,3,2,32,150)
        # Both phase directions share weights; pair orientation cannot change pooling.
        flipped=torch.stack([p[:,:,0],-p[:,:,1]],2)
        e=self.pairs(torch.cat([p,flipped],1).reshape(-1,2,32,150)).reshape(b,6,128)
        strength=s[:,6:12].reshape(-1,2,3)
        clock=t[:,960:];duration=(clock[:,0,-1:]-clock[:,0,:1]).clamp_min(0)
        meta=torch.cat([strength.mean(-1),strength.amax(-1),s[:,12:16],torch.log1p(duration)],-1)
        z=self.fusion(torch.cat([self.node_pool(h),self.pair_pool(e),meta],-1))
        presence=self.presence(z);conditional=self.count(z).log_softmax(-1)
        logp=torch.cat([F.logsigmoid(-presence),F.logsigmoid(presence)+conditional],-1)
        return dict(logits=logp,ordinal=self.ordinal(z),embedding=F.normalize(z,dim=-1))
    def forward(self,t,s):return self.tasks(t,s)['logits']

def crossroom_loss(z,y,room):
    valid=~torch.eye(len(z),dtype=torch.bool,device=z.device)
    # Unknown rooms cannot be asserted to be distinct physical environments.
    positive=(y[:,None]==y[None,:])&(room[:,None]!=room[None,:])&(room[:,None]>=0)&(room[None,:]>=0)&valid
    scores=z@z.T/.15;logp=scores-scores.masked_fill(~valid,-1e4).logsumexp(-1,keepdim=True)
    n=positive.sum(-1);terms=-(logp*positive).sum(-1)/n.clamp_min(1)
    return (terms*(n>0)).sum()/(n>0).sum().clamp_min(1)
