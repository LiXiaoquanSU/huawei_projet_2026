"""Joint state/change/relation evidence, context modulation and temporal disagreement."""
import torch
from torch import nn
from torch.nn import functional as F

def spread(x,dim):
    return ((x-x.mean(dim,keepdim=True)).square().mean(dim)+1e-5).sqrt()

class LocalEncoder(nn.Module):
    def __init__(self,channels):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv2d(16,32,3,stride=(2,1),padding=1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,32,3,stride=(2,1),padding=1),nn.GroupNorm(8,32),nn.SiLU())
    def forward(self,x):return self.net(x).mean(2)

class EvidenceChain(nn.Module):
    def __init__(self,variant='full'):
        super().__init__();self.variant=variant
        self.state_encoder=nn.Sequential(nn.Conv1d(2,16,5,padding=2),nn.SiLU(),nn.Conv1d(16,32,5,stride=2,padding=2),nn.SiLU(),nn.AvgPool1d(8),nn.Flatten(),nn.Linear(256,64),nn.LayerNorm(64),nn.SiLU())
        self.change=LocalEncoder(2);self.relation=LocalEncoder(2)
        self.context=nn.Sequential(nn.Linear(128+13,64),nn.LayerNorm(64),nn.SiLU())
        self.modulate=nn.Linear(64,256);nn.init.zeros_(self.modulate.weight);nn.init.zeros_(self.modulate.bias)
        self.time_embed=nn.Conv1d(3,128,1)
        self.temporal=nn.ModuleList([nn.Sequential(nn.Conv1d(128,128,3,padding=d,dilation=d,groups=128),nn.Conv1d(128,128,1),nn.GroupNorm(8,128),nn.SiLU()) for d in (1,2,4)])
        self.local_head=nn.Conv1d(128,5,1)
        self.head=nn.Sequential(nn.Linear(128*3+10+64,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.2),nn.Linear(128,5))
    def tasks(self,t,s):
        b=t.shape[0];a=t[:,:768].reshape(-1,6,128,150)
        static=a[:,:3];state=torch.stack([static.mean(-1),spread(static,-1)],2).reshape(-1,2,128)
        state=self.state_encoder(state).reshape(b,3,64);state=torch.cat([state.mean(1),spread(state,1)],-1)
        scale=s[:,:12].reshape(-1,4,3);summary=torch.cat([scale.mean(-1),spread(scale,-1),s[:,12:17]],-1)
        context=self.context(torch.cat([state,summary],-1))
        dynamic=a[:,3:].reshape(-1,1,128,150)
        slow=F.avg_pool2d(F.pad(dynamic,(2,2,0,0),mode='replicate'),(1,5),stride=1)
        # Preserve local time slices until the joint evidence stage (30 slices/view).
        change=F.avg_pool2d(torch.cat([dynamic, dynamic-slow],1),(4,5))
        change=self.change(change).reshape(b,3,32,30);change=torch.cat([change.mean(1),spread(change,1)],1)
        p=t[:,768:960].reshape(-1,3,2,32,150);reverse=torch.stack([p[:,:,0],-p[:,:,1]],2)
        p=F.avg_pool2d(torch.cat([p,reverse],1).reshape(-1,2,32,150),(1,5))
        relation=self.relation(p).reshape(b,6,32,30);relation=torch.cat([relation.mean(1),spread(relation,1)],1)
        local=torch.cat([change,relation],1)
        gamma,beta=(.2*self.modulate(context).tanh()).chunk(2,-1)
        enabled=0. if self.variant=='no_context' else 1.
        local=local*(1+enabled*gamma[:,:,None])+enabled*beta[:,:,None]
        clock=t[:,960:].reshape(-1,3,30,5).mean(-1)
        clock=torch.cat([torch.log1p(clock[:,:1].clamp_min(0)),clock[:,1:]],1)
        local=local+self.time_embed(clock)
        for layer in self.temporal:local=local+layer(local)
        lp=self.local_head(local).softmax(1);mean=local.mean(-1);std=spread(local,-1);peak=local.amax(-1);disagreement=spread(lp,-1)
        # Matched parameter-count ablation: global mean/context remain; variability removed.
        if self.variant=='no_spread':std=std*0;peak=mean;disagreement=disagreement*0
        pooled=torch.cat([mean,std,peak,lp.mean(-1),disagreement,context],-1)
        logits=self.head(pooled).log_softmax(-1)
        return dict(logits=logits,local_probabilities=lp,context=context,modulation=torch.cat([gamma,beta],-1))
    def forward(self,t,s):return self.tasks(t,s)['logits']
