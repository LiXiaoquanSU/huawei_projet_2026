"""Larger permutation-equivariant receiver graph with multiscale temporal evidence."""
import torch
from torch import nn
from torch.nn import functional as F

def spread(x,dim):
    return ((x-x.mean(dim,keepdim=True)).square().mean(dim)+1e-5).sqrt()

class Encoder(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channels,48,3,padding=1),nn.GroupNorm(8,48),nn.SiLU(),nn.Conv2d(48,96,3,stride=(2,1),padding=1),nn.GroupNorm(8,96),nn.SiLU(),nn.Conv2d(96,128,3,stride=(2,1),padding=1),nn.GroupNorm(8,128),nn.SiLU())
        self.project=nn.Linear(128*4,256)
    def forward(self,x):
        x=self.net(x);x=x.reshape(x.shape[0],128,4,-1,30).mean(3)
        return self.project(x.flatten(1,2).transpose(1,2))

class GraphBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(768,1024),nn.SiLU(),nn.Linear(1024,256))
        self.update=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,1024),nn.SiLU(),nn.Linear(1024,256))
        self.norm=nn.LayerNorm(256);self.edge_norm=nn.LayerNorm(256);self.gain=nn.Parameter(torch.tensor(-2.2))
    def forward(self,n,e):
        # Directed edges:01,02,12,10,20,21. Endpoint correspondence is retained.
        src=n[:,[0,0,1,1,2,2]];dst=n[:,[1,2,2,0,0,1]]
        m=self.message(torch.cat([src,dst,e],-1))
        incoming=torch.stack([(m[:,3]+m[:,4])*.5,(m[:,0]+m[:,5])*.5,(m[:,1]+m[:,2])*.5],1)
        g=self.gain.sigmoid();return self.norm(n+g*self.update(torch.cat([n,incoming],-1))),self.edge_norm(e+g*m)

class EvidenceChain(nn.Module):
    def __init__(self,variant='full'):
        super().__init__();assert variant=='full'
        self.change=Encoder(3);self.relation=Encoder(2)
        self.state=nn.Sequential(nn.Linear(256,512),nn.LayerNorm(512),nn.SiLU(),nn.Linear(512,256))
        self.scale=nn.Sequential(nn.Linear(9,128),nn.SiLU(),nn.Linear(128,256))
        self.graph=nn.ModuleList([GraphBlock() for _ in range(2)])
        self.clock=nn.Linear(3,256)
        self.state_norm=nn.LayerNorm(256);self.change_norm=nn.LayerNorm(256);self.relation_norm=nn.LayerNorm(256)
        self.context_gain=nn.Parameter(torch.tensor(-2.2));self.clock_gain=nn.Parameter(torch.tensor(-2.2));self.temporal_gain=nn.Parameter(torch.full((4,),-2.2))
        self.dynamic_head=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,5))
        self.state_head=nn.Sequential(nn.Linear(512,256),nn.SiLU(),nn.Linear(256,5))
        self.local_conv=nn.Sequential(nn.Conv1d(256,256,5,padding=2,groups=256),nn.Conv1d(256,256,1),nn.SiLU())
        self.temporal=nn.ModuleList([nn.TransformerEncoderLayer(256,8,1536,dropout=.1,batch_first=True,norm_first=True,activation='gelu') for _ in range(4)])
        self.context_query=nn.MultiheadAttention(256,8,dropout=.1,batch_first=True)
        self.context_norm=nn.LayerNorm(256)
        self.local_head=nn.Linear(256,5)
        self.head=nn.Sequential(nn.Linear(256*5+10,1024),nn.LayerNorm(1024),nn.SiLU(),nn.Dropout(.2),nn.Linear(1024,512),nn.SiLU(),nn.Linear(512,5))
    def tasks(self,t,s):
        b=t.shape[0];a=t[:,:768].reshape(-1,6,128,150);static=a[:,:3]
        state=self.state(torch.cat([static.mean(-1),spread(static,-1)],-1))
        scales=torch.cat([s[:,:12].reshape(-1,4,3).transpose(1,2),s[:,None,12:17].expand(-1,3,-1)],-1)
        context=self.state_norm(state+self.scale(scales))
        d=a[:,3:].reshape(-1,1,128,150)
        slow5=F.avg_pool2d(F.pad(d,(2,2,0,0),mode='replicate'),(1,5),stride=1)
        slow25=F.avg_pool2d(F.pad(d,(12,12,0,0),mode='replicate'),(1,25),stride=1)
        n=self.change(F.avg_pool2d(torch.cat([d,d-slow5,d-slow25],1),(4,5))).reshape(b,3,30,256)
        p=t[:,768:960].reshape(-1,3,2,32,150);reverse=torch.stack([p[:,:,0],-p[:,:,1]],2)
        e=self.relation(F.avg_pool2d(torch.cat([p,reverse],1).reshape(-1,2,32,150),(1,5))).reshape(b,6,30,256)
        n=self.change_norm(n);e=self.relation_norm(e)
        dynamic=n.reshape(b,90,256)
        dynamic_logits=self.dynamic_head(torch.cat([dynamic.mean(1),spread(dynamic,1),dynamic.amax(1)],-1))
        state_logits=self.state_head(torch.cat([context.mean(1),spread(context,1)],-1))
        n=n+self.context_gain.sigmoid()*context[:,:,None]
        for block in self.graph:n,e=block(n,e)
        clock=t[:,960:].reshape(-1,3,30,5).mean(-1).transpose(1,2)
        clock=torch.cat([torch.log1p(clock[:,:,:1].clamp_min(0)),clock[:,:,1:]],-1)
        n=n+.1*self.local_conv(n.reshape(b*3,30,256).transpose(1,2)).transpose(1,2).reshape(b,3,30,256)
        n=n+self.clock_gain.sigmoid()*self.clock(clock)[:,None]
        # Context tokens and local tokens jointly attend; no receiver ID embeddings.
        x=torch.cat([context.reshape(b*3,1,256),n.reshape(b*3,30,256)],1)
        for i,block in enumerate(self.temporal):x=x+self.temporal_gain[i].sigmoid()*(block(x)-x)
        x=x.reshape(b,3,31,256);ctx=x[:,:,0];local=x[:,:,1:]
        q=ctx.mean(1,keepdim=True);kv=local.reshape(b,90,256)
        joint=self.context_norm(q+self.context_query(q,kv,kv,need_weights=False)[0]).squeeze(1)
        pooled=torch.cat([kv.mean(1),spread(kv,1),kv.amax(1),joint,spread(ctx,1)],-1)
        lp=self.local_head(kv).softmax(-1)
        main=self.head(torch.cat([pooled,lp.mean(1),spread(lp,1)],-1))
        logits=(main+.25*dynamic_logits+.1*state_logits).log_softmax(-1)
        return dict(logits=logits,aux_dynamic=dynamic_logits.log_softmax(-1),aux_state=state_logits.log_softmax(-1),local_probabilities=lp.transpose(1,2),modulation=joint)
    def forward(self,t,s):return self.tasks(t,s)['logits']
