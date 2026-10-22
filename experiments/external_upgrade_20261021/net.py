"""Shared temporal evidence bank; original four routes are preserved."""
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from model import StructuredModel

def bank_input(t):return t[:,:384].reshape(-1,3,32,4,150).mean(3).reshape(-1,96,150)

class TemporalBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(1,16,7,2,3),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv1d(16,32,5,2,2),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv1d(32,32,3,2,1),nn.GroupNorm(4,32),nn.SiLU())
        self.link=nn.Sequential(nn.Linear(65,64),nn.LayerNorm(64),nn.SiLU())
        self.pool=nn.Sequential(nn.Linear(192,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,256),nn.LayerNorm(256),nn.SiLU())
    def forward(self,a):
        # Statistical set pooling supports genuine variable channel counts.
        b,c,n=a.shape;center=a-a.mean(-1,keepdim=True)
        var=center.square().mean(-1);std=(var+1e-5).sqrt();mask=(var>1e-7).to(a.dtype)
        count=mask.sum(1,keepdim=True).clamp_min(1)
        global_std=((var*mask).sum(1,keepdim=True)/count+1e-5).sqrt()
        strength=torch.log1p(std/global_std).clamp_max(5)
        x=(center/std[:,:,None]).clamp(-6,6).reshape(b*c,1,n)
        x=self.conv(x);mean=x.mean(-1);sd=(x.sub(mean[:,:,None]).square().mean(-1)+1e-5).sqrt()
        h=self.link(torch.cat([mean,sd,strength.reshape(-1,1)],1)).reshape(b,c,64)
        mean=(h*mask[:,:,None]).sum(1)/count
        sd=((h.sub(mean[:,None]).square()*mask[:,:,None]).sum(1)/count+1e-5).sqrt()
        maximum=(h+(1-mask[:,:,None])*(-10000)).amax(1)
        maximum=torch.where(mask.sum(1,keepdim=True)>0,maximum,torch.zeros_like(maximum))
        return self.pool(torch.cat([mean,sd,maximum],1))

class UpgradeModel(StructuredModel):
    def __init__(self):
        super().__init__('spectral','physical')
        self.bank=TemporalBank();self.bank_count=nn.Linear(256,5);self.bank_activity=nn.Linear(256,4)
        self.fuse=nn.Linear(256,256)
        nn.init.zeros_(self.fuse.weight);nn.init.zeros_(self.fuse.bias)
    def refine_latent(self,z,t,summary):
        h=self.bank(bank_input(t));self.bank_embedding=h;self.latent=z+self.fuse(h)
        return self.latent
    def forward_tasks(self,t,s):
        tasks=super().forward_tasks(t,s)
        tasks['latent']=self.latent;tasks['bank_logits']=self.bank_count(self.bank_embedding)
        return tasks

class Submission(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,frames,intervals):
        t,s=self.model.preprocess(frames[None].double(),intervals[None].double())
        return self.model.forward_features(t.float(),s.float())[0]

def fresh(seed):torch.manual_seed(seed);return UpgradeModel()
