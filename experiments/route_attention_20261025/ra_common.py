"""Frozen054set evidence -> bounded conditional residual; actual ONNX features."""
import sys,copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/next053_20261025'))
import n53_common as prev
old=prev.old;read,save,sha=prev.read,prev.save,prev.sha
PREV=prev.WORK;WORK=Path('D:/huawei2026-experiments/route_attention_20261025')
SOURCE=ROOT/'models/next053_20261025/01_g54_selective_lowcount.onnx'
SOURCE_SHA='d4f5b05464a0754a105da21c1d991850abf001207b14e1295224075ea3f0ed41'

def moments(e):
    mean=e.mean(1);std=((e-mean[:,None]).square().mean(1)+1e-4).sqrt();return torch.cat([mean,std],-1)
def auxiliary(raw,b,T):
    p=(raw.exp().mean(2).clamp_min(1e-12).log()/T+b).softmax(-1)
    return p.std(1,unbiased=False)

class RouteResidual(nn.Module):
    def __init__(self,architecture='attention'):
        super().__init__();self.architecture=architecture
        self.project=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,64),nn.SiLU())
        self.route=nn.Parameter(torch.randn(1,3,64)*.01)
        if architecture=='attention':
            self.norm=nn.LayerNorm(64);self.qkv=nn.Linear(64,192);self.out=nn.Linear(64,64)
            self.ffnorm=nn.LayerNorm(64);self.ff=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64))
        self.readout=nn.Sequential(nn.LayerNorm(138),nn.Linear(138,64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,5,bias=False))
        nn.init.zeros_(self.readout[-1].weight)
    def forward(self,stats,base,aux):
        x=self.project(stats)+self.route
        if self.architecture=='attention':
            q,k,v=self.qkv(self.norm(x)).chunk(3,-1)
            def heads(z):return z.reshape(-1,3,4,16).transpose(1,2)
            q,k,v=heads(q),heads(k),heads(v);a=(q@k.transpose(-1,-2)/4).softmax(-1)
            x=x+self.out((a@v).transpose(1,2).reshape(-1,3,64));x=x+self.ff(self.ffnorm(x))
        z=torch.cat([x.mean(1),x.amax(1),base.softmax(-1),aux],-1)
        delta=self.readout(z).tanh();return delta-delta.mean(-1,keepdim=True)

def deploy(base,delta,amount=.5):
    proposed=base+amount*delta
    return torch.where((base.argmax(-1)<2)[:,None],proposed,base)
def model_from(cp):
    m=RouteResidual(cp['architecture']);m.load_state_dict(cp['state']);return m.eval()
