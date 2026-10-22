"""ROCKET-inspired fixed random dictionary with train-only standardized ridge readout."""
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from model import StructuredPreprocessor

class RandomDictionary(nn.Module):
    def __init__(self):
        super().__init__();self.preprocess=StructuredPreprocessor(128);self.filters=nn.ModuleList()
        for k in (7,9,11):
            for dilation in (1,4,16):
                layer=nn.Conv1d(288,64,k,padding=(k-1)*dilation//2,dilation=dilation)
                with torch.no_grad():
                    w=torch.randn_like(layer.weight);w=w-w.mean(-1,keepdim=True)
                    mask=(torch.rand(64,288,1)<.05).float();mask[:,0]=1
                    w=w*mask;w=w/(w.square().sum((1,2),keepdim=True)+1e-6).sqrt()
                    layer.weight.copy_(w);layer.bias.uniform_(-1,1)
                layer.requires_grad_(False);self.filters.append(layer)
        self.register_buffer('center',torch.zeros(1169));self.register_buffer('scale',torch.ones(1169))
        self.register_buffer('coef',torch.zeros(1169,5));self.register_buffer('intercept',torch.zeros(5))
    def features(self,t,summary):
        a=t[:,384:768].reshape(-1,3,32,4,150).mean(3)
        p=t[:,768:960].reshape(-1,6,32,150)
        x=torch.cat([a,p],1).flatten(1,2);parts=[]
        for layer in self.filters:
            z=layer(x);parts.extend([z.amax(-1),(z>0).to(z.dtype).mean(-1)])
        return torch.cat(parts+[summary],1)
    def forward_features(self,t,summary):
        x=(self.features(t,summary)-self.center)/self.scale
        return (x@self.coef+self.intercept).log_softmax(-1)
    def forward(self,frames,intervals):return self.forward_features(*self.preprocess(frames,intervals))
def build(variant='random_dictionary'):
    assert variant=='random_dictionary';return RandomDictionary()
