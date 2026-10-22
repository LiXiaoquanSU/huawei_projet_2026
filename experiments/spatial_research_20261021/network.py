"""Three spatial representation hypotheses; unchanged context-events temporal route."""
import importlib.util,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('global_local_reference',ROOT/'experiments/global_local_20261021/network.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class AttentionFusion(nn.Module):
    def __init__(self,c):
        super().__init__();h=c//4
        self.local=nn.Sequential(nn.Conv2d(c,h,1),nn.GroupNorm(8,h),nn.SiLU(),nn.Conv2d(h,c,1))
        self.global_net=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,h,1),nn.SiLU(),nn.Conv2d(h,c,1))
        for layer in (self.local[-1],self.global_net[-1]):nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,x):return (self.local(x)+self.global_net(x)).sigmoid()

class AFFBlock(nn.Module):
    def __init__(self,original):
        super().__init__();self.body=original.body;self.skip=original.skip
        self.attention=AttentionFusion(self.skip.out_channels)
    def forward(self,x):
        a,b=self.body(x),self.skip(x);w=self.attention(a+b)
        # Factor2 preserves the original residual sum at initialized gate=.5.
        return nn.functional.silu(2*(w*a+(1-w)*b))

class Res2Block(nn.Module):
    def __init__(self,original):
        super().__init__();c=original.skip.out_channels;self.stem=nn.Sequential(*list(original.body.children())[:3]);self.skip=original.skip
        self.chunks=nn.ModuleList([nn.Sequential(nn.Conv2d(c//4,c//4,3,1,1,bias=False),nn.GroupNorm(8,c//4),nn.SiLU()) for _ in range(3)])
        self.mix=nn.Sequential(nn.Conv2d(c,c,1,bias=False),nn.GroupNorm(8,c))
    def forward(self,x):
        pieces=self.stem(x).chunk(4,1);out=[pieces[0]]
        for k,layer in enumerate(self.chunks):out.append(layer(pieces[k+1]+out[-1]))
        return nn.functional.silu(self.mix(torch.cat(out,1))+self.skip(x))

class PyramidEncoder(nn.Module):
    def __init__(self,original):
        super().__init__();self.layers=original.layers;self.style=original.style;self.project=original.project;self.frequency_bins=original.frequency_bins
        self.shallow=nn.ModuleList([nn.Sequential(nn.Linear(c*3*4,256),nn.LayerNorm(256),nn.SiLU()) for c in (128,256)])
        self.scales=nn.Parameter(torch.full((2,),-2.0))
    def forward(self,x):
        evidence=[]
        for i,layer in enumerate(self.layers):
            x=layer(x)
            if i<2:x=self.style(x)
            if i in (1,2):
                mean=x.mean(-1);std=((x-mean[:,:,:,None]).square().mean(-1)+1e-5).sqrt();maximum=x.amax(-1)
                pooled=torch.cat([nn.functional.adaptive_avg_pool1d(v,4).flatten(1) for v in (mean,std,maximum)],1)
                evidence.append(self.shallow[i-1](pooled))
        mean=x.mean(-1);std=((x-mean[:,:,:,None]).square().mean(-1)+1e-5).sqrt()
        z=self.project(torch.cat((mean.flatten(1),std.flatten(1),x.amax(-1).flatten(1)),1))
        return z+sum(w*v for w,v in zip(self.scales.sigmoid(),evidence))

def build(variant):
    if variant in ('context_events','cross_events'):return old.build(variant)
    model=old.build('context_events')
    for name in ('static','dynamic','phase'):
        encoder=getattr(model,name)
        if variant=='pyramid_context':setattr(model,name,PyramidEncoder(encoder))
        elif variant=='aff_context':encoder.layers=nn.ModuleList([AFFBlock(layer) for layer in encoder.layers])
        elif variant=='res2_context':encoder.layers=nn.ModuleList([Res2Block(layer) for layer in encoder.layers])
        else:raise ValueError(variant)
    return model
