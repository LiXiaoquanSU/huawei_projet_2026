"""Dual CSI encoders preserving static spectral shape as well as dynamics.

Unlike the first relative-only ablation, the amplitude branch retains its
window-level frequency profile. This tests a representation hypothesis rather
than adding capacity or correcting individual classes.
"""
import torch
from torch import nn
from tf_model import TFPreprocessor
from phase_models import RelativePreprocessor
from domain_model import DomainModel,Encoder

def combine(tf,phase):
    b=tf.shape[0]
    amplitude=tf.reshape(b,6,150,32,4).mean(-1).permute(0,1,3,2)
    relative=phase[:,96:288].reshape(b,6,32,150,4).mean(-1)
    return torch.cat([amplitude,relative],1).reshape(b,384,150)

class MultiviewPreprocessor(nn.Module):
    def __init__(self):
        super().__init__();self.amplitude=TFPreprocessor();self.relative=RelativePreprocessor()
    def forward(self,frames,intervals):
        a=self.amplitude(frames,intervals);p,_=self.relative(frames,intervals)
        return combine(a,p),torch.zeros_like(intervals[:,:17])

class MultiviewDomainModel(DomainModel):
    def __init__(self):
        super().__init__();self.preprocess=MultiviewPreprocessor();self.amplitude=Encoder(6)
        self.decoder=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,12*8*30))
    def encode(self,t):
        x=t.reshape(-1,12,32,150)
        return self.bottleneck(torch.cat([self.amplitude(x[:,:6]),self.phase(x[:,6:])],1))

def load_multiview_bank():
    from run_iterations import load_bank
    tf,_,y,rows=load_bank(tf=True);phase,s,yp,rp=load_bank(phase=True)
    assert torch.equal(y,yp)
    assert [(r['identifier'],r['start']) for r in rows]==[(r['identifier'],r['start']) for r in rp]
    return combine(tf,phase),s,y,rows
