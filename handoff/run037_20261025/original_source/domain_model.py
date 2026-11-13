"""DASECount-inspired dual representation, AirFi-inspired training objectives.

This is an adaptation, not a reproduction: no labelled target-room adaptation.
"""
import torch
from torch import nn
from phase_models import RelativePreprocessor

class Encoder(nn.Module):
    def __init__(self,cin):
        super().__init__();layers=[]
        for cout in [24,48,96]:
            layers.extend([nn.Conv2d(cin,cout,(3,7),stride=(2,4),padding=(1,3),bias=False),
                nn.GroupNorm(8,cout),nn.SiLU()]);cin=cout
        self.layers=nn.Sequential(*layers)
    def forward(self,x):
        x=self.layers(x);mean=x.mean((2,3))
        std=(x.sub(mean[:,:,None,None]).square().mean((2,3))+1e-5).sqrt()
        return torch.cat([mean,std],1)

class DomainModel(nn.Module):
    def __init__(self):
        super().__init__();self.preprocess=RelativePreprocessor()
        self.amplitude=Encoder(3);self.phase=Encoder(6)
        self.bottleneck=nn.Sequential(nn.Linear(384,128),nn.LayerNorm(128),nn.SiLU())
        self.classifier=nn.Sequential(nn.Dropout(.4),nn.Linear(128,5))
        # Decoder is training-only and omitted from exported prediction graph.
        self.decoder=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,9*8*30))
    def encode(self,t):
        b=t.shape[0]
        amp=t[:,:96].reshape(b,3,32,600)
        phase=t[:,96:288].reshape(b,6,32,600)
        return self.bottleneck(torch.cat([self.amplitude(amp),self.phase(phase)],1))
    def forward_features(self,t,unused=None):return self.classifier(self.encode(t))
    def forward(self,frames,intervals):return self.forward_features(*self.preprocess(frames,intervals))

def conditional_mmd(z,y,rooms):
    """Biased RBF MMD, only same-class source room pairs with >=2 samples each."""
    terms=[];z=nn.functional.normalize(z,dim=1)
    for c in y.unique():
        groups=[z[(y==c)&(rooms==r)] for r in rooms[y==c].unique()]
        groups=[g for g in groups if len(g)>=2]
        for i,a in enumerate(groups):
            for b in groups[i+1:]:
                def kernel(u,v):
                    d=torch.cdist(u,v).square()
                    return sum(torch.exp(-d/s) for s in [.25,1.,4.])/3
                terms.append(kernel(a,a).mean()+kernel(b,b).mean()-2*kernel(a,b).mean())
    return torch.stack(terms).mean() if terms else z.sum()*0
