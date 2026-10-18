"""Training-only nuisance projection followed by a regularized kernel head."""
import torch
from torch import nn
from physics_model import PhysicsPreprocessor
from improved_models import ImprovedModel
from phase_models import PhaseModel


class DescriptorBank(nn.Module):
    def __init__(self,source='all'):
        super().__init__();self.source=source
        self.physics=PhysicsPreprocessor()
        self.amplitude=ImprovedModel('spectrum')
        self.phase=PhaseModel('phase_spectrum')

    def forward(self,frames,intervals):
        pieces=[]
        if self.source in ['physics','all']:pieces.append(self.physics(frames,intervals))
        if self.source in ['amplitude','all']:
            t,s=self.amplitude.preprocess(frames,intervals)
            pieces.append(self.amplitude.spectral_features(self.amplitude.signal(t),t))
        if self.source in ['phase','all']:
            t,s=self.phase.preprocess(frames,intervals)
            pieces.append(self.phase.spectral(t[:,:288],t))
        return torch.cat(pieces,dim=1)


class ProjectedClassifier(nn.Module):
    def __init__(self,source,dimension,kernel,n_support,rank):
        super().__init__();self.descriptors=DescriptorBank(source);self.dimension=dimension;self.kernel=kernel
        for name,value in [('mean',torch.zeros(dimension)),('scale',torch.ones(dimension)),
                           ('directions',torch.zeros(rank,dimension)),('strength',torch.tensor(1.)),
                           ('support',torch.zeros(n_support,dimension)),('gamma',torch.tensor(1.)),
                           ('weights',torch.zeros(dimension+1 if kernel=='linear' else n_support,5))]:
            self.register_buffer(name,value)

    def forward_features(self,features):
        x=((features-self.mean)/self.scale).clamp(-8,8)
        x=x-self.strength*(x@self.directions.T)@self.directions
        if self.kernel=='linear':x=torch.cat([x,torch.ones_like(x[:,:1])],dim=1)
        else:
            d=(x.square().sum(-1,keepdim=True)+self.support.square().sum(-1)[None]-2*x@self.support.T).clamp_min(0)
            x=torch.exp(-self.gamma*d)
        return x@self.weights

    def forward(self,frames,intervals):return self.forward_features(self.descriptors(frames,intervals))
