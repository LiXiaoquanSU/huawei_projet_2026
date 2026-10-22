"""Class-conditioned retrieval of four routes and five temporal segment tokens."""
import torch
from torch import nn
from network import old
from model import ordered_probabilities

class QueryTemporal(old.ContextEvents):
    def forward_details(self,a,p,t):
        tokens,features=self.local_features(a,p,t)
        tokens=self.context(self.segment(tokens)+self.clock(old.time_features(t)))
        return self.project(torch.cat((features,old.pool(tokens,1)),1)),tokens

class QueryModel(old.StructuredModel):
    def __init__(self,variant):
        super().__init__('spectral','physical')
        self.temporal=QueryTemporal();self.segment_projection=nn.Linear(128,256)
        self.class_queries=nn.Parameter(torch.randn(1,5,256)*.02)
        self.retrieve=old.CrossExchange()
        self.score=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64),nn.SiLU(),nn.Linear(64,1))
        self.variant=variant

    def forward_tasks(self,t,summary):
        # Original augmentation, four expert routes, losses and ordinal head retained.
        amplitude=t[:,:768].reshape(-1,6,128,150)
        phase=t[:,768:960].reshape(-1,6,32,150);timing=t[:,960:]
        if self.training:
            keep=(torch.rand(amplitude.shape[0],1,32,1,device=amplitude.device)>.05).to(amplitude.dtype)
            amplitude=(amplitude+torch.randn_like(amplitude)*.02)*keep.repeat_interleave(4,2)
            phase=(phase+torch.randn_like(phase)*.02)*keep
        # Match route call order, including MixStyle randomness.
        evidence=[self.static(amplitude[:,:3]),self.dynamic(amplitude[:,3:]),self.phase(phase)]
        temporal,segments=self.temporal.forward_details(amplitude[:,3:],phase,timing);evidence.append(temporal)
        content=torch.cat(evidence,1);context=self.context(torch.cat((summary[:,:6],summary[:,12:17]),1))
        gate=.5*self.gate(torch.cat((content,context),1)).softmax(-1)+.125
        experts=torch.stack([head(z) for head,z in zip(self.experts,evidence)],1)
        present=torch.ones(amplitude.shape[0],4,device=amplitude.device,dtype=amplitude.dtype)
        if self.training:
            present=(torch.rand(amplitude.shape[0],4,device=amplitude.device)>.25).to(amplitude.dtype)
            present=torch.where(present.sum(1,keepdim=True)>0,present,torch.ones_like(present))
            gate=gate*present;gate=gate/gate.sum(1,keepdim=True)
            content=torch.cat([z*present[:,k:k+1] for k,z in enumerate(evidence)],1)
        mixture=(experts.softmax(-1)*gate[:,:,None]).sum(1)
        z=self.shared(content);strength=torch.cat((summary[:,6:12],summary[:,12:17]),1);z=z+self.strength(strength)
        routes=torch.stack(evidence,1)*present[:,:,None]
        # Dropping temporal route also drops its segment memory; no bypass of route dropout.
        memory=torch.cat((routes,self.segment_projection(segments)*present[:,3,None,None]),1)
        query=self.class_queries.expand(z.shape[0],-1,-1)+z[:,None]
        readout=self.score(self.retrieve(query,memory)).squeeze(-1)
        logits=self.classifier(z)+readout
        direct=.5*logits.softmax(-1)+.5*mixture
        ordinal_logits=self.ordinal(z);presence=self.presence(torch.cat((*evidence[1:],strength),1))
        ordinal_logits=torch.cat((presence,ordinal_logits[:,1:]),1)
        ordered,cumulative=ordered_probabilities(ordinal_logits);probabilities=.5*(direct+ordered)
        return dict(logits=probabilities.clamp_min(1e-8).log(),direct=direct,ordinal=ordered,conditional_logits=ordinal_logits,cumulative=cumulative,experts=experts,gate=gate,signal=self.motion(torch.cat(evidence[1:],1)))

def build(variant):
    assert variant=='query_context'
    return QueryModel(variant)
