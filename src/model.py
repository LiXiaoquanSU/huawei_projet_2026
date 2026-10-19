"""The unchanged four-route 82 MB model used by both retained checkpoints."""
import math
import torch
from torch import nn
from features import TFPreprocessor,MixStyle,RelativePreprocessor,CSIPreprocessor,combine

def pack_features(tf,phase,resolution=32):
    b=tf.shape[0]
    if resolution==32:image=combine(tf,phase)
    else:
        amplitude=tf.permute(0,1,3,2).reshape(b,768,150)
        relative=phase[:,96:288].reshape(b,192,150,4).mean(-1)
        image=torch.cat([amplitude,relative],1)
    dt=.005*(torch.exp(phase[:,288])-1)
    time=torch.cumsum(dt,dim=-1).reshape(b,150,4).mean(-1)
    step=torch.log1p(dt.reshape(b,150,4).sum(-1)/.02)
    gap=phase[:,289].reshape(b,150,4).amax(-1)
    return torch.cat([image,time[:,None],step[:,None],gap[:,None]],1)

class StructuredPreprocessor(nn.Module):
    def __init__(self,resolution=32):
        super().__init__();self.resolution=resolution;self.tf=TFPreprocessor();self.phase=RelativePreprocessor();self.statistics=CSIPreprocessor()
    def forward(self,frames,intervals):
        tf=self.tf(frames,intervals);phase,_=self.phase(frames,intervals)
        _,summary=self.statistics(frames,intervals)
        return pack_features(tf,phase,self.resolution),summary

class Residual2D(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(cin,cout,3,2,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
            nn.Conv2d(cout,cout,3,1,1,bias=False),nn.GroupNorm(8,cout))
        self.skip=nn.Conv2d(cin,cout,1,2,bias=False)
    def forward(self,x):return nn.functional.silu(self.body(x)+self.skip(x))

class EvidenceEncoder(nn.Module):
    def __init__(self,cin,widths,embed,style=False,frequency_bins=None):
        super().__init__();self.layers=nn.ModuleList([Residual2D(a,b) for a,b in zip([cin]+widths[:-1],widths)])
        self.style=MixStyle() if style else nn.Identity()
        self.frequency_bins=frequency_bins
        self.project=nn.Sequential(nn.Linear(widths[-1]*3*(frequency_bins//16 if frequency_bins else 1),embed),nn.LayerNorm(embed),nn.SiLU())
    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x=layer(x)
            if i<2:x=self.style(x)
        if self.frequency_bins:
            mean=x.mean(-1);std=(x.sub(mean[:,:,:,None]).square().mean(-1)+1e-5).sqrt()
            return self.project(torch.cat([mean.flatten(1),std.flatten(1),x.amax(-1).flatten(1)],1))
        mean=x.mean((2,3));std=(x.sub(mean[:,:,None,None]).square().mean((2,3))+1e-5).sqrt()
        return self.project(torch.cat([mean,std,x.amax((2,3))],1))

class AttentionBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.width=width;self.heads=8;self.norm1=nn.LayerNorm(width);self.qkv=nn.Linear(width,width*3)
        self.out=nn.Linear(width,width);self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Dropout(.1),nn.Linear(width*4,width),nn.Dropout(.1))
    def forward(self,x):
        b,n,c=x.shape
        qkv=self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,c//self.heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attention=(q@k.transpose(-1,-2)/math.sqrt(c//self.heads)).softmax(-1)
        x=x+self.out((attention@v).transpose(1,2).reshape(b,n,c))
        return x+self.ff(self.norm2(x))

class TimedEncoder(nn.Module):
    def __init__(self,width,depth,embed):
        super().__init__();self.input=nn.Conv1d(288,width,5,2,2)
        self.register_buffer('frequencies',torch.tensor([.5,1.,2.,4.,8.,16.]))
        self.time=nn.Linear(14,width);self.blocks=nn.Sequential(*[AttentionBlock(width) for _ in range(depth)])
        self.project=nn.Sequential(nn.Linear(width*2,embed),nn.LayerNorm(embed),nn.SiLU())
    def forward(self,amplitude,phase,timing):
        if amplitude.shape[2]==128:amplitude=amplitude.reshape(-1,3,32,4,150).mean(3)
        dynamic=torch.cat([amplitude,phase],1).reshape(-1,288,150)
        x=self.input(dynamic).transpose(1,2)
        clock=timing.reshape(-1,3,75,2).mean(-1).transpose(1,2)
        angle=clock[:,:,:1]*self.frequencies[None,None,:]*6.283185307179586
        x=x+self.time(torch.cat([angle.sin(),angle.cos(),clock[:,:,1:]],-1))
        x=self.blocks(x);mean=x.mean(1);std=(x.sub(mean[:,None]).square().mean(1)+1e-5).sqrt()
        return self.project(torch.cat([mean,std],1))

def ordered_probabilities(conditional_logits):
    """CORN-style cumulative conditional probabilities, monotonically decreasing."""
    p=conditional_logits.sigmoid();q=[];v=p[:,0]
    q.append(v)
    for k in range(1,4):v=v*p[:,k];q.append(v)
    cumulative=torch.stack(q,1)
    probabilities=torch.cat([1-cumulative[:,:1],cumulative[:,:-1]-cumulative[:,1:],cumulative[:,-1:]],1)
    return probabilities,cumulative

def signal_targets(t):
    """Measured amplitude/phase coherence; no human activity ground truth implied."""
    resolution=(t.shape[1]-195)//6
    amplitude=t[:,:6*resolution].reshape(-1,6,resolution,150)[:,3:]
    if resolution==128:amplitude=amplitude.reshape(-1,3,32,4,150).mean(3)
    phase=t[:,6*resolution:6*resolution+192].reshape(-1,192,150);clock=t[:,6*resolution+192];terms=[]
    for x in [amplitude.reshape(-1,96,150),phase]:
        for lag in [1,4,16]:
            a,b=x[:,:,:-lag],x[:,:,lag:]
            corr=(a*b).mean(1)/(a.square().mean(1)*b.square().mean(1)+1e-6).sqrt()
            valid=((clock[:,lag:]-clock[:,:-lag])<(.04*lag+.01)).to(x.dtype)
            terms.append((corr.clamp(-1,1)*valid).sum(-1)/valid.sum(-1).clamp_min(1))
    return torch.stack(terms,1)

class StructuredModel(nn.Module):
    def refine_latent(self,z,t,summary):
        return z

    def __init__(self,size='small',mode='auxiliary'):
        super().__init__();self.size=size;self.mode=mode;self.resolution=128 if size=='spectral' else 32;self.preprocess=StructuredPreprocessor(self.resolution)
        widths=[32,64,128,192] if size=='small' else [64,128,256,384]
        embed=128 if size=='small' else 256
        self.static=EvidenceEncoder(3,widths,embed,style=True,frequency_bins=self.resolution if size=='spectral' else None)
        self.dynamic=EvidenceEncoder(3,widths,embed,frequency_bins=self.resolution if size=='spectral' else None)
        self.phase=EvidenceEncoder(6,widths,embed,frequency_bins=32 if size=='spectral' else None)
        self.temporal=TimedEncoder(128 if size=='small' else 256,2 if size=='small' else 3,embed)
        self.context=nn.Sequential(nn.Linear(11,64),nn.LayerNorm(64),nn.SiLU())
        self.gate=nn.Sequential(nn.Linear(embed*4+64,128),nn.SiLU(),nn.Linear(128,4))
        self.experts=nn.ModuleList([nn.Linear(embed,5) for _ in range(4)])
        self.shared=nn.Sequential(nn.Linear(embed*4,embed*2),nn.LayerNorm(embed*2),nn.SiLU(),nn.Dropout(.4),nn.Linear(embed*2,embed),nn.SiLU())
        self.classifier=nn.Linear(embed,5);self.ordinal=nn.Linear(embed,4)
        self.motion=nn.Sequential(nn.Linear(embed*3,128),nn.SiLU(),nn.Linear(128,6),nn.Tanh())
        if mode=='physical':
            self.strength=nn.Sequential(nn.Linear(11,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,embed))
            self.presence=nn.Sequential(nn.Linear(embed*3+11,128),nn.SiLU(),nn.Linear(128,1))

    def forward_tasks(self,t,summary):
        amplitude=t[:,:6*self.resolution].reshape(-1,6,self.resolution,150)
        phase=t[:,6*self.resolution:6*self.resolution+192].reshape(-1,6,32,150);timing=t[:,6*self.resolution+192:]
        if self.training:
            keep=(torch.rand(amplitude.shape[0],1,32,1,device=amplitude.device)>.05).to(amplitude.dtype)
            amplitude=(amplitude+torch.randn_like(amplitude)*.02)*(keep.repeat_interleave(4,2) if self.resolution==128 else keep)
            phase=(phase+torch.randn_like(phase)*.02)*keep
        evidence=[self.static(amplitude[:,:3]),self.dynamic(amplitude[:,3:]),self.phase(phase),self.temporal(amplitude[:,3:],phase,timing)]
        content=torch.cat(evidence,1)
        context=self.context(torch.cat([summary[:,:6],summary[:,12:17]],1))
        # A route retains at least 1/8 weight at inference; no single route >5/8.
        gate=.5*self.gate(torch.cat([content,context],1)).softmax(-1)+.125
        experts=torch.stack([head(z) for head,z in zip(self.experts,evidence)],1)
        if self.training:
            present=(torch.rand(amplitude.shape[0],4,device=amplitude.device)>.25).to(amplitude.dtype)
            present=torch.where(present.sum(1,keepdim=True)>0,present,torch.ones_like(present))
            gate=gate*present;gate=gate/gate.sum(1,keepdim=True)
            content=torch.cat([z*present[:,k:k+1] for k,z in enumerate(evidence)],1)
        mixture=(experts.softmax(-1)*gate[:,:,None]).sum(1)
        z=self.shared(content)
        if self.mode=='physical':
            # Preserve absolute disturbance strength removed by image normalization.
            # Labels supervise presence directly; these statistics are NOT motion labels.
            strength=torch.cat([summary[:,6:12],summary[:,12:17]],1)
            z=z+self.strength(strength)
        z=self.refine_latent(z,t,summary)
        logits=self.classifier(z)
        direct=.5*logits.softmax(-1)+.5*mixture
        ordinal_logits=self.ordinal(z)
        if self.mode=='physical':
            presence=self.presence(torch.cat([*evidence[1:],strength],1))
            ordinal_logits=torch.cat([presence,ordinal_logits[:,1:]],1)
        ordered,cumulative=ordered_probabilities(ordinal_logits)
        probabilities=direct if self.mode=='direct' else (direct+ordered)*.5
        return dict(logits=probabilities.clamp_min(1e-8).log(),direct=direct,ordinal=ordered,
            conditional_logits=ordinal_logits,cumulative=cumulative,experts=experts,gate=gate,
            signal=self.motion(torch.cat(evidence[1:],1)))
    def forward_features(self,t,summary):return self.forward_tasks(t,summary)['logits']
    def forward(self,frames,intervals):return self.forward_features(*self.preprocess(frames,intervals))
