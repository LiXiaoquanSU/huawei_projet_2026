"""Frequency/coherence descriptors with gain/common-phase invariant quantities."""
import torch
from torch import nn


class PhysicsPreprocessor(nn.Module):
    def forward(self,frames,intervals):
        # Deterministic subsampling for static frequency statistics (not motion timing).
        r,i=frames[:,::3,0,:,:,0],frames[:,::3,0,:,:,1]
        p=r.square()+i.square();mask=(p>1e-8).to(frames.dtype)
        scale=torch.sqrt(p.mean(-1,keepdim=True)+1e-4)
        r,i=r/scale,i/scale;p=r.square()+i.square()
        b=frames.shape[0]
        features=[]
        def stats(x):
            mean=x.mean(dim=1)
            std=torch.sqrt((x-mean[:,None]).square().mean(dim=1)+1e-6)
            return [mean.reshape(b,-1),std.reshape(b,-1)]
        for lag in [1,2,4,8,16,32,64,128,256]:
            re=(r[:,:,:,:-lag]*r[:,:,:,lag:]+i[:,:,:,:-lag]*i[:,:,:,lag:]).mean(-1)
            im=(i[:,:,:,:-lag]*r[:,:,:,lag:]-r[:,:,:,:-lag]*i[:,:,:,lag:]).mean(-1)
            coherence=torch.sqrt(re.square()+im.square()+1e-6)
            features+=stats(coherence)
        moments=[p.square().mean(-1)/10,
                 (torch.log(p+1e-4)*mask).sum(-1)/mask.sum(-1).clamp_min(1),
                 p.max(-1).values/10,
                 torch.sqrt((p-1).square().mean(-1)+1e-6),
                 (p[:,:,:,1:]-p[:,:,:,:-1]).abs().mean(-1)]
        for x in moments: features+=stats(x)
        corrs=[]
        for u,v in [(0,1),(0,2),(1,2)]:
            re=(r[:,:,u]*r[:,:,v]+i[:,:,u]*i[:,:,v]).mean(-1)
            im=(i[:,:,u]*r[:,:,v]-r[:,:,u]*i[:,:,v]).mean(-1)
            corrs.append((re,im))
            features+=stats(torch.sqrt(re.square()+im.square()+1e-6))
        for lag in [1,2,4,8,16]:
            ac=(p[:,lag:]*p[:,:-lag]).mean(dim=(1,3))/10
            features.append(ac)
        norms=[re.square()+im.square() for re,im in corrs]
        re01,im01=corrs[0];re02,im02=corrs[1];re12,im12=corrs[2]
        triple=(re01*re12-im01*im12)*re02+(re01*im12+im01*re12)*im02
        features+=stats((3+2*sum(norms))/9)
        features+=stats(1+2*triple-sum(norms))
        invariants=torch.cat(features,dim=1)
        # Normalized average frequency profile and relative temporal profile variance.
        logamp=.5*torch.log1p(p)
        pooled=(logamp*mask).reshape(b,200,3,64,16).sum(-1)/mask.reshape(b,200,3,64,16).sum(-1).clamp_min(1)
        mean=pooled.mean(1);std=torch.sqrt((pooled-mean[:,None]).square().mean(1)+1e-6)
        mean=mean-mean.mean(-1,keepdim=True)
        mean=mean/torch.sqrt(mean.square().mean(-1,keepdim=True)+1e-4)
        std=torch.log1p(std/.01)
        profile=torch.cat([mean.reshape(b,-1),std.reshape(b,-1)],dim=1)
        return torch.cat([invariants,profile],dim=1)


class PhysicsClassifier(nn.Module):
    def __init__(self,dimension,kernel='linear',n_support=1):
        super().__init__();self.preprocess=PhysicsPreprocessor();self.dimension=dimension;self.kernel=kernel
        self.register_buffer('mean',torch.zeros(dimension));self.register_buffer('scale',torch.ones(dimension))
        self.register_buffer('weights',torch.zeros(dimension+1 if kernel=='linear' else n_support,5))
        self.register_buffer('support',torch.zeros(n_support,dimension));self.register_buffer('gamma',torch.tensor(1.))

    def forward_features(self,features):
        z=((features[:,:self.dimension]-self.mean)/self.scale).clamp(-8,8)
        if self.kernel=='linear':
            z=torch.cat([z,torch.ones_like(z[:,:1])],dim=1)
        else:
            distance=(z.square().sum(-1,keepdim=True)+self.support.square().sum(-1)[None,:]-2*z@self.support.T).clamp_min(0)
            z=torch.exp(-self.gamma*distance)
        return z@self.weights

    def forward(self,frames,intervals):return self.forward_features(self.preprocess(frames,intervals))
