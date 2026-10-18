"""Time-frequency CNN with optional training-only feature style mixing."""
import torch
from torch import nn


class TFPreprocessor(nn.Module):
    def forward(self,frames,intervals):
        b=frames.shape[0]
        p=frames[:,:,0,:,:,:].square().sum(-1)
        valid=(p>1e-8).to(frames.dtype)
        amp=.5*torch.log1p(p)
        amp=(amp*valid).reshape(b,600,3,128,8).sum(-1)/valid.reshape(b,600,3,128,8).sum(-1).clamp_min(1)
        image=amp.reshape(b,150,4,3,128).mean(2).permute(0,2,1,3)
        center=image.mean((2,3),keepdim=True)
        static=(image-center)/torch.sqrt((image-center).square().mean((2,3),keepdim=True)+1e-4)
        dynamic=image-image.mean(2,keepdim=True)
        dynamic=dynamic/torch.sqrt(dynamic.square().mean((2,3),keepdim=True)+1e-6)
        return torch.cat([static.clamp(-6,6),dynamic.clamp(-6,6)],dim=1)


class MixStyle(nn.Module):
    def forward(self,x):
        if not self.training:return x
        mean=x.mean((2,3),keepdim=True)
        std=torch.sqrt((x-mean).square().mean((2,3),keepdim=True)+1e-5)
        mean,std=mean.detach(),std.detach()
        order=torch.randperm(x.shape[0],device=x.device)
        weight=torch.distributions.Beta(.2,.2).sample((x.shape[0],1,1,1)).to(x.device)
        return (x-mean)/std*(weight*std+(1-weight)*std[order])+weight*mean+(1-weight)*mean[order]


class TFModel(nn.Module):
    def __init__(self,mode='tfcnn'):
        super().__init__();self.mode=mode;self.preprocess=TFPreprocessor()
        blocks=[];cin=6
        for j,cout in enumerate([24,48,96,128]):
            blocks.append(nn.Sequential(nn.Conv2d(cin,cout,3,stride=2,padding=1,bias=False),
                          nn.GroupNorm(8,cout),nn.SiLU(),
                          nn.Conv2d(cout,cout,3,padding=1,groups=cout,bias=False),
                          nn.Conv2d(cout,cout,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
                          MixStyle() if mode in ['tfcnn_mixstyle','tfcnn_domain'] and j<2 else nn.Identity()))
            cin=cout
        self.blocks=nn.Sequential(*blocks)
        self.head=nn.Sequential(nn.Linear(384,128),nn.SiLU(),nn.Dropout(.5),nn.Linear(128,5))

    def forward_features(self,x,unused=None):
        if self.mode=='tfcnn_dynamic':
            x=torch.cat([torch.zeros_like(x[:,:3]),x[:,3:]],dim=1)
        if self.training:
            if self.mode=='tfcnn_domain':
                # Random smooth frequency responses simulate nuisance room coloration.
                noise=torch.randn(x.shape[0],3,8,device=x.device)
                noise=torch.nn.functional.interpolate(noise,size=128,mode='linear',align_corners=False)
                noise=noise[:,:,None,:]*torch.rand(x.shape[0],3,1,1,device=x.device)*2
                x=torch.cat([x[:,:3]+noise,x[:,3:]],dim=1)
            # One frequency shift per batch, no interpolation across time gaps.
            shift=int(torch.randint(-16,17,()).item())
            x=torch.roll(x,shift,dims=-1)
            gain=torch.exp(torch.randn(x.shape[0],6,1,1,device=x.device)*.4)
            x=x*gain
            reverse=torch.rand(x.shape[0],1,1,1,device=x.device)<.5
            x=torch.where(reverse,x.flip(2),x)
            keep=(torch.rand(x.shape[0],1,1,128,device=x.device)>.1).to(x.dtype)
            x=x*keep
        x=self.blocks(x);mean=x.mean((2,3));std=torch.sqrt((x-mean[:,:,None,None]).square().mean((2,3))+1e-5)
        return self.head(torch.cat([mean,std,x.amax((2,3))],dim=1))

    def forward(self,frames,intervals):return self.forward_features(self.preprocess(frames,intervals))
