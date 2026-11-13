"""Larger residual time-frequency model; identical cached/raw preprocessing."""
import torch
from torch import nn
from tf_model import TFPreprocessor, MixStyle


class Residual(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.GroupNorm(8, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout))
        self.skip = nn.Conv2d(cin, cout, 1, stride, bias=False)
    def forward(self, x):
        return nn.functional.silu(self.net(x) + self.skip(x))


class UpgradeModel(nn.Module):
    def __init__(self, mode='residual'):
        super().__init__()
        self.mode = mode
        self.preprocess = TFPreprocessor()
        widths = [32,64,128,192]
        self.blocks = nn.ModuleList([Residual(a,b,2) for a,b in zip([6]+widths[:-1],widths)])
        self.style = MixStyle()
        self.head = nn.Sequential(nn.Linear(192*3,192),nn.SiLU(),nn.Dropout(.5),nn.Linear(192,5))
    def forward_features(self,x,unused=None):
        if self.training:
            x = torch.roll(x,int(torch.randint(-16,17,()).item()),-1)
            x = x * torch.exp(torch.randn(x.shape[0],6,1,1,device=x.device)*.3)
            x = torch.where(torch.rand(x.shape[0],1,1,1,device=x.device)<.5,x.flip(2),x)
            x = x * (torch.rand(x.shape[0],1,1,128,device=x.device)>.1)
            if self.mode == 'residual_domain':
                noise = torch.nn.functional.interpolate(torch.randn(x.shape[0],3,8,device=x.device),size=128,mode='linear',align_corners=False)
                x = torch.cat([x[:,:3]+noise[:,:,None,:],x[:,3:]],1)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i<2 and self.mode != 'residual_plain': x=self.style(x)
        mean=x.mean((2,3))
        std=((x-mean[:,:,None,None]).square().mean((2,3))+1e-5).sqrt()
        return self.head(torch.cat([mean,std,x.amax((2,3))],1))
    def forward(self,frames,intervals):
        return self.forward_features(self.preprocess(frames,intervals))
