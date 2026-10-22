"""Additional raw-CSI antenna-ratio ablation; no inferred/synthetic antennas."""
import argparse
import gc
from pathlib import Path
import blosc2
import numpy as np
import torch
from torch import nn
import run
from net import ROOT,MotionModel,SeriesEncoder

class RatioPreprocessor(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,frames,intervals):
        t,s=self.base(frames,intervals)
        power=frames[:,:,0].square().sum(-1)
        valid=power>1e-8
        logamp=.5*torch.log1p(power)
        features=[]
        for i,j in [(0,1),(0,2),(1,2)]:
            mask=(valid[:,:,i]&valid[:,:,j]).to(frames.dtype)
            # Difference of log magnitudes is a stabilized amplitude ratio.
            # Keep a bounded value when either side is weak; no direct division.
            value=((logamp[:,:,i]-logamp[:,:,j]).clamp(-4,4)*mask).reshape(-1,600,32,32)
            weight=mask.reshape(-1,600,32,32).sum(-1)
            pooled=value.sum(-1)/weight.clamp_min(1)
            pooled=pooled.reshape(-1,150,4,32).mean(2).transpose(1,2)
            centered=pooled-pooled.mean(-1,keepdim=True)
            features.append(centered)
        return t,torch.cat([s,torch.cat(features,1).flatten(1)],1)

class RatioModel(MotionModel):
    def __init__(self):
        super().__init__('baseline')
        self.preprocess=RatioPreprocessor(self.preprocess)
        self.ratio_encoder=SeriesEncoder()
        self.ratio_fuse=nn.Linear(64,256)
        nn.init.normal_(self.ratio_fuse.weight,std=.001);nn.init.zeros_(self.ratio_fuse.bias)
    def refine_latent(self,z,t,summary):
        vector=summary[:,17:].reshape(-1,96,150)
        return z+self.ratio_fuse(self.ratio_encoder(vector))

def fresh_ratio(seed):torch.manual_seed(seed);return RatioModel()

def prepare(d):
    p=run.WORK/'ratio_summary.pt'
    if p.exists():return torch.load(p,map_location='cpu',weights_only=True)
    model=fresh_ratio(42).eval();parts=[]
    for identifier in dict.fromkeys(r['identifier'] for r in d['rows']):
        rows=[r for r in d['rows'] if r['identifier']==identifier]
        csi=blosc2.load_array(str(Path('D:/huawei2026-data/labeled')/f'CSI_{identifier}.b2'))
        ati=blosc2.load_array(str(Path('D:/huawei2026-data/labeled')/f'ATI_{identifier}.b2'))
        for start in range(0,len(rows),4):
            rr=rows[start:start+4]
            x=np.stack([csi[int(r['start']):int(r['stop'])].view(np.float32).reshape(600,1,3,1024,2) for r in rr])
            dt=np.stack([ati[int(r['start']):int(r['stop'])].copy() for r in rr]);dt[:,-1]=0
            with torch.no_grad():_,s=model.preprocess(torch.from_numpy(x).double(),torch.from_numpy(dt).double())
            parts.append(s[:,17:].float())
        print('RATIO_FEATURES',identifier,flush=True)
        del csi,ati;gc.collect()
    extra=torch.cat(parts);assert extra.shape==(960,14400) and torch.isfinite(extra).all()
    torch.save(extra,p);return extra

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);d=run.load_data();extra=prepare(d)
    if not a.prepare_only:
        d['s']=torch.cat([d['s'],extra],1)
        run.fresh=lambda variant,seed:fresh_ratio(seed)
        torch.backends.cudnn.deterministic=True
        for fold in run.FOLDS:run.fit(d,'ratio',42,fold)
        run.save(run.WORK/'ratio_summary.json',run.summary('ratio',42))
