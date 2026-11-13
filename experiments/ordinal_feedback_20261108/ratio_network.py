"""Matched shared pair encoders for raw bounded ratio versus unit relative phase."""
import sys,random,gc
from pathlib import Path
import numpy as np
import torch,blosc2
from torch import nn
from torch.nn import functional as F
from ratio_probe import transform,base,ROOT
W=base.BASE/'complex_ratio_network_20261108'
class Model(nn.Module):
 def __init__(self):
  super().__init__();layers=[]
  for a,b in [(3,32),(32,64),(64,96)]:layers.extend([nn.Conv2d(a,b,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,b),nn.SiLU()])
  self.conv=nn.Sequential(*layers);self.project=nn.Sequential(nn.Linear(1152,192),nn.LayerNorm(192),nn.SiLU());self.head=nn.Sequential(nn.Linear(401,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.3),nn.Linear(128,5))
 def forward(self,x,s):
  n=len(x);x=x.reshape(n,3,3,32,150);reverse=x*x.new_tensor([1,-1,-1])[None,None,:,None,None];x=torch.stack([x,reverse],2).reshape(-1,3,32,150);z=self.conv(x);z=torch.cat([z.mean(-1),z.std(-1,unbiased=False),z.amax(-1)],1).flatten(1);z=self.project(z).reshape(n,6,192);v=torch.cat([z.mean(1),z.std(1,unbiased=False),s],1);return self.head(v).log_softmax(-1)
def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True);rows=base.load(base.CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda');splits=base.load(base.WORK/'protocol.json')['splits']
 if not (W/'features_complete.json').exists():
  j=np.lib.format.open_memmap(W/'joint.npy',mode='w+',dtype='float32',shape=(960,9,32,150));p=np.lib.format.open_memmap(W/'phase.npy',mode='w+',dtype='float32',shape=(960,9,32,150))
  for ident in sorted({r['identifier'] for r in rows}):
   raw=blosc2.load_array(f'D:/huawei2026-data/labeled/CSI_{ident}.b2');ix=[i for i,r in enumerate(rows) if r['identifier']==ident]
   for lo in range(0,len(ix),4):
    ii=ix[lo:lo+4];h=torch.from_numpy(np.stack([raw[int(rows[i]['start']):int(rows[i]['stop'])].reshape(600,3,1024) for i in ii])).cuda()
    with torch.no_grad():a,b=transform(h,maps=True);b=b.reshape(-1,3,2,32,150);b=torch.cat([b,torch.zeros_like(b[:,:,:1])],2).flatten(1,2);j[ii]=a.cpu().numpy();p[ii]=b.cpu().numpy()
    del h
   print('CACHE',ident,flush=True);del raw;gc.collect()
  j.flush();p.flush();del j,p;base.save(W/'features_complete.json',dict(rows=rows,shape=[960,9,32,150]))
 assert base.load(W/'features_complete.json')['rows']==rows
 s0=torch.from_numpy(np.array(np.load(base.CACHE/'s.npy',mmap_mode='r')[:,0])).cuda();results=[]
 base.save(W/'protocol.json',dict(variants=['phase','joint'],architecture='same shared3channel32/64/96conv pair encoder; explicit conjugate/reverse orientation; pooled mean/std over6orientedpairs; original17summary,128hidden,5classes',seed=42,epochs=12,lr=.0003,weight_decay=.01,batch=24,selection='fixed12 only; no target checkpoint selection; frozen protocol before results',input='raw-window per-subcarrier joint bounded coordinates or unit-relative-phase zero-padded to3channels; time150,freq32,absolute joint geometry retained',summary='standardized on training records only',source='https://arxiv.org/abs/1907.03994',limitations='Motivation from CSI ratio, not reproduction of FarSense respiration algorithm; no breathing/person decomposition claims. Reused record folds.',splits=splits))
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];w=base.weights(rows,y.cpu(),fi);s=((s0-s0[fi].mean(0))/s0[fi].std(0,unbiased=False).clamp_min(.001)).clamp(-8,8)
  for variant in ['phase','joint']:
   dest=W/f'{fold}_{variant}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():results.append(base.load(dest/'result.json'));continue
   x=torch.from_numpy(np.array(np.load(W/f'{variant}.npy',mmap_mode='r'))).cuda();random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);m=Model().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);gen=torch.Generator().manual_seed(42);history=[]
   for epoch in range(1,13):
    m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)];losses=[]
    for ix in draw.split(24):
     z=m(x[ix],s[ix]);loss=F.nll_loss(z,y[ix]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
    history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
    if epoch in [1,6,12]:print('TRAIN',fold,variant,history[-1],flush=True)
   m.eval();out=[]
   with torch.no_grad():
    for lo in range(0,len(te),24):ii=te[lo:lo+24];out.append(m(x[ii],s[ii]).cpu())
   z=torch.cat(out);labels=y[te].cpu();r=dict(fold=fold,variant=variant,parameters=sum(p.numel() for p in m.parameters()),history=history,metrics=base.metrics(z,labels));torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},summary_mean=s0[fi].mean(0).cpu(),summary_std=s0[fi].std(0,unbiased=False).clamp_min(.001).cpu(),fit_identifiers=sorted({rows[i]['identifier'] for i in fi}),epoch=12),dest/'final.pt');torch.save(dict(logits=z,labels=labels,indices=te),dest/'predictions.pt');base.save(dest/'result.json',r);results.append(r);base.save(W/'results.json',results);print('RESULT',fold,variant,r['metrics'],flush=True);del m,opt,x;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
