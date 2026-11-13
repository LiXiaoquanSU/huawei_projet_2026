"""Frozen texture evidence + bounded spectral correction, matched interaction controls."""
import sys,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from train_chain import old,sample_weights,metric
BASE=Path('D:/huawei2026-experiments/texture_chain_20261028');CACHE=Path('D:/huawei2026-experiments/distribution_pool_20261030');SPEC=Path('D:/huawei2026-experiments/timed_spectrum_20261030/features.npz');WORK=Path('D:/huawei2026-experiments/context_spectrum_20261031')

class Correction(nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant
  self.context=nn.Sequential(nn.Linear(1664,128),nn.LayerNorm(128),nn.SiLU())
  self.spectral=nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,64,3,stride=(1,2),padding=1),nn.GroupNorm(8,64),nn.SiLU())
  self.head=nn.Sequential(nn.Linear(384,64),nn.SiLU(),nn.Dropout(.3),nn.Linear(64,5));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,x,s,base):
  if self.variant=='zero':s=s*0
  c=self.context(x);z=self.spectral(s);v=torch.cat([z.mean((2,3)),z.flatten(2).std(-1,unbiased=False)],-1);product=c.tanh()*v.tanh()
  if self.variant=='no_product':product=product*0
  delta=.5*torch.tanh(self.head(torch.cat([c,v,product],-1)))
  return (base+delta).log_softmax(-1)

def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;WORK.mkdir(parents=True,exist_ok=True)
 old.save(WORK/'protocol.json',dict(variants=['product','no_product','zero'],seeds=[42,2026],epochs=30,lr=.0005,weight_decay=.02,correction_bound=.5,selection='source calibration balanced NLL including original epoch0; fixed30 preserved',data='Same source-only frozen early texture network per fold; fit-only summary scaling; spectrum input per-window',hypothesis='Can bounded condition-dependent spectral evidence help without drifting texture backbone?',limits='Reused development folds. no_product still permits implicit interaction through MLP. Equal allocated parameters not equal effective capacity. New seeds change correction only. No full-data or platform model.'))
 spectrum=torch.tensor(np.load(SPEC)['actual'].reshape(-1,4,7,32),device='cuda');results=[]
 for fold in range(3):
  d=torch.load(BASE/f'record{fold}_full_s42/predictions.pt',weights_only=False,map_location='cpu');fi,cal,te=[d[k] for k in ('fit_indices','calibration_indices','test_indices')];raw=np.load(CACHE/f'record{fold}_features.npz')['old'];mean=raw[fi].mean(0);std=raw[fi].std(0).clip(.001);x=torch.tensor((raw-mean)/std,device='cuda');base=d['logits'].cuda();y=d['labels'].cuda();w=sample_weights(d['rows'],d['labels'],fi)
  for variant in ('product','no_product','zero'):
   for seed in (42,2026):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=Correction(variant).cuda();gen=torch.Generator().manual_seed(seed);opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=.02);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00005)
    def predict(ix):
     m.eval()
     with torch.no_grad():return torch.cat([m(x[j],spectrum[j],base[j]).cpu() for j in [ix[k:k+64] for k in range(0,len(ix),64)]])
    def cal_loss():
     z=predict(cal);yy=y[cal].cpu();cw=1/torch.bincount(yy,minlength=5)[yy].float();cw/=cw.sum();return float((-z[range(len(cal)),yy]*cw).sum())
    assert float((predict(cal).exp()-d['logits'][cal].exp()).abs().max())<1e-6
    best=dict(epoch=0,nll=cal_loss(),state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()}));history=[]
    for epoch in range(1,31):
     m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)]
     for j in draw.split(32):
      loss=F.nll_loss(m(x[j],spectrum[j],base[j]),y[j]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
     sched.step();nll=cal_loss();history.append(dict(epoch=epoch,calibration_nll=nll))
     if nll<best['nll']:best=dict(epoch=epoch,nll=nll,state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()}))
    last=predict(list(range(len(raw))));m.load_state_dict(best['state']);z=predict(list(range(len(raw))));key=f'record{fold}_{variant}_s{seed}';r=dict(fold=fold,variant=variant,seed=seed,epoch=best['epoch'],calibration_nll=best['nll'],test=metric(z[te],d['labels'][te]),last_test=metric(last[te],d['labels'][te]),parameters=sum(v.numel() for v in m.parameters()),history=history);results.append(r)
    torch.save(dict(model_state=best['state'],mean=mean,std=std,epoch=best['epoch']),WORK/f'{key}_final.pt');torch.save(dict(logits=z,last_logits=last,test_indices=te),WORK/f'{key}_predictions.pt');old.save(WORK/'results.json',dict(results=results));print('DONE',key,r['epoch'],r['test']['correct'],r['last_test']['correct'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
