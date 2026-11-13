"""Spectral pattern CNN; actual versus uniform timestamps, record and room validation."""
import sys,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from train_chain import old,split,sample_weights,metric,CACHE
WORK=Path('D:/huawei2026-experiments/timed_spectrum_20261030')

class SpectralNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.conv=nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,64,3,stride=(1,2),padding=1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU())
  self.head=nn.Sequential(nn.Linear(269,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.3),nn.Linear(128,5))
 def forward(self,x,s):
  z=self.conv(x);z=z.reshape(len(z),64,7,2,8);v=torch.cat([z.mean((2,4)),z.std((2,4),unbiased=False)],1).flatten(1)
  return self.head(torch.cat([v,s],1)).log_softmax(-1)

def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dest=WORK/'neural';dest.mkdir(exist_ok=True);rows=old.read(CACHE/'rows.json');y=torch.tensor([v['headcount'] for v in rows],device='cuda');raw=np.load(CACHE/'s.npy')[:,0];rs=raw[:,:12].reshape(-1,4,3);ss=np.concatenate([rs.mean(-1),rs.std(-1),raw[:,12:]],1);features=np.load(WORK/'features.npz')
 old.save(dest/'protocol.json',dict(variants=['actual','uniform'],seeds=[42,2026],epochs=30,lr=.0005,weight_decay=.02,batch=32,architecture='4x7x32 spectrum -> Conv32 Conv64 Conv64 -> two frequency-region mean/std -> summary13 ->128 ->5',selection='source calibration balanced NLL, fixed30 also saved',data='48 labeled records, same record and room splits; source-only summary normalization',limits='Repeated development folds; seed repeats whole spectral network. Spectrum loses temporal order and retains finite-window limitations. No platform labels or U.'))
 results=[]
 for fold in ('record0','record1','record2','a1','c1','g1','e1'):
  fi,cal,te=split(rows,fold);mean=ss[fi].mean(0);std=ss[fi].std(0).clip(.001);s=torch.tensor((ss-mean)/std,device='cuda',dtype=torch.float32);w=sample_weights(rows,y.cpu(),fi)
  for mode in ('actual','uniform'):
   x=torch.tensor(features[mode].reshape(-1,4,7,32),device='cuda')
   for seed in (42,2026):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);gen=torch.Generator().manual_seed(seed);m=SpectralNet().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=.02);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00005);best=None;history=[]
    def predict(ix):
     m.eval()
     with torch.no_grad():return torch.cat([m(x[j],s[j]).cpu() for j in [ix[k:k+64] for k in range(0,len(ix),64)]])
    for epoch in range(1,31):
     m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)];losses=[]
     for j in draw.split(32):
      loss=nn.functional.nll_loss(m(x[j],s[j]),y[j]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
     sched.step();z=predict(cal);yy=y[cal].cpu();cw=1/torch.bincount(yy,minlength=5)[yy].float();cw/=cw.sum();nll=float((-z[range(len(cal)),yy]*cw).sum());history.append(dict(epoch=epoch,loss=float(np.mean(losses)),calibration_nll=nll))
     if best is None or nll<best['nll']:best=dict(epoch=epoch,nll=nll,state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()}))
    key=f'{fold}_{mode}_s{seed}';last=predict(list(range(len(rows))));torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},mean=mean,std=std,epoch=30),dest/f'{key}_last.pt');m.load_state_dict(best['state']);z=predict(list(range(len(rows))));torch.save(dict(model_state=best['state'],mean=mean,std=std,epoch=best['epoch']),dest/f'{key}_final.pt');torch.save(dict(logits=z,last_logits=last,fit_indices=fi,calibration_indices=cal,test_indices=te),dest/f'{key}_predictions.pt')
    r=dict(fold=fold,mode=mode,seed=seed,epoch=best['epoch'],calibration_nll=best['nll'],parameters=sum(v.numel() for v in m.parameters()),test=metric(z[te],y[te].cpu()),last_test=metric(last[te],y[te].cpu()),fit=metric(z[fi],y[fi].cpu()),history=history);results.append(r);old.save(dest/'results.json',dict(results=results));print('DONE',key,r['epoch'],r['test']['correct'],r['last_test']['correct'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
