"""One versus four learned modes per count on identical source-only frozen representations."""
import sys,copy,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from train_chain import old,sample_weights,metric
BASE=Path('D:/huawei2026-experiments/texture_chain_20261028');FEATURES=Path('D:/huawei2026-experiments/distribution_pool_20261030');WORK=Path('D:/huawei2026-experiments/count_modes_20261030')

class CountModes(nn.Module):
 def __init__(self,k):
  super().__init__();self.k=k;self.encoder=nn.Sequential(nn.Linear(1664,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.3));self.prototypes=nn.Parameter(torch.randn(5,k,128)*.1)
 def components(self,x):return 10*torch.einsum('bd,ckd->bck',F.normalize(self.encoder(x),dim=-1),F.normalize(self.prototypes,dim=-1))
 def forward(self,x):return (torch.logsumexp(self.components(x),dim=-1)-math.log(self.k)).log_softmax(-1)

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;WORK.mkdir(parents=True,exist_ok=True)
 old.save(WORK/'protocol.json',dict(hypothesis='Same count may have multiple signal states; a single feature center may be inadequate',modes=[1,4],seeds=[42,2026],folds=3,epochs=60,lr=.001,weight_decay=.02,temperature=.1,selection='source calibration balanced NLL only; fixed60 also saved',representation='fold-specific source-selected frozen early texture CNN prehead1664',limits='Learned modes are not people or supervised activities. Heads vary seeds, backbones do not. Four-mode has1920 additional prototype parameters. Reused dev splits.'))
 results=[]
 for fold in range(3):
  d=torch.load(BASE/f'record{fold}_full_s42/predictions.pt',weights_only=False,map_location='cpu');fi,cal,te=[d[k] for k in ('fit_indices','calibration_indices','test_indices')];raw=np.load(FEATURES/f'record{fold}_features.npz')['old'];mean=raw[fi].mean(0);std=raw[fi].std(0).clip(.001);x=torch.tensor((raw-mean)/std,device='cuda');y=d['labels'].cuda();w=sample_weights(d['rows'],d['labels'],fi)
  for k in (1,4):
   for seed in (42,2026):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=CountModes(k).cuda();gen=torch.Generator().manual_seed(seed);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.02);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,60,eta_min=.0001);best=None;history=[]
    for epoch in range(1,61):
     m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)]
     for j in draw.split(32):
      loss=F.nll_loss(m(x[j]),y[j]);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
     sched.step();m.eval()
     with torch.no_grad():z=m(x[cal]);yy=y[cal];cw=1/torch.bincount(yy,minlength=5)[yy].float();cw/=cw.sum();nll=float((-z[range(len(cal)),yy]*cw).sum())
     history.append(dict(epoch=epoch,calibration_nll=nll))
     if best is None or nll<best['nll']:best=dict(epoch=epoch,nll=nll,state=copy.deepcopy({a:b.detach().cpu() for a,b in m.state_dict().items()}))
    with torch.no_grad():last=m(x).cpu()
    m.load_state_dict(best['state'])
    with torch.no_grad():z=m(x).cpu();component=m.components(x[fi]).argmax(-1).cpu();occupancy=[torch.bincount(component[d['labels'][fi]==c,c],minlength=k).tolist() for c in range(5)]
    key=f'record{fold}_k{k}_s{seed}';r=dict(fold=fold,k=k,seed=seed,epoch=best['epoch'],calibration_nll=best['nll'],test=metric(z[te],d['labels'][te]),last_test=metric(last[te],d['labels'][te]),fit_mode_occupancy=occupancy,history=history);results.append(r)
    torch.save(dict(model_state=best['state'],mean=mean,std=std,epoch=best['epoch']),WORK/f'{key}_final.pt');torch.save(dict(logits=z,last_logits=last,test_indices=te),WORK/f'{key}_predictions.pt');old.save(WORK/'results.json',dict(results=results));print('DONE',key,r['epoch'],r['test']['correct'],r['last_test']['correct'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
