"""Class-shared antisymmetric candidate comparisons using frozen054 proxy views."""
import sys,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/pair_views_20261102');CACHE=Path('D:/huawei2026-experiments/g60_20261025')

def features(raw,b,T,collapsed=False):
 z=(raw/T+b).reshape(-1,12,5);p=z.softmax(-1)
 if collapsed:z=z.mean(1,keepdim=True).expand(-1,12,-1);p=p.mean(1,keepdim=True).expand(-1,12,-1)
 d=(z[:,:,:,None]-z[:,:,None,:]).clamp(-12,12);mass=p[:,:,:,None]+p[:,:,None,:]
 return torch.stack([d.mean(1),d.std(1,unbiased=False),d.amin(1),d.amax(1),d.median(1).values,d.abs().mean(1),d.tanh().mean(1),mass.mean(1),mass.std(1,unbiased=False)],-1)

class Compare(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(9,24),nn.SiLU(),nn.Linear(24,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):
  a=self.net(x).squeeze(-1);a=(a-a.transpose(1,2))*.5
  return .5*torch.tanh(a.mean(-1))

def main():
 torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
 old.save(WORK/'protocol.json',dict(hypothesis='View disagreement may contain conditional evidence beyond mean scores',variants=['views_protected','mean_protected','views_all','mean_all'],seeds=[42,2026],steps=200,lr=.003,weight_decay=.02,correction_bound=.5,selection='source calibration class-room balanced NLL at steps0,20,50,100,200; target only after selection',architecture='Shared9->24->1 pair scorer, antisymmetric across candidate pairs, mean pair evidence -> bounded correction; no classID or roomID',limits='4 reused room proxy backbones; actual054 all-data weight never used for heldout evaluation. Collapsed view control same parameter allocation but reduced information. A new readout does not establish human causal features.'))
 results=[]
 for room in ('a1','c1','g1','e1'):
  d=torch.load(CACHE/f'{room}.pt',weights_only=True);assert d['checkpoint_sha256']==old.sha(old.checkpoint(room));raw=d['raw'];y=d['labels'];b,T=old.calibration(room);fi,cal,te=old.split(room,d['rows']);base=old.policy(raw,b,T);l,q,p,js=old.components(raw,b,T)
  v0=raw[:,0,0]/T+b;v45=l[:,0];allowed=(v0.argmax(-1)!=1)&(v45.argmax(-1)!=0)&(v45.argmax(-1)!=2);wi=old.balance(d['rows'],y,fi);cw=old.balance(d['rows'],y,cal)
  feature_cache={False:features(raw,b,T),True:features(raw,b,T,True)}
  for variant in ('views_protected','mean_protected','views_all','mean_all'):
   x=feature_cache[variant.startswith('mean')];mask=allowed if variant.endswith('protected') else torch.ones(len(y),dtype=torch.bool)
   for seed in (42,2026):
    torch.manual_seed(seed);m=Compare();opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.02);best=dict(step=0,nll=float((F.cross_entropy(base[cal],y[cal],reduction='none')*cw).sum()),state=copy.deepcopy(m.state_dict()));history=[]
    for step in range(1,201):
     z=base+m(x)*mask[:,None];loss=(F.cross_entropy(z[fi],y[fi],reduction='none')*wi).sum();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
     if step in (20,50,100,200):
      with torch.no_grad():z=base+m(x)*mask[:,None];nll=float((F.cross_entropy(z[cal],y[cal],reduction='none')*cw).sum())
      history.append(dict(step=step,calibration_nll=nll))
      if nll<best['nll']:best=dict(step=step,nll=nll,state=copy.deepcopy(m.state_dict()))
    with torch.no_grad():last=base+m(x)*mask[:,None]
    m.load_state_dict(best['state'])
    with torch.no_grad():z=base+m(x)*mask[:,None]
    if variant.endswith('protected'):assert torch.equal(z[~allowed],base[~allowed])
    key=f'{room}_{variant}_s{seed}';r=dict(room=room,variant=variant,seed=seed,step=best['step'],calibration_nll=best['nll'],test=old.metrics(z,y,base,te),last=old.metrics(last,y,base,te),history=history,parameters=sum(v.numel() for v in m.parameters()));results.append(r)
    torch.save(dict(state=best['state'],step=best['step'],teacher_sha256=d['checkpoint_sha256']),WORK/f'{key}_head.pt');torch.save(dict(logits=z,last_logits=last,base=base,labels=y,rows=d['rows'],test_indices=te),WORK/f'{key}_predictions.pt');old.save(WORK/'results.json',dict(results=results));print(key,best['step'],r['test']['corrected'],r['test']['broken'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
