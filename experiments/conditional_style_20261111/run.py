"""Matched same-count versus unrestricted different-record MixStyle."""
import sys,random,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/nested_context_20261107'));import run as base
W=base.BASE/'conditional_style_20261111'
class MatchedStyle(nn.Module):
 def __init__(self):super().__init__();self.partner=None
 def forward(self,x):
  if not self.training:return x
  mean=x.mean((2,3),keepdim=True);std=((x-mean).square().mean((2,3),keepdim=True)+1e-5).sqrt();mean,std=mean.detach(),std.detach()
  # Consume the same Torch random draws in both arms and the original MixStyle.
  unused=torch.randperm(len(x),device=x.device);weight=torch.distributions.Beta(.2,.2).sample((len(x),1,1,1)).to(x.device);order=self.partner
  assert order is not None and len(order)==len(x)
  return (x-mean)/std*(weight*std+(1-weight)*std[order])+weight*mean+(1-weight)*mean[order]
def partners(ii,rows,rng,variant):
 ids=[rows[i]['identifier'] for i in ii];labels=[rows[i]['headcount'] for i in ii];out=[]
 for j in range(len(ii)):
  same=[k for k in range(len(ii)) if ids[k]!=ids[j] and labels[k]==labels[j]];other=[k for k in range(len(ii)) if ids[k]!=ids[j]];u=rng.random()
  pool=same if variant=='same_class' else other
  out.append(pool[min(int(u*len(pool)),len(pool)-1)] if same else j)
 return torch.tensor(out)

def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True)
 rows=base.load(base.CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda');t=torch.from_numpy(np.array(np.load(base.CACHE/'t.npy',mmap_mode='r')[:,0])).cuda();s=torch.from_numpy(np.array(np.load(base.CACHE/'s.npy',mmap_mode='r')[:,0])).cuda();target=base.signal_targets(t);splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 base.save(W/'protocol.json',dict(architecture='Unchanged original037 spectral physical',seeds=[42],variants=['matched_random','same_class'],epochs=5,batch=32,optimizer='AdamW lr.0004 wd.01 clip3',change='Original losses and architecture. MixStyle partners from different records within batch: same_class requires same count, matched_random any count. Both mix only if same-class/different-record partner exists. Same eligibility mask, Beta strength, per-batch partner random uniforms and Torch random draws. Partner shared across two style layers, independent Beta weights per layer. No change at inference.',rng='Seed before initialization, native DataLoader WeightedRandomSampler float64, no post-initialization seed reset. Exact matched initialization/draw streams across loss arms.',selection='Fixed5. Seed42 first screen; if mean gain >=3pp and >=2/3folds improve over matched_random, confirm with seed7 before considering export. No target checkpoint/bias search; no full fit/export unless robust new evidence.',limits='Reused development record folds; these are not blind tests. Does not use original trained weights or U pseudo labels. Label used only to choose training augmentation partners, never inference. More same-count style overlap does not prove preservation of physical occupancy. Original control is contextual, direct causal contrast is matched_random. Existing folds reused.',splits=splits))
 for fold,split in splits.items():
  fi,te=split['source'],split['test'];fitids=sorted({rows[i]['identifier'] for i in fi});assert not set(fitids)&{rows[i]['identifier'] for i in te};keys=[(int(y[i]),rows[i]['identifier'] if rows[i]['room']=='unknown' else rows[i]['room']) for i in fi];counts={k:keys.count(k) for k in set(keys)};groups={c:sum(k[0]==c for k in counts) for c in range(5)};w=[1/(counts[k]*groups[k[0]]) for k in keys]
  for seed in [42]:
   for variant in ['matched_random','same_class']:
    dest=W/f'{fold}_s{seed}_{variant}';dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists():results.append(base.load(dest/'result.json'));continue
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=base.build('baseline').cuda();m.static.style=MatchedStyle();pair_rng=np.random.default_rng(seed+1111);pair_log=[];opt=torch.optim.AdamW(m.parameters(),lr=.0004,weight_decay=.01);loader=DataLoader(TensorDataset(torch.tensor(fi)),batch_size=32,sampler=WeightedRandomSampler(w,len(fi),replacement=True,generator=torch.Generator().manual_seed(seed)),num_workers=0);history=[];draws=[]
    for epoch in range(1,6):
     m.train();losses=[];epdraw=[]
     for (ii,) in loader:
      partner=partners(ii.tolist(),rows,pair_rng,variant);m.static.style.partner=partner.cuda();pair_log.append(dict(indices=ii.tolist(),partners=partner.tolist()));tasks=m.forward_tasks(t[ii],s[ii]);loss,parts=base.task_loss(tasks,y[ii],target[ii],'physical')
      assert variant in ['matched_random','same_class']
      assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()));epdraw.extend(ii.tolist())
     draws.append(epdraw);history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
     if epoch in [1,5]:print('TRAIN',fold,seed,variant,history[-1],flush=True)
    m.eval();out=[];fitcorrect=0
    with torch.no_grad():
     for start in range(0,len(fi),32):ii=fi[start:start+32];fitcorrect+=int((m.forward_features(t[ii],s[ii]).argmax(-1)==y[ii]).sum())
     for start in range(0,len(te),32):ii=te[start:start+32];out.append(m.forward_features(t[ii],s[ii]).cpu())
    z=torch.cat(out);labels=y[te].cpu();r=dict(fold=fold,seed=seed,variant=variant,metrics=base.metrics(z,labels),source_fit=dict(correct=fitcorrect,total=len(fi)),history=history);torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},fit_identifiers=fitids,seed=seed,variant=variant,epoch=5),dest/'final.pt');torch.save(dict(logits=z,labels=labels,indices=te,draws=draws,training_pairs=pair_log),dest/'predictions.pt');base.save(dest/'result.json',r);results.append(r);base.save(W/'results.json',results);print('RESULT',fold,seed,variant,r['metrics']['correct'],flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
