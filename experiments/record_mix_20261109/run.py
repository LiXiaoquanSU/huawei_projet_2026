"""Same-label latent interpolation: same-record versus cross-record control."""
import sys,importlib.util,random,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('dynamic_stage',ROOT/'experiments/dynamic_evidence_20261108/run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
base=old.base;W=base.BASE/'record_mix_20261109'
class Model(old.Model):
 def features(self,x,s):
  n=len(x);x=x.reshape(n,3,3,32,150);reverse=x*x.new_tensor([1,-1,-1])[None,None,:,None,None];x=torch.stack([x,reverse],2).reshape(-1,3,32,150);z=self.conv(x);z=torch.cat([z.mean(-1),z.std(-1,unbiased=False),z.amax(-1)],1).flatten(1);z=self.project(z).reshape(n,6,192);return torch.cat([z.mean(1),z.std(1,unbiased=False),s],1)
 def forward(self,x,s):return self.head(self.features(x,s)).log_softmax(-1)
def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True)
 rows=base.load(base.CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda');x=torch.from_numpy(np.array(np.load(old.OLD/'joint.npy',mmap_mode='r'))).cuda();x=old.input_view(x,'dynamic');summary=torch.zeros((len(rows),17),device='cuda');splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 # Refactoring must not change inference for the existing baseline weights.
 torch.manual_seed(42);a=Model().cuda().eval();b=old.Model().cuda().eval();b.load_state_dict(a.state_dict())
 with torch.no_grad():error=float((a(x[:4],summary[:4])-b(x[:4],summary[:4])).abs().max())
 assert error==0,error;del a,b;gc.collect();torch.cuda.empty_cache()
 base.save(W/'protocol.json',dict(seed=42,epochs=12,lr=.0003,weight_decay=.01,batch=24,variants=['same_record','cross_record'],control='Reuse dynamic_evidence_20261108 dynamic seed42 fixed12 weights/predictions; identical model initialization and source sampler',mix='Before head: beta(.4,.4) latent interpolation, both examples same headcount; same_record pairs a different window from same original trace; cross_record pairs a different original trace, no room preference.',loss='NLL against unchanged headcount, no auxiliary targets',selection='fixed12; no outer label selection of epoch/lambda/bias',fairness='Two mixing methods have same two-example computation, identical anchor sampler, same model capacity. Pair choice changes only eligibility. Baseline has one-example computation.',splits=splits,refactoring_max_error=error,limitations='Latent regularization is not a claim of physically valid synthetic CSI. Same old development folds, not independent blind test. Fixedseed evidence requires replication before recommendation.'))
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];fitids={rows[i]['identifier'] for i in fi};assert fitids.isdisjoint({rows[i]['identifier'] for i in te});w=base.weights(rows,y.cpu(),fi)
  for mode in ['same_record','cross_record']:
   dest=W/f'{fold}_{mode}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():results.append(base.load(dest/'result.json'));continue
   pools={i:[j for j in fi if j!=i and rows[j]['headcount']==rows[i]['headcount'] and ((rows[j]['identifier']==rows[i]['identifier'])==(mode=='same_record'))] for i in fi};assert all(pools.values())
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);m=Model().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);gen=torch.Generator().manual_seed(42);pair_rng=np.random.default_rng(42009);mix_rng=np.random.default_rng(42010);history=[];pairs=[]
   for epoch in range(1,13):
    m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)];losses=[];epoch_pairs=[]
    for ix in draw.split(24):
     jj=torch.tensor([pair_rng.choice(pools[i]) for i in ix.tolist()]);assert torch.equal(y[ix],y[jj]);lam=torch.from_numpy(mix_rng.beta(.4,.4,size=(len(ix),1)).astype(np.float32)).cuda();z=m.features(x[torch.cat([ix,jj])],summary[torch.cat([ix,jj])]);mixed=lam*z[:len(ix)]+(1-lam)*z[len(ix):];loss=F.cross_entropy(m.head(mixed),y[ix]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()));epoch_pairs.extend(zip(ix.tolist(),jj.tolist()))
    pairs.append(epoch_pairs);history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
    if epoch in [1,6,12]:print('TRAIN',fold,mode,history[-1],flush=True)
   m.eval();parts=[];fitcorrect=0
   with torch.no_grad():
    for lo in range(0,len(fi),24):ii=fi[lo:lo+24];fitcorrect+=int((m(x[ii],summary[ii]).argmax(-1)==y[ii]).sum())
    for lo in range(0,len(te),24):ii=te[lo:lo+24];parts.append(m(x[ii],summary[ii]).cpu())
   z=torch.cat(parts);labels=y[te].cpu();r=dict(fold=fold,mode=mode,history=history,metrics=base.metrics(z,labels),source_fit=dict(correct=fitcorrect,total=len(fi)));base.save(dest/'result.json',r);torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=12,fit_identifiers=sorted(fitids)),dest/'final.pt');torch.save(dict(logits=z,labels=labels,indices=te,training_pairs=pairs),dest/'predictions.pt');results.append(r);base.save(W/'results.json',results);print('RESULT',fold,mode,r['metrics'],flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
