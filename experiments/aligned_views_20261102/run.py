"""Early aligned joint-RX convolution versus RX-permutation augmentation."""
import sys,copy,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from train_chain import old,split,sample_weights,metric,CACHE
from network import Block
WORK=Path('D:/huawei2026-experiments/aligned_views_20261102')

class Aligned(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(Block(12,32),Block(32,64),Block(64,128));self.head=nn.Sequential(nn.Linear(1549,256),nn.LayerNorm(256),nn.SiLU(),nn.Dropout(.3),nn.Linear(256,5))
 def forward(self,t,s):
  amp=t[:,:768].reshape(-1,6,128,150);phase=t[:,768:960].reshape(-1,6,32,150).repeat_interleave(4,2)
  z=self.conv(F.avg_pool2d(torch.cat([amp,phase],1),(1,2)));z=z.reshape(len(z),128,4,4,z.shape[-1]).mean(3);mean=z.mean(-1);std=(z.var(-1,unbiased=False)+1e-5).sqrt();v=torch.cat([mean,std,z.amax(-1)],1).flatten(1);a=s[:,:12].reshape(-1,4,3);summary=torch.cat([a.mean(-1),a.std(-1,unbiased=False),s[:,12:17]],-1)
  return self.head(torch.cat([v,summary],-1)).log_softmax(-1)

def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;WORK.mkdir(parents=True,exist_ok=True)
 old.save(WORK/'protocol.json',dict(hypothesis='Preserve early aligned RX/pair interactions, then learn permutation robustness instead of removing RX assignment before interactions',variants=['identity_train','permuted_train'],seed=42,epochs=12,lr=.0003,weight_decay=.02,batch=24,selection='source calibration balanced NLL separately for identity and6RX inference; both use time reversal average; fixed12 preserved',architecture='joint12channel amplitude/phase -> shared residual CNN32/64/128 -> four frequency-region mean/std/max ->256 ->5',limitations='Phase32bins repeated to128 does not restore frequency detail. Same parameter count and initial weights across training variants; permutation augmentation only difference. Reused record folds; room tests require separate training.'))
 rows=old.read(CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda');t=torch.from_numpy(np.array(np.load(CACHE/'t.npy',mmap_mode='r'))).cuda();s=torch.from_numpy(np.array(np.load(CACHE/'s.npy',mmap_mode='r'))).cuda();perms=[tuple(v.cuda() for v in old.permutation_indices(p)) for p in old.PERMS];results=[]
 for fold in ('record0','record1','record2'):
  fi,cal,te=split(rows,fold);w=sample_weights(rows,y.cpu(),fi)
  for variant in ('identity_train','permuted_train'):
   torch.manual_seed(42);torch.cuda.manual_seed_all(42);m=Aligned().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.02);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,12,eta_min=.00003);gen=torch.Generator().manual_seed(42);rng=np.random.default_rng(42);best={};history=[]
   def predict(indices,views):
    m.eval();out=[]
    with torch.no_grad():
     for start in range(0,len(indices),24):
      jj=indices[start:start+24];tt=t[jj].flatten(0,1);ss=s[jj].flatten(0,1);pp=[]
      for ix,sg,si in perms[:views]:pp.append(m(tt.index_select(1,ix)*sg,ss.index_select(1,si)).exp().reshape(-1,2,5).mean(1))
      out.append(torch.stack(pp).mean(0).clamp_min(1e-12).log().cpu())
    return torch.cat(out)
   for epoch in range(1,13):
    m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)]
    for jj in draw.split(24):
     tt=t[jj].flatten(0,1);ss=s[jj].flatten(0,1)
     if variant=='permuted_train':
      ix,sg,si=perms[int(rng.integers(6))];tt=tt.index_select(1,ix)*sg;ss=ss.index_select(1,si)
     z=m(tt,ss);p=z.exp().reshape(-1,2,5);q=p.mean(1,keepdim=True);loss=F.nll_loss(z,y[jj,None].expand(-1,2).flatten())+.05*(p*(p.clamp_min(1e-8).log()-q.clamp_min(1e-8).log())).sum(-1).mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
    sched.step();entry={'epoch':epoch}
    for views in (1,6):
     z=predict(cal,views);yy=y[cal].cpu();cw=1/torch.bincount(yy,minlength=5)[yy].float();cw/=cw.sum();nll=float((-z[range(len(cal)),yy]*cw).sum());entry[f'nll_v{views}']=nll
     if views not in best or nll<best[views]['nll']:best[views]=dict(epoch=epoch,nll=nll,state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()}))
    history.append(entry)
    if epoch in (1,6,12):print('TRAIN',fold,variant,entry,flush=True)
   laststate=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()})
   for views in (1,6):
    m.load_state_dict(laststate);last=predict(list(range(len(rows))),views);m.load_state_dict(best[views]['state']);z=predict(list(range(len(rows))),views);key=f'{fold}_{variant}_v{views}';r=dict(fold=fold,variant=variant,views=views,epoch=best[views]['epoch'],calibration_nll=best[views]['nll'],test=metric(z[te],y[te].cpu()),last_test=metric(last[te],y[te].cpu()),parameters=sum(p.numel() for p in m.parameters()),history=history);results.append(r);torch.save(dict(model_state=best[views]['state'],epoch=best[views]['epoch']),WORK/f'{key}_final.pt');torch.save(dict(logits=z,last_logits=last,test_indices=te),WORK/f'{key}_predictions.pt');old.save(WORK/'results.json',dict(results=results));print('DONE',key,r['epoch'],r['test']['correct'],r['last_test']['correct'],flush=True)
   torch.save(dict(model_state=laststate,epoch=12),WORK/f'{fold}_{variant}_last.pt');del m,opt;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
