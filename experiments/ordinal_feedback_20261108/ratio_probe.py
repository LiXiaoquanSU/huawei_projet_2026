"""Bounded pairwise complex-ratio coordinates; fixed record-disjoint probes."""
import json,sys,gc
from pathlib import Path
import numpy as np
import torch,blosc2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/nested_context_20261107'));import run as base
W=base.BASE/'complex_ratio_probe_20261108'
def transform(h,maps=False):
 # h: B,T,3,F complex64. Pooled bounded stereographic coordinates of hi/hj.
 power=h.abs().square();out=[];phase=[]
 for u,v in [(0,1),(0,2),(1,2)]:
  a,b=power[:,:,u],power[:,:,v];mask=((a>1e-8)&(b>1e-8)).float();den=(a+b).clamp_min(1e-8);cross=h[:,:,u]*h[:,:,v].conj();coords=torch.stack([2*cross.real/den,2*cross.imag/den,(a-b)/den],2)
  unit=cross/(a*b).sqrt().clamp_min(1e-8);unit=torch.stack([unit.real,unit.imag],2)
  def pool(x):
   q=(x*mask[:,:,None]).reshape(len(h),600,x.shape[2],32,32).sum(-1)/mask.reshape(len(h),600,32,32).sum(-1).clamp_min(1)[:,:,None]
   return q.reshape(len(h),150,4,x.shape[2],32).mean(2).permute(0,2,3,1)
  out.append(pool(coords));phase.append(pool(unit))
 if maps:return torch.cat(out,1),torch.cat(phase,1)
 def stats(x):return torch.cat([x.mean(-1),x.std(-1,unbiased=False),(x[...,1:]-x[...,:-1]).abs().mean(-1),(x[...,4:]-x[...,:-4]).abs().mean(-1)],1).flatten(1)
 return stats(torch.cat(out,1)),stats(torch.cat(phase,1))
def main():
 torch.set_num_threads(3);threadpool_limits(2);W.mkdir(exist_ok=True);rows=base.load(base.CACHE/'rows.json');y=np.array([r['headcount'] for r in rows]);cache=W/'features.npz'
 base.save(W/'protocol.json',dict(source='https://arxiv.org/abs/1907.03994',hypothesis='Joint relative amplitude and phase in bounded pair coordinates, before frequency/time pooling. FarSense concerns respiration; no claim of occupancy performance or literal person components.',formula='[2Re(hi*conj(hj)),2Im(hi*conj(hj)),|hi|²-|hj|²]/(|hi|²+|hj|²), valid-carrier masking; pair01/02/12',features='32 frequency bands,150 time points; per-coordinate temporal mean/std and mean abs lag1/4 increments. Concatenate original17 summaries.',controls=['summary','phase_plus_summary','joint_plus_summary'],classifier='StandardScaler source-fit only, multinomial LogisticRegression fixed C=.01,max_iter2000,class-room weights',split='same three reused whole-record folds, no target tuning',limitations='Fixed descriptors may miss temporal detail; common-gain cancellation requires shared multiplicative nuisance and valid carriers. Threshold and noise affect practical invariance. No estimate of individual breathing or number of independent paths.'))
 if not cache.exists():
  joint=np.zeros((len(rows),1152),np.float32);phase=np.zeros((len(rows),768),np.float32);invariance=None
  for ident in sorted({r['identifier'] for r in rows}):
   raw=blosc2.load_array(f'D:/huawei2026-data/labeled/CSI_{ident}.b2');ix=[i for i,r in enumerate(rows) if r['identifier']==ident]
   for start in range(0,len(ix),4):
    jj=ix[start:start+4];h=torch.from_numpy(np.stack([raw[int(rows[i]['start']):int(rows[i]['stop'])].reshape(600,3,1024) for i in jj])).cuda()
    with torch.no_grad():
     a,b=transform(h);joint[jj]=a.cpu().numpy();phase[jj]=b.cpu().numpy()
     if invariance is None:
      torch.manual_seed(42);gain=(.7+.6*torch.rand(len(h),600,1,1,device='cuda'))*torch.exp(1j*torch.rand(len(h),600,1,1,device='cuda'));aa,bb=transform(h*gain);invariance=dict(joint_max_error=float((a-aa).abs().max()),phase_max_error=float((b-bb).abs().max()));assert max(invariance.values())<1e-4,invariance
    del h
   print('FEATURES',ident,flush=True);del raw;gc.collect()
  assert np.isfinite(joint).all() and np.isfinite(phase).all();np.savez_compressed(cache,joint=joint,phase=phase);base.save(W/'invariance.json',invariance)
 d=np.load(cache);summary=np.array(np.load(base.CACHE/'s.npy',mmap_mode='r')[:,0]);xs={'summary':summary,'phase_plus_summary':np.concatenate([d['phase'],summary],1),'joint_plus_summary':np.concatenate([d['joint'],summary],1)};splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];weights=base.weights(rows,torch.from_numpy(y),fi).numpy()*len(fi)
  for name,x in xs.items():
   scaler=StandardScaler().fit(x[fi]);xx=scaler.transform(x);model=LogisticRegression(C=.01,max_iter=2000,random_state=42).fit(xx[fi],y[fi],sample_weight=weights);p=model.predict_proba(xx[te]);metrics=base.metrics(torch.from_numpy(np.log(p.clip(1e-12))),torch.from_numpy(y[te]));r=dict(fold=fold,variant=name,metrics=metrics,iterations=model.n_iter_.tolist());results.append(r);np.savez_compressed(W/f'{fold}_{name}.npz',probabilities=p,indices=te,labels=y[te]);print('RESULT',fold,name,metrics['correct'],flush=True);base.save(W/'results.json',results)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
