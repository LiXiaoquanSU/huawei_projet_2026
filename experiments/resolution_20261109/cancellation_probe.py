"""Same-size fixed probes: retain per-carrier variation before band averaging."""
import importlib.util,gc,argparse
from pathlib import Path
import numpy as np
import torch,blosc2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
spec=importlib.util.spec_from_file_location('resolution_cache_stage',Path(__file__).with_name('cache.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
base=c.base;W=c.W/'cancellation_probe'
def extract(h):
 power=h.abs().square();before=[];after=[]
 for u,v in [(0,1),(0,2),(1,2)]:
  a,b=power[:,:,u],power[:,:,v];mask=((a>1e-8)&(b>1e-8)).float();den=(a+b).clamp_min(1e-8);cross=h[:,:,u]*h[:,:,v].conj();q=torch.stack([2*cross.real/den,2*cross.imag/den,(a-b)/den],2)
  # Only carriers valid throughout the entire window; no transition artifacts.
  valid=mask.amin(1).reshape(len(h),32,32);denom=valid.sum(-1).clamp_min(1);q=q.reshape(len(h),600,3,32,32);pooled=(q*valid[:,None,None]).sum(-1)/denom[:,None,None]
  qmean=q.mean(1,keepdim=True);pmean=pooled.mean(1,keepdim=True)
  aa=[];bb=[]
  for dq,dp in [(q-qmean,pooled-pmean),(q[:,1:]-q[:,:-1],pooled[:,1:]-pooled[:,:-1]),(q[:,4:]-q[:,:-4],pooled[:,4:]-pooled[:,:-4])]:
   aa.append((dq.square().sum(2).mean(1)*valid).sum(-1)/denom)
   bb.append(dp.square().sum(2).mean(1))
  before.append(torch.stack(aa,1));after.append(torch.stack(bb,1))
 return [torch.log1p(1000*torch.cat(v,1).clamp_min(0).sqrt()).flatten(1) for v in [before,after]]
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--device',default='cuda');args=pa.parse_args();torch.set_num_threads(2);threadpool_limits(2);W.mkdir(exist_ok=True);rows=base.load(base.CACHE/'rows.json');y=np.array([r['headcount'] for r in rows]);cache=W/'features.npz'
 base.save(W/'protocol.json',dict(features='Three rotationally invariant energies per pair/band: temporal variance, lag1 and lag4 squared increments. Compare square before vs after band averaging. Native600frames,32bands; total288features. Carriers valid throughout each window only. log1p(1000*sqrt(energy)).',model='source-only StandardScaler, LogisticRegression C=.01 fixed, class-room sample weighting; no label-dependent preprocessing',hypothesis='Opposite signed carrier changes cancel when averaging before nonlinear energy extraction; no assertion that activity magnitude determines count',limitations='Activity/noise can both increase energy. Reused development record splits, not blind.',splits=base.load(base.WORK/'protocol.json')['splits']))
 if not cache.exists():
  x=[np.zeros((960,288),np.float32) for _ in range(2)]
  for ident in sorted({r['identifier'] for r in rows}):
   raw=blosc2.load_array(f'D:/huawei2026-data/labeled/CSI_{ident}.b2');ix=[i for i,r in enumerate(rows) if r['identifier']==ident]
   for lo in range(0,len(ix),4):
    ii=ix[lo:lo+4];h=torch.from_numpy(np.stack([raw[int(rows[i]['start']):int(rows[i]['stop'])].reshape(600,3,1024) for i in ii])).to(args.device)
    with torch.no_grad():z=extract(h)
    for a,b in zip(x,z):a[ii]=b.cpu().numpy()
    del h,z
   del raw;gc.collect();print('ENERGY_CACHE',ident,flush=True)
  assert all(np.isfinite(z).all() for z in x);assert np.min(x[0]-x[1])>-1e-4;np.savez_compressed(cache,before=x[0],after=x[1])
 data=np.load(cache);results=[]
 for fold,split in base.load(base.WORK/'protocol.json')['splits'].items():
  fi,te=split['source'],split['test'];w=base.weights(rows,torch.from_numpy(y),fi).numpy()*len(fi)
  for mode in ['before','after']:
   x=data[mode];scale=StandardScaler().fit(x[fi]);model=LogisticRegression(C=.01,max_iter=2000,random_state=42).fit(scale.transform(x[fi]),y[fi],sample_weight=w);p=model.predict_proba(scale.transform(x[te]));r=dict(fold=fold,mode=mode,metrics=base.metrics(torch.tensor(np.log(p.clip(1e-12))),torch.tensor(y[te])),iterations=model.n_iter_.tolist());results.append(r);np.savez_compressed(W/f'{fold}_{mode}.npz',probabilities=p,indices=te,labels=y[te]);base.save(W/'results.json',results);print('PROBE_RESULT',fold,mode,r['metrics']['correct'],flush=True)
 print('PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
