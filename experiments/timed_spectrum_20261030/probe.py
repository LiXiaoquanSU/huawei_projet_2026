"""Two protocol families, source-selected linear and nonlinear spectral probes."""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from train_chain import old,split,sample_weights,CACHE
WORK=Path('D:/huawei2026-experiments/timed_spectrum_20261030')

def main():
 torch.set_num_threads(4);threadpool_limits(4);rows=old.read(CACHE/'rows.json');y=np.array([r['headcount'] for r in rows]);s=np.load(CACHE/'s.npy')[:,0];ss=s[:,:12].reshape(-1,4,3);base=np.concatenate([ss.mean(-1),ss.std(-1),s[:,12:]],1)
 data=np.load(WORK/'features.npz');modes={'summary':base,**{k:np.concatenate([base,data[k]],1) for k in ('actual','uniform','coarse')}};results=[];predictions={}
 for fold in ('record0','record1','record2','a1','c1','g1','e1'):
  fi,cal,te=[np.array(v) for v in split(rows,fold)];w=sample_weights(rows,torch.tensor(y),fi.tolist()).numpy()*len(fi);cw=1/np.bincount(y[cal],minlength=5)[y[cal]]
  for mode,raw in modes.items():
   scaler=StandardScaler().fit(raw[fi]);x=scaler.transform(raw)
   for family in ('linear','rbf'):
    transform=RBFSampler(gamma=1/x.shape[1],n_components=512,random_state=42) if family=='rbf' else None
    xx=transform.fit(x[fi]).transform(x) if transform else x
    best=None;trials=[]
    for c in (.001,.01,.1,1):
     model=LogisticRegression(C=c,max_iter=2000,random_state=42).fit(xx[fi],y[fi],sample_weight=w);loss=log_loss(y[cal],model.predict_proba(xx[cal]),labels=np.arange(5),sample_weight=cw);trials.append(dict(C=c,calibration_nll=float(loss),iterations=model.n_iter_.tolist()))
     if best is None or loss<best[0]:best=(loss,c,model)
    loss,c,model=best;p=model.predict_proba(xx[te]);key=f'{fold}_{mode}_{family}'
    results.append(dict(fold=fold,mode=mode,family=family,C=c,calibration_nll=float(loss),correct=int((p.argmax(1)==y[te]).sum()),total=len(te),confusion=confusion_matrix(y[te],p.argmax(1),labels=np.arange(5)).tolist(),trials=trials))
    predictions[key]=dict(indices=te.tolist(),probabilities=p.tolist())
  old.save(WORK/'probe_results.json',dict(results=results,fit_count=224,selection='source calibration balanced NLL, same splits and record-aware weights',limitations='RBF random map fixed by seed/dimension only; no target-fitted preprocessing. Reused development folds, not independent test.'))
  old.save(WORK/'predictions.json',predictions);print(fold,[(r['mode'],r['family'],r['correct']) for r in results if r['fold']==fold],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
