"""Fixed CSI statistics, non-neural readout, predeclared three controls."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import joblib
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent/'tree_probe';CACHE=Path('D:/huawei2026-experiments/next053_20261025')
def main():
 W.mkdir(exist_ok=True);rows=json.loads((CACHE/'rows.json').read_text());splits=json.loads(Path('D:/huawei2026-experiments/nested_context_20261107/protocol.json').read_text())['splits'];y=np.array([r['headcount'] for r in rows]);features=[]
 t=np.load(CACHE/'t.npy',mmap_mode='r');s=np.load(CACHE/'s.npy',mmap_mode='r')
 for lo in range(0,len(rows),32):
  z=np.array(t[lo:lo+32,0]);a=z[:,:768].reshape(-1,6,32,4,150).mean(3);p=z[:,768:960].reshape(-1,6,32,150);v=np.concatenate([a,p],axis=1)
  feat=[v.mean(-1),v.std(-1),np.abs(np.diff(v,axis=-1)).mean(-1)]
  for lag in [4,16]: feat.append((v[...,:-lag]*v[...,lag:]).mean(-1)/(np.square(v).mean(-1)+1e-5))
  features.append(np.concatenate([f.reshape(len(v),-1) for f in feat]+[np.array(s[lo:lo+32,0])],axis=1))
 x=np.concatenate(features);assert np.isfinite(x).all();result=[]
 for fold in range(3):
  fi=np.array(splits[f'record{fold}']['source']);te=np.array(splits[f'record{fold}']['test']);keys=[(int(y[i]),rows[i]['identifier'] if rows[i]['room']=='unknown' else rows[i]['room']) for i in fi];counts={k:keys.count(k) for k in set(keys)};domains={c:sum(k[0]==c for k in counts) for c in range(5)};weights=np.array([1/(counts[k]*domains[k[0]]) for k in keys]);weights*=len(fi)/weights.sum()
  for name,model in [('summary_trees',ExtraTreesClassifier(n_estimators=300,max_depth=10,min_samples_leaf=10,max_features=.5,n_jobs=2,random_state=42)),('full_trees',ExtraTreesClassifier(n_estimators=300,max_depth=10,min_samples_leaf=10,max_features=.5,n_jobs=2,random_state=42)),('full_boost',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=42))]:
   xx=x[:,-17:] if name=='summary_trees' else x
   with threadpool_limits(limits=2): model.fit(xx[fi],y[fi],sample_weight=weights);p=model.predict_proba(xx[te])
   pred=p.argmax(1);cm=np.bincount(y[te]*5+pred,minlength=25).reshape(5,5);r=dict(fold=fold,method=name,correct=int(np.trace(cm)),total=len(te),confusion=cm.tolist());result.append(r);print(r,flush=True)
   np.savez_compressed(W/f'{fold}_{name}.npz',indices=te,probabilities=p,labels=y[te]);joblib.dump(model,W/f'{fold}_{name}.joblib')
 (W/'results.json').write_text(json.dumps(dict(features=x.shape[1],protocol='3 record folds; fixed handcrafted features; source class-room weighting; predeclared model settings; no held-out fitting or hyperparameter selection',results=result),indent=2)+'\n',encoding='utf-8')
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
