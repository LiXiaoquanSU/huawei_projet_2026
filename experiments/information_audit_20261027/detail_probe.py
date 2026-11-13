"""Matched linear probes on whole-record and whole-room splits, source-only C selection."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from detail_features import ROOT,WORK
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
def main():
    torch.set_num_threads(4);rows=old.read(WORK/'rows.json');y=np.array([r['headcount'] for r in rows]);ids=np.array([r['identifier'] for r in rows]);d=np.load(WORK/'features.npz');s=np.load('D:/huawei2026-experiments/next053_20261025/s.npy')[:,0]
    # Symmetric summary across receiver order; timing retained.
    ss=s[:,:12].reshape(-1,4,3);base=np.concatenate([ss.mean(-1),ss.std(-1),s[:,12:]],1)
    variants={'summary':base,'summary_time':np.concatenate([base,d['temporal']],1),'summary_phase':np.concatenate([base,d['phase']],1),'summary_both':np.concatenate([base,d['temporal'],d['phase']],1)}
    splits={r:old.split(r,rows) for r in ('a1','c1','g1','e1')}
    for k,(source,test) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=20261027).split(base,y,ids)):
        chosen=[]
        for c in range(5):
            options=sorted(set(ids[source[y[source]==c]]),key=lambda s:hashlib.sha256(('detail_cal:'+s).encode()).hexdigest());assert len(options)>=2;chosen.append(options[0])
        cal=[i for i in source if ids[i] in chosen];fit=[i for i in source if ids[i] not in chosen];splits[f'record{k}']=(fit,cal,list(test))
    result={'protocol':'4 reused heldout-room development folds and 3 stratified group(record) folds. Same fit/cal/test for all variants. C selected on source calibration only. These are lightweight feature probes, not platform candidates. No old all-data weights used.','folds':{}}
    for name,(fit,cal,test) in splits.items():
        assert not set(ids[fit])&set(ids[cal]);assert not set(ids[fit])&set(ids[test]);assert not set(ids[cal])&set(ids[test]);assert len(set(y[fit]))==5
        w=old.balance(rows,torch.tensor(y),fit).numpy()*len(fit);cw=1/np.bincount(y[cal],minlength=5)[y[cal]];cw/=cw.sum()
        r={'fit_records':sorted(set(ids[fit])),'calibration_records':sorted(set(ids[cal])),'test_records':sorted(set(ids[test])),'variants':{}}
        for variant,x in variants.items():
            scaler=StandardScaler().fit(x[fit]);xx=scaler.transform(x);best=None;trials=[]
            for C in (.01,.1,1.):
                m=LogisticRegression(C=C,max_iter=3000).fit(xx[fit],y[fit],sample_weight=w);p=m.predict_proba(xx);nll=float((-np.log(p[cal,y[cal]].clip(1e-12))*cw).sum());trials.append(dict(C=C,nll=nll,iterations=int(m.n_iter_.max())))
                if best is None or nll<best[0]:best=(nll,C,p,m)
            nll,C,p,m=best;pred=p.argmax(1);cm=np.bincount(y[test]*5+pred[test],minlength=25).reshape(5,5)
            r['variants'][variant]=dict(C=C,trials=trials,correct=int(cm.trace()),total=len(test),confusion=cm.tolist(),records=[dict(identifier=ident,label=int(y[np.where(ids==ident)[0][0]]),correct=int((pred[jj]==y[jj]).sum()),total=len(jj)) for ident in sorted(set(ids[test])) if (jj:=[i for i in test if ids[i]==ident])])
            np.savez_compressed(WORK/f'{name}_{variant}.npz',probabilities=p,test_indices=test,coef=m.coef_,intercept=m.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
        result['folds'][name]=r;old.save(WORK/'probe_results.json',result);print(name,{k:(v['correct'],v['total']) for k,v in r['variants'].items()},flush=True)
    for group,folds in [('room',('a1','c1','g1','e1')),('record',('record0','record1','record2'))]:
        result[group]={v:dict(correct=sum(result['folds'][f]['variants'][v]['correct'] for f in folds),total=sum(result['folds'][f]['variants'][v]['total'] for f in folds)) for v in variants}
    old.save(WORK/'probe_results.json',result);print(json.dumps({k:result[k] for k in ('room','record')}),flush=True)
if __name__=='__main__':main()
