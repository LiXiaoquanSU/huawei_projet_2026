"""Test measurable correspondence before prescribing a synchrony auxiliary task."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,confusion_matrix,roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/synchrony_20261029');CACHE=Path('D:/huawei2026-experiments/next053_20261025')
SPECTRAL='--spectral' in sys.argv
if SPECTRAL:WORK=WORK/'spectral_control'
FOLDS=('record0','record1','record2','a1','c1','g1','e1');PAIRS=((0,1),(0,2),(1,2));CS=(.01,.1,1.)

def corr(x,y):
    x=x-x.mean(-1,keepdims=True);y=y-y.mean(-1,keepdims=True)
    return (x*y).mean(-1)/np.sqrt((x*x).mean(-1)*(y*y).mean(-1)+1e-8)

def features(a,p,partner,condition,mode):
    # Match32 frequency bands; both positive and negative features use120 samples.
    result=[]
    for width in (1,5,15):
        pairs=[]
        for k,(u,v) in enumerate(PAIRS):
            if mode=='rx':x=a[:,u];y=a[:,v]
            else:
                x=(np.abs(np.diff(a[:,u],axis=-1,prepend=a[:,u,:,:1]))+np.abs(np.diff(a[:,v],axis=-1,prepend=a[:,v,:,:1])))*.5
                y=np.sqrt(np.square(np.diff(p[:,k],axis=-1,prepend=p[:,k,:,:,:1])).sum(1)+1e-8)
            if condition=='other_window':y=y[partner]
            offset=30 if condition=='shift' else 0
            # Start at1 to exclude the artificial first derivative sample.
            x=x[:,:,1:121];y=y[:,:,1+offset:121+offset] if offset==0 else y[:,:,30:150]
            x=x.reshape(len(a),32,120//width,width).mean(-1);y=y.reshape(len(a),32,120//width,width).mean(-1)
            r=corr(x,y);en=corr(np.abs(x).mean(1),np.abs(y).mean(1))
            pairs.append(np.stack([r.mean(1),np.abs(r).mean(1),r.std(1),en],1))
        z=np.stack(pairs,1);result.append(np.concatenate([z.mean(1),z.std(1),z.min(1),z.max(1)],1))
    return np.concatenate(result,1).astype(np.float32)

def split(rows,fold):
    if not fold.startswith('record'):return old.split(fold,rows)
    r=old.read(Path('D:/huawei2026-experiments/detail_features_20261027/probe_results.json'))['folds'][fold]
    return tuple([i for i,v in enumerate(rows) if v['identifier'] in set(r[k])] for k in ('fit_records','calibration_records','test_records'))

def fit_predict(x,y,fi,cal,te,nclasses):
    chosen=None
    for c in CS:
        m=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=1500,random_state=42))
        m.fit(x[fi],y[fi]);p=m.predict_proba(x[cal]);w=1/np.bincount(y[cal],minlength=nclasses)[y[cal]]
        loss=log_loss(y[cal],p,labels=np.arange(nclasses),sample_weight=w)
        if chosen is None or loss<chosen[0]:chosen=(loss,c,m)
    loss,c,m=chosen;p=m.predict_proba(x[te]);return p,dict(C=c,calibration_balanced_nll=float(loss),iterations=int(m[-1].n_iter_.max()))

def main():
    WORK.mkdir(parents=True,exist_ok=True);rows=old.read(CACHE/'rows.json');n=len(rows);y=np.array([r['headcount'] for r in rows]);partner=[]
    for i,r in enumerate(rows):
        same=[j for j,v in enumerate(rows) if v['identifier']==r['identifier']];partner.append(same[(same.index(i)+7)%len(same)])
    partner=np.array(partner);t=np.load(CACHE/'t.npy',mmap_mode='r');a=np.array(t[:,0,384:768]).reshape(n,3,32,4,150).mean(3);p=np.array(t[:,0,768:960]).reshape(n,3,2,32,150)
    if SPECTRAL:a=a-a.mean(2,keepdims=True)
    fs={f'{mode}_{condition}':features(a,p,partner,condition,mode) for mode in ('rx','amp_phase') for condition in ('sync','shift','other_window')}
    s=np.load(CACHE/'s.npy')[:,0];v=s[:,:12].reshape(n,4,3);summary=np.concatenate([v.mean(-1),v.std(-1),s[:,12:17]],1)
    np.savez_compressed(WORK/'features.npz',**fs,summary=summary,labels=y,partner=partner)
    outputs=[];predictions={};manifests={}
    for fold in FOLDS:
        fi,cal,te=split(rows,fold);fi=np.array(fi);cal=np.array(cal);te=np.array(te)
        sets=[{rows[i]['identifier'] for i in z} for z in (fi,cal,te)];assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        assert all(rows[i]['identifier']==rows[partner[i]]['identifier'] for i in te)
        manifests[fold]=dict(fit_records=sorted(sets[0]),calibration_records=sorted(sets[1]),test_records=sorted(sets[2]))
        for mode in ('rx','amp_phase'):
          for negative in ('shift','other_window'):
            x=np.concatenate([fs[f'{mode}_sync'],fs[f'{mode}_{negative}']]);target=np.r_[np.ones(n,dtype=int),np.zeros(n,dtype=int)]
            ix=lambda z:np.r_[z,z+n]
            prob,meta=fit_predict(x,target,ix(fi),ix(cal),ix(te),2);pred=prob.argmax(1);yy=target[ix(te)]
            r=dict(task='synchrony',fold=fold,mode=mode,negative=negative,total=len(yy),correct=int((pred==yy).sum()),auc=float(roc_auc_score(yy,prob[:,1])),per_count={str(k):dict(total=int((y[te]==k).sum()*2),correct=int(((pred==yy)&np.tile(y[te]==k,2)).sum())) for k in range(5)},**meta)
            outputs.append(r);predictions[f'{fold}_{mode}_{negative}']=dict(indices=te.tolist(),probabilities=prob.tolist())
        relation=np.concatenate([fs['rx_sync'],fs['amp_phase_sync']],1)
        for name,x in [('summary',summary),('relation',relation),('combined',np.concatenate([summary,relation],1))]:
            prob,meta=fit_predict(x,y,fi,cal,te,5);pred=prob.argmax(1)
            outputs.append(dict(task='count',fold=fold,mode=name,total=len(te),correct=int((pred==y[te]).sum()),confusion=confusion_matrix(y[te],pred,labels=np.arange(5)).tolist(),**meta))
            predictions[f'{fold}_count_{name}']=dict(indices=te.tolist(),probabilities=prob.tolist())
        print('DONE',fold,[(r['mode'],r.get('negative',''),r['correct'],r['total']) for r in outputs if r['fold']==fold],flush=True)
    old.save(WORK/'results.json',dict(results=outputs,manifests=manifests,rows=rows,protocol=dict(feature_dim_per_mode=48,C=list(CS),seed=42,source_calibration_only=True,negative_partner_offset_windows=7,temporal_shift_cached_samples=29,subtract_per_receiver_frequency_common=SPECTRAL,preprocessing='audited cached features only, same frequency bands and view0',limitations='All folds reused development sets. Synchrony can arise from hardware/common interference, not necessarily people. Other-window negatives stay within recording; no negative crosses train/test. Correlation features do not establish causal physical identifiability. Shift uses crops not circular wrapping. Removing frequency-common amplitude also removes possible human-induced broadband changes; it is not hardware isolation.',hashes={k:old.sha(CACHE/k) for k in ('rows.json','t.npy','s.npy')},code_sha256=old.sha(Path(__file__)))))
    old.save(WORK/'predictions.json',predictions)

if __name__=='__main__':main()
