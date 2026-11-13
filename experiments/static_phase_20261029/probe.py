"""Source-calibrated linear/nonlinear probes, same declared seven folds."""
import sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/synchrony_20261029'))
from run import old,split,FOLDS,CACHE
WORK=Path('D:/huawei2026-experiments/static_phase_20261029')
def main():
    threadpool_limits(4)
    rows=old.read(CACHE/'rows.json');y=np.array([r['headcount'] for r in rows]);c=np.load(WORK/'features.npz')['static_phase'];s=np.load(Path('D:/huawei2026-experiments/synchrony_20261029/features.npz'))['summary'];results=[];predictions={}
    for fold in FOLDS:
        fi,cal,te=[np.array(x) for x in split(rows,fold)]
        for mode,x in [('summary',s),('curvature',c),('combined',np.concatenate([s,c],1))]:
          for model in ('linear','rbf'):
            best=None
            for strength in (.01,.1,1.):
                steps=[StandardScaler()]
                if model=='rbf':steps.append(RBFSampler(gamma=1/x.shape[1],n_components=512,random_state=42))
                steps.append(LogisticRegression(C=strength,class_weight='balanced',max_iter=1500,random_state=42));m=make_pipeline(*steps);m.fit(x[fi],y[fi]);p=m.predict_proba(x[cal]);w=1/np.bincount(y[cal],minlength=5)[y[cal]]
                loss=log_loss(y[cal],p,labels=np.arange(5),sample_weight=w)
                if best is None or loss<best[0]:best=(loss,strength,m)
            loss,strength,m=best;p=m.predict_proba(x[te]);pred=p.argmax(1);key=f'{fold}_{mode}_{model}'
            results.append(dict(fold=fold,mode=mode,model=model,C=strength,calibration_nll=float(loss),correct=int((pred==y[te]).sum()),total=len(te),confusion=confusion_matrix(y[te],pred,labels=np.arange(5)).tolist()))
            predictions[key]=dict(indices=te.tolist(),probabilities=p.tolist(),fit_records=sorted({rows[i]['identifier'] for i in fi}),calibration_records=sorted({rows[i]['identifier'] for i in cal}),test_records=sorted({rows[i]['identifier'] for i in te}))
        print('DONE',fold,[(r['mode'],r['model'],r['correct']) for r in results if r['fold']==fold],flush=True)
    old.save(WORK/'probe_results.json',dict(results=results,rows=rows,fit_count=126,code_sha256=old.sha(Path(__file__)),limits='Repeated development folds, source-only scaling/C selection; random Fourier features512dim gamma1/inputdim seed42 fixed before results. Linear probe failure does not rule out neural use. No full-data submission model trained.'))
    old.save(WORK/'predictions.json',predictions)
if __name__=='__main__':main()
