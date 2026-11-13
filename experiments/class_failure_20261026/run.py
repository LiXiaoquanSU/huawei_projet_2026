"""Audit frozen-fold failures; source-only linear probes, no target-label tuning."""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/class_failure_20261026')
EVID=Path('D:/huawei2026-experiments/subnet_expectation_20261025/evidence')
STRENGTH=Path('D:/huawei2026-experiments/strength_ablation_20261026')

def stats(z,y,ix):
    z=np.asarray(z)[ix]; yy=np.asarray(y)[ix]; p=z.argmax(1)
    cm=np.bincount(yy*5+p,minlength=25).reshape(5,5)
    rank=(z>z[np.arange(len(yy)),yy,None]).sum(1)+1
    return dict(correct=int((p==yy).sum()),total=len(ix),confusion=cm.tolist(),
      true_class_mean_rank=[float(rank[yy==c].mean()) if (yy==c).any() else None for c in range(5)])

def main():
    torch.set_num_threads(4); WORK.mkdir(parents=True,exist_ok=True)
    result={'protocol':'Four reused leave-room development folds. Linear probe regularization selected on five source calibration records only. No hidden labels.', 'rooms':{}}
    for room in ('a1','c1','g1','e1'):
        d=torch.load(EVID/f'{room}.pt',weights_only=True); rows=d['rows']; y=d['labels'].numpy(); fit,cal,test=old.split(room,rows)
        b,T=old.calibration(room); raw=d['raw']; l,q,p,js=old.components(raw,b,T)
        scores={'uncalibrated_identity':raw[:,0,0], 'uncalibrated_group':raw.exp().mean((1,2)).log(),
                'calibrated_identity':raw[:,0,0]/T+b,'calibrated_group':q.mean(1).log(), 'protected054_proxy':old.policy(raw,b,T)}
        new=torch.load(STRENGTH/f'{room}_strength_s2026/predictions.pt',weights_only=True)
        assert new['rows']==rows
        scores['new_strength_joint']=new['logits']
        z=new['logits'].clone(); nonempty=z[:,0].exp()<.5; z[nonempty,0]=-1e9; z[~nonempty,1:]=-1e9
        scores['new_strength_hard']=z
        r={'bias':b.tolist(),'temperature':T,'splits':{},'methods':{},'probes':{}}
        for name,ix in [('fit',fit),('calibration',cal),('test',test)]:
            r['splits'][name]=[dict(label=c,windows=int((y[ix]==c).sum()),records=len({rows[i]['identifier'] for i in ix if y[i]==c}),rooms=sorted({rows[i]['room'] for i in ix if y[i]==c})) for c in range(5)]
        for name,z in scores.items():
            r['methods'][name]={part:stats(z.numpy(),y,ix) for part,ix in [('fit',fit),('calibration',cal),('test',test)]}
        # Class and room balanced fit weights, exactly as existing source protocol.
        w=old.balance(rows,torch.tensor(y),fit).numpy()*len(fit)
        cw=1/np.bincount(y[cal],minlength=5)[y[cal]]; cw=cw/cw.sum()
        emb=d['e'].mean((1,2)).reshape(len(y),-1).numpy()
        # Summary statistics are receiver-order invariant after averaging all permutations.
        summary=d['s'].mean((1,2)).numpy()
        for name,x in [('embedding',emb),('summary',summary),('embedding_summary',np.concatenate([emb,summary],1))]:
            scaler=StandardScaler().fit(x[fit]); xx=scaler.transform(x); best=None; trials=[]
            for C in (.01,.1,1.,10.):
                model=LogisticRegression(C=C,max_iter=2000,solver='lbfgs',random_state=2026)
                model.fit(xx[fit],y[fit],sample_weight=w)
                prob=model.predict_proba(xx); nll=float((-np.log(prob[cal,y[cal]].clip(1e-12))*cw).sum())
                trials.append(dict(C=C,source_calibration_nll=nll,iterations=int(model.n_iter_.max())))
                if best is None or nll<best[0]: best=(nll,C,prob,model)
            nll,C,prob,model=best
            r['probes'][name]=dict(C=C,trials=trials,metrics={part:stats(prob,y,ix) for part,ix in [('fit',fit),('calibration',cal),('test',test)]})
            np.savez_compressed(WORK/f'{room}_{name}.npz',probabilities=prob,labels=y,test_indices=test,fit_indices=fit,calibration_indices=cal,coef=model.coef_,intercept=model.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
        result['rooms'][room]=r
        old.save(WORK/'results.json',result)
        print(room, json.dumps({k:v['test']['correct'] for k,v in r['methods'].items()}),json.dumps({k:v['metrics']['test']['correct'] for k,v in r['probes'].items()}),flush=True)
    aggregate={}
    for category in ('methods','probes'):
        for name in result['rooms']['a1'][category]:
            ms=[r[category][name]['test'] if category=='methods' else r[category][name]['metrics']['test'] for r in result['rooms'].values()]
            cm=np.array([m['confusion'] for m in ms]).sum(0)
            aggregate[name]=dict(correct=int(cm.trace()),total=int(cm.sum()),confusion=cm.tolist(),recall=(cm.diagonal()/cm.sum(1)).tolist())
    result['aggregate']=aggregate;old.save(WORK/'results.json',result); print(json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
