"""Source-calibrated physical presence branch; preserve conditional count distribution."""
from run import *

def main():
    torch.set_num_threads(4);result={}
    for room in ('a1','c1','g1','e1'):
        d=torch.load(EVID/f'{room}.pt',weights_only=True); rows=d['rows']; y=d['labels'].numpy();fit,cal,test=old.split(room,rows)
        x=d['s'].mean((1,2)).numpy();scaler=StandardScaler().fit(x[fit]);x=scaler.transform(x)
        w=old.balance(rows,torch.tensor(y),fit).numpy()*len(fit)
        cw=1/np.bincount(y[cal],minlength=5)[y[cal]];cw/=cw.sum()
        best=None;trials=[]
        for C in (.01,.1,1.,10.):
            model=LogisticRegression(C=C,max_iter=2000,random_state=2026).fit(x[fit],(y[fit]>0).astype(int),sample_weight=w)
            p=model.predict_proba(x)[:,0];nll=float((-np.log(np.where(y[cal]==0,p[cal],1-p[cal]).clip(1e-12))*cw).sum())
            trials.append(dict(C=C,source_nll=nll))
            if best is None or nll<best[0]:best=(nll,C,p,model)
        nll,C,p,model=best;b,T=old.calibration(room)
        new=torch.load(STRENGTH/f'{room}_strength_s2026/predictions.pt',weights_only=True)
        bases={'old054_proxy':old.policy(d['raw'],b,T).softmax(-1).numpy(),'strength':new['logits'].exp().numpy()}
        r={'presence_C':C,'presence_trials':trials,'bases':{}}
        for name,base in bases.items():
            candidates=[]; probs=[]
            for alpha in (0.,.25,.5,.75,1.):
                # Blend empty probability only. Relative probabilities of 1--4 unchanged.
                p0=(1-alpha)*base[:,0]+alpha*p
                q=np.concatenate([p0[:,None],base[:,1:]/base[:,1:].sum(1,keepdims=True)*(1-p0[:,None])],1)
                nll=float((-np.log(q[cal,y[cal]].clip(1e-12))*cw).sum());probs.append(q)
                candidates.append(dict(alpha=alpha,source_nll=nll))
            k=int(np.argmin([c['source_nll'] for c in candidates]));q=probs[k]
            r['bases'][name]=dict(selected_alpha=candidates[k]['alpha'],selection=candidates,
                baseline=stats(base,y,test),test=stats(q,y,test),calibration=stats(q,y,cal))
            np.savez_compressed(WORK/f'{room}_{name}_presence.npz',probabilities=q,labels=y,test_indices=test,alpha=candidates[k]['alpha'],coef=model.coef_,intercept=model.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
        result[room]=r;print(room,json.dumps(r['bases']),flush=True)
    old.save(WORK/'presence_results.json',result)

if __name__=='__main__':main()
