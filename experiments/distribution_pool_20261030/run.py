"""Source-only local pattern dictionary versus moment pooling on identical frozen CNNs."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from train_chain import EvidenceChain,sample_weights,old,CACHE
BASE=Path('D:/huawei2026-experiments/texture_chain_20261028');WORK=Path('D:/huawei2026-experiments/distribution_pool_20261030')

def dictionary_features(x,fi,seed):
    # x: windows, frequency, unordered receiver/direction/time/view tokens, channels32
    rng=np.random.default_rng(seed);train=x[fi].reshape(-1,32);train=train[rng.choice(len(train),min(24000,len(train)),replace=False)]
    scaler=StandardScaler().fit(train);train=scaler.transform(train)
    km=MiniBatchKMeans(n_clusters=8,random_state=seed,n_init=3,batch_size=1024,max_iter=100).fit(train)
    centers=km.cluster_centers_;distance=((train[:,None,:]-centers[None,:,:])**2).mean(-1)
    temp=max(float(np.median(distance.min(-1))),1e-3)
    dist=[];mom=[]
    for start in range(0,len(x),24):
        z=scaler.transform(x[start:start+24].reshape(-1,32)).reshape(x[start:start+24].shape)
        residual=z[:,:,:,None,:]-centers[None,None,None,:,:]
        scores=-(residual**2).mean(-1)/temp;scores-=scores.max(-1,keepdims=True);a=np.exp(scores);a/=a.sum(-1,keepdims=True)
        hist=a.mean(2);v=(a[...,None]*residual).mean(2)
        dist.append(np.concatenate([hist[...,None],v],-1).reshape(len(z),-1))
        mom.append(np.concatenate([z.mean(2),z.std(2)],-1).reshape(len(z),-1))
    return np.concatenate(dist),np.concatenate(mom),dict(centers=centers,mean=scaler.mean_,scale=scaler.scale_,temperature=temp)

def main():
    torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    WORK.mkdir(parents=True,exist_ok=True)
    old.save(WORK/'protocol.json',dict(hypothesis='Does local pattern distribution preserve count evidence lost by moment pooling?',folds=3,backbone='source-selected early frozen texture CNN, same per fold',projection='fixed seed30 orthonormal 256 to32',dictionary='8 source-fit-only local prototypes per amplitude/phase branch; frequency position retained; receiver/time/view aggregated',modes=['old','moments','distribution','old_plus_distribution'],C=[.001,.01,.1,1],selection='source calibration balanced NLL',limitations='Repeated development folds. Prototypes are signal patterns, not people. Dictionary training uses fit records only; no target scaling or centroids. No platform result.',code_sha256=old.sha(Path(__file__))))
    t=torch.from_numpy(np.array(np.load(CACHE/'t.npy',mmap_mode='r'))).cuda();s=torch.from_numpy(np.array(np.load(CACHE/'s.npy',mmap_mode='r'))).cuda()
    projection=np.linalg.qr(np.random.default_rng(30).normal(size=(256,32)))[0].astype('float32');proj=torch.tensor(projection,device='cuda')
    results=[];predictions={}
    for fold in range(3):
        folder=BASE/f'record{fold}_full_s42';d=torch.load(folder/'predictions.pt',weights_only=False,map_location='cpu');state=torch.load(folder/'final.pt',weights_only=False,map_location='cpu')
        y=d['labels'].numpy();fi=np.array(d['fit_indices']);cal=np.array(d['calibration_indices']);te=np.array(d['test_indices'])
        m=EvidenceChain().cuda().eval();m.load_state_dict(state['model_state']);capture={};features={k:[] for k in ('amp','phase','old')};outputs=[]
        hooks=[m.amp.conv.register_forward_hook(lambda mod,args,out:capture.update(amp=out.detach())),m.phase.conv.register_forward_hook(lambda mod,args,out:capture.update(phase=out.detach())),m.head.register_forward_pre_hook(lambda mod,args:capture.update(old=args[0].detach()))]
        with torch.inference_mode():
            for start in range(0,len(y),24):
                batch=min(24,len(y)-start);z=m(t[start:start+batch].flatten(0,1),s[start:start+batch].flatten(0,1));outputs.append(z.exp().reshape(-1,2,5).mean(1).cpu())
                features['old'].append(capture['old'].reshape(batch,2,-1).mean(1).cpu().numpy())
                for branch,n in [('amp',3),('phase',6)]:
                    v=capture[branch];_,channels,freq,time=v.shape
                    v=(v.permute(0,2,3,1)@proj).reshape(batch,2,n,freq,time,32).permute(0,3,1,2,4,5).reshape(batch,freq,-1,32)
                    features[branch].append(v.cpu().numpy())
        err=float((torch.cat(outputs)-d['logits'].exp()).abs().max());assert err<1e-5,err
        for h in hooks:h.remove()
        del m;torch.cuda.empty_cache()
        oldx=np.concatenate(features['old']);ds=[];ms=[];artifacts={}
        for branch in ('amp','phase'):
            dx,mx,artifact=dictionary_features(np.concatenate(features[branch]),fi,30);ds.append(dx);ms.append(mx)
            for k,v in artifact.items():artifacts[branch+'_'+k]=v
        modes=dict(old=oldx,moments=np.concatenate(ms,1),distribution=np.concatenate(ds,1),old_plus_distribution=np.concatenate([oldx,*ds],1))
        np.savez_compressed(WORK/f'record{fold}_features.npz',**modes,projection=projection,**artifacts)
        w=sample_weights(d['rows'],d['labels'],fi.tolist()).numpy()*len(fi)
        for mode,x in modes.items():
            scaler=StandardScaler().fit(x[fi]);x=scaler.transform(x);best=None;trials=[]
            for c in (.001,.01,.1,1):
                model=LogisticRegression(C=c,max_iter=2000,random_state=42).fit(x[fi],y[fi],sample_weight=w)
                cw=1/np.bincount(y[cal],minlength=5)[y[cal]];loss=log_loss(y[cal],model.predict_proba(x[cal]),labels=np.arange(5),sample_weight=cw)
                trials.append(dict(C=c,calibration_nll=float(loss),iterations=model.n_iter_.tolist()))
                if best is None or loss<best[0]:best=(loss,c,model)
            loss,c,model=best;p=model.predict_proba(x[te]);key=f'record{fold}_{mode}'
            results.append(dict(fold=fold,mode=mode,dimensions=x.shape[1],C=c,calibration_nll=float(loss),correct=int((p.argmax(1)==y[te]).sum()),total=len(te),confusion=confusion_matrix(y[te],p.argmax(1),labels=np.arange(5)).tolist(),trials=trials,reproduction_error=err))
            predictions[key]=dict(indices=te.tolist(),probabilities=p.tolist())
            np.savez_compressed(WORK/f'{key}_classifier.npz',coefficient=model.coef_,intercept=model.intercept_,mean=scaler.mean_,scale=scaler.scale_)
            print(key,results[-1]['correct'],'C',c,flush=True)
        old.save(WORK/'results.json',dict(results=results));old.save(WORK/'predictions.json',predictions)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
