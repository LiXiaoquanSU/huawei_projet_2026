"""Frozen representation audit; all probe selection uses source calibration only."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,confusion_matrix
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from network import EvidenceChain
from train_chain import sample_weights
CACHE=Path('D:/huawei2026-experiments/next053_20261025')
BASE=Path('D:/huawei2026-experiments/texture_chain_20261028')
WORK=Path('D:/huawei2026-experiments/readout_audit_20261030')

def main():
    torch.set_num_threads(4);threadpool_limits(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    WORK.mkdir(parents=True,exist_ok=True)
    protocol=dict(hypothesis='Does late backbone retain count information that its trained head fails to generalize?',folds=3,checkpoints=['selected','last'],layers=['prehead','penultimate'],C=[.001,.01,.1,1],selection='source calibration balanced NLL only',weighting='class-room-record-window same as neural sampler',limits='Reused development folds; probes are diagnostic, not independent validation. Same-record windows stay together. No new submission model.')
    (WORK/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    t=torch.from_numpy(np.array(np.load(CACHE/'t.npy',mmap_mode='r'))).cuda()
    s=torch.from_numpy(np.array(np.load(CACHE/'s.npy',mmap_mode='r'))).cuda()
    results=[];saved={}
    for fold in range(3):
      folder=BASE/f'record{fold}_full_s42'
      data=torch.load(folder/'predictions.pt',weights_only=False,map_location='cpu')
      y=data['labels'].numpy();fi=np.array(data['fit_indices']);cal=np.array(data['calibration_indices']);te=np.array(data['test_indices'])
      w=sample_weights(data['rows'],data['labels'],fi.tolist()).numpy()*len(fi)
      for cp,file in [('selected','final.pt'),('last','last.pt')]:
        m=EvidenceChain().cuda().eval();state=torch.load(folder/file,weights_only=False,map_location='cpu');m.load_state_dict(state['model_state'])
        captured={};features={k:[] for k in ('prehead','penultimate')};outputs=[]
        h1=m.head.register_forward_pre_hook(lambda module,args:captured.update(prehead=args[0].detach()))
        h2=m.head[-1].register_forward_pre_hook(lambda module,args:captured.update(penultimate=args[0].detach()))
        with torch.inference_mode():
          for start in range(0,len(y),24):
            z=m(t[start:start+24].flatten(0,1),s[start:start+24].flatten(0,1))
            outputs.append(z.exp().reshape(-1,2,5).mean(1).cpu())
            for k in features:features[k].append(captured[k].reshape(-1,2,captured[k].shape[-1]).mean(1).cpu().numpy())
        expected=data['logits' if cp=='selected' else 'last_logits'].exp()
        err=float((torch.cat(outputs)-expected).abs().max());assert err<1e-5,err
        h1.remove();h2.remove()
        for layer,parts in features.items():
          x=np.concatenate(parts);sc=StandardScaler().fit(x[fi]);x=sc.transform(x);best=None;trials=[]
          for c in protocol['C']:
            model=LogisticRegression(C=c,max_iter=2000,random_state=42).fit(x[fi],y[fi],sample_weight=w)
            cw=1/np.bincount(y[cal],minlength=5)[y[cal]]
            loss=log_loss(y[cal],model.predict_proba(x[cal]),labels=np.arange(5),sample_weight=cw)
            trials.append(dict(C=c,calibration_nll=float(loss),iterations=model.n_iter_.tolist()))
            if best is None or loss<best[0]:best=(loss,c,model)
          loss,c,model=best;p=model.predict_proba(x[te]);pred=p.argmax(1);key=f'record{fold}_{cp}_{layer}'
          r=dict(fold=fold,checkpoint=cp,epoch=state['epoch'],layer=layer,C=c,calibration_nll=float(loss),correct=int((pred==y[te]).sum()),total=len(te),fit_correct=int((model.predict(x[fi])==y[fi]).sum()),original_correct=int((expected[te].argmax(1).numpy()==y[te]).sum()),confusion=confusion_matrix(y[te],pred,labels=np.arange(5)).tolist(),trials=trials,reproduction_error=err)
          results.append(r);saved[key]=dict(indices=te.tolist(),probabilities=p.tolist())
          print(key,r['correct'],r['original_correct'],'C',c,flush=True)
        del m;torch.cuda.empty_cache()
        (WORK/'results.json').write_text(json.dumps(dict(protocol=protocol,results=results),indent=2),encoding='utf-8')
        (WORK/'predictions.json').write_text(json.dumps(saved),encoding='utf-8')
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
