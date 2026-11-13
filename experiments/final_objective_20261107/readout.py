"""Fixed regularized readout of frozen context representation; no target fitting."""
import sys,gc
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import joblib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/nested_context_20261107'));import run as base
W=base.BASE/'context_readout_20261107'
def main():
 torch.set_num_threads(4);threadpool_limits(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True)
 rows=base.load(base.CACHE/'rows.json');y=np.array([r['headcount'] for r in rows]);t=torch.from_numpy(np.array(np.load(base.CACHE/'t.npy',mmap_mode='r')[:,0])).cuda();s=torch.from_numpy(np.array(np.load(base.CACHE/'s.npy',mmap_mode='r')[:,0])).cuda();splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 base.save(W/'protocol.json',dict(architecture='frozen context_events epoch5 seed42',readout='multinomial logistic regression C=0.01 fixed; standardized on source only; class-room sample weights; no intercept penalty from sklearn',layers=['latent256','routes1024'],evaluation='same3 held-record folds; no hyperparameter search; no target labels for fitting. Existing backbone also excludes test records.',splits=splits))
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];dest=W/fold;dest.mkdir(exist_ok=True)
  if (dest/'results.json').exists():results.extend(base.load(dest/'results.json'));continue
  m=base.build('context_events').cuda().eval();cp=torch.load(base.WORK/f'{fold}_context_events_outer/epoch5.pt',map_location='cpu',weights_only=False);assert not set(cp['fit_identifiers'])&{rows[i]['identifier'] for i in te};m.load_state_dict(cp['model_state']);captured={}
  h1=m.classifier.register_forward_pre_hook(lambda module,args:captured.update(latent256=args[0].detach()))
  h2=m.shared.register_forward_pre_hook(lambda module,args:captured.update(routes1024=args[0].detach()))
  h3=m.classifier.register_forward_hook(lambda module,args,output:captured.update(pure_logits=output.detach().log_softmax(-1)))
  feats={k:[] for k in ['latent256','routes1024']};parts={k:[] for k in ['logits','direct','ordinal','pure_logits']}
  with torch.inference_mode():
   for start in range(0,len(rows),32):
    tasks=m.forward_tasks(t[start:start+32],s[start:start+32])
    for k in feats:feats[k].append(captured[k].cpu().numpy())
    for k in parts:parts[k].append((captured[k] if k=='pure_logits' else tasks[k]).cpu())
  h1.remove();h2.remove();h3.remove();w=base.weights(rows,torch.from_numpy(y),fi).numpy()*len(fi);out=[]
  for name,pieces in feats.items():
   x=np.concatenate(pieces);sc=StandardScaler().fit(x[fi]);xx=sc.transform(x);model=LogisticRegression(C=.01,max_iter=2000,random_state=42).fit(xx[fi],y[fi],sample_weight=w);p=model.predict_proba(xx[te]);z=torch.from_numpy(np.log(np.maximum(p,1e-12)));r=dict(fold=fold,layer=name,metrics=base.metrics(z,torch.from_numpy(y[te])),fit_correct=int((model.predict(xx[fi])==y[fi]).sum()),iterations=model.n_iter_.tolist());out.append(r);print('RESULT',r,flush=True)
   joblib.dump(dict(scaler=sc,model=model,fit_indices=fi,test_indices=te),dest/f'{name}.joblib');np.savez_compressed(dest/f'{name}.npz',probabilities=p,indices=te,labels=y[te])
  allparts={k:torch.cat(v) for k,v in parts.items()};base.save(dest/'components.json',{k:base.metrics((v if k in ['logits','pure_logits'] else v.clamp_min(1e-8).log())[te],torch.from_numpy(y[te])) for k,v in allparts.items()});torch.save(allparts,dest/'baseline_components.pt');base.save(dest/'results.json',out);results.extend(out);base.save(W/'results.json',results);del m;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
