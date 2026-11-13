"""Known-room diagnostic; never treat subset accuracy as five-class platform score."""
import sys,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
spec=importlib.util.spec_from_file_location('dynamic_evidence_stage',Path(__file__).with_name('run.py'));stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
base,OLD,input_view,W=stage.base,stage.OLD,stage.input_view,stage.W
def main():
 threadpool_limits(2);torch.set_num_threads(2);rows=base.load(base.CACHE/'rows.json');y=np.array([r['headcount'] for r in rows]);x=np.load(OLD/'joint.npy',mmap_mode='r');features={k:[] for k in ['raw','static','dynamic']}
 for start in range(0,len(rows),32):
  batch=torch.from_numpy(np.array(x[start:start+32]))
  for mode in features:
   v=input_view(batch,mode)
   if mode=='static':z=v.mean(-1).flatten(1)
   else:z=torch.cat([v.mean(-1),v.std(-1,unbiased=False),(v[...,1:]-v[...,:-1]).abs().mean(-1),(v[...,4:]-v[...,:-4]).abs().mean(-1)],1).flatten(1)
   features[mode].append(z.numpy())
 features={k:np.concatenate(v) for k,v in features.items()};results=[];manifest={}
 for room,classes in [('a1',[1,2,3,4]),('c1',[1,2,3,4]),('g1',[0,1])]:
  selected=[i for i,r in enumerate(rows) if r['room']==room and int(y[i]) in classes];assignment={}
  for c in classes:
   ids=sorted({rows[i]['identifier'] for i in selected if y[i]==c},key=lambda v:hashlib.sha256(('dynamic_room_20261108:'+v).encode()).hexdigest());assert len(ids)>=2
   for j,ident in enumerate(ids):assignment[ident]=j%2
  manifest[room]=dict(classes=classes,assignment=assignment)
  for fold in range(2):
   fi=[i for i in selected if assignment[rows[i]['identifier']]!=fold];te=[i for i in selected if assignment[rows[i]['identifier']]==fold];assert {rows[i]['identifier'] for i in fi}.isdisjoint({rows[i]['identifier'] for i in te});w=1/np.bincount(y[fi],minlength=5)[y[fi]];w*=len(fi)/w.sum()
   for mode,x in features.items():
    sc=StandardScaler().fit(x[fi]);xx=sc.transform(x);model=LogisticRegression(C=.01,max_iter=2000,random_state=42).fit(xx[fi],y[fi],sample_weight=w);pred=model.predict(xx[te]);cm=confusion_matrix(y[te],pred,labels=classes);correct=int((pred==y[te]).sum());r=dict(room=room,classes=classes,fold=fold,mode=mode,correct=correct,total=len(te),balanced_accuracy=float((np.diag(cm)/cm.sum(1)).mean()),majority_baseline_correct=max(np.bincount(y[te],minlength=5)).item(),confusion=cm.tolist(),fit_records=len({rows[i]['identifier'] for i in fi}),test_records=len({rows[i]['identifier'] for i in te}));results.append(r);print('WITHIN',room,fold,mode,correct,'/',len(te),flush=True)
 base.save(W/'within_room.json',dict(protocol='Fixed C=.01 source-only standardization, class-balanced weighting, 2 record folds independently in a1/c1/g1. Room and supported class set are known for diagnostic only. Each original record wholly held out.',limitations='Not a deployable unknown-room classifier. a1/c1 omit0, g1 only0/1. No pooled accuracy may be compared to target0.60 five-class platform score. Room identity alone does not identify session/layout confounders.',manifest=manifest,results=results));print('COMPLETE',flush=True)
if __name__=='__main__':main()
