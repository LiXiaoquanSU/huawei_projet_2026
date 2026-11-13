"""Inference intervention audit; diagnostic only, never selects weights."""
import sys
from pathlib import Path
import torch
import numpy as np
from train_chain import EvidenceChain,predict,metric,old,CACHE,WORK

def main():
 torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 t=torch.from_numpy(np.array(np.load(CACHE/'t.npy',mmap_mode='r'))).cuda();base=torch.from_numpy(np.array(np.load(CACHE/'s.npy',mmap_mode='r'))).cuda();spectrum=torch.from_numpy(np.load('D:/huawei2026-experiments/timed_spectrum_20261030/features.npz')['actual']).cuda();s=torch.cat([base,spectrum[:,None].expand(-1,2,-1)],-1)
 rows=old.read(CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows]);shift=list(range(len(rows)))
 for record in sorted({r['identifier'] for r in rows}):
  ix=[i for i,r in enumerate(rows) if r['identifier']==record]
  for k,i in enumerate(ix):shift[i]=ix[(k+7)%len(ix)]
 variants={'zero':torch.cat([base,torch.zeros_like(s[:,:,17:])],-1),'within_record_shift':torch.cat([base,s[shift,:,17:]],-1)};results=[]
 for fold in range(3):
  folder=WORK/f'record{fold}_full_s42';checkpoint=torch.load(folder/'final.pt',weights_only=False,map_location='cpu');d=torch.load(folder/'predictions.pt',weights_only=False,map_location='cpu');m=EvidenceChain().cuda().eval();m.load_state_dict(checkpoint['model_state']);ref=predict(m,t,s,list(range(len(rows))));err=float((ref.exp()-d['logits'].exp()).abs().max());assert err<1e-5,err
  for name,altered in variants.items():
   z=predict(m,t,altered,list(range(len(rows))));ix=d['test_indices'];results.append(dict(fold=fold,intervention=name,original=metric(ref[ix],y[ix]),altered=metric(z[ix],y[ix]),changed=int((z[ix].argmax(1)!=ref[ix].argmax(1)).sum()),mean_probability_l1=float((z[ix].exp()-ref[ix].exp()).abs().sum(-1).mean()),reproduction_error=err))
  print('AUDITED',fold,flush=True)
 old.save(WORK/'audit.json',dict(results=results,limits='Zero can be OOD. Within-record replacement preserves known record label but may break simultaneous temporal relations. Sensitivity is not evidence of causal human detection. No intervention selects model or modifies stored weights.'))
if __name__=='__main__':main()
