"""Fixed calibration ablation; no target-based tuning or new candidate selection."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'));import g60_common as old
WORK=Path('D:/huawei2026-experiments/pair_views_20261102');CACHE=Path('D:/huawei2026-experiments/g60_20261025')

def main():
 torch.set_num_threads(4);results=[]
 for room in ('a1','c1','g1','e1'):
  d=torch.load(CACHE/f'{room}.pt',weights_only=True);raw=d['raw'];y=d['labels'];fi,cal,te=old.split(room,d['rows']);b,T=old.calibration(room);base=old.policy(raw,b,T)
  # Hypothesis only: uniform class marginal in balanced source-fit distribution.
  w=old.balance(d['rows'],y,fi);q=raw[fi].exp().mean((1,2));source_marginal=(q*w[:,None]).sum(0);prior=-source_marginal.clamp_min(1e-6).log();prior-=prior.mean()
  variants={'existing':(b,T),'raw':(torch.zeros_like(b),1.),'temperature_only':(torch.zeros_like(b),T),'bias_only':(b,1.),'source_uniform':(prior,1.)}
  for name,(bias,temp) in variants.items():
   z=old.policy(raw,bias,temp);r=dict(room=room,variant=name,bias=bias.tolist(),temperature=float(temp),test=old.metrics(z,y,base,te));results.append(r)
   np.savez_compressed(WORK/f'calibration_{room}_{name}.npz',logits=z.numpy(),test_indices=te)
  print(room,[(v['variant'],v['test']['correct']) for v in results if v['room']==room],flush=True)
 summary={name:dict(correct=sum(v['test']['correct'] for v in results if v['variant']==name),total=880,confusion=np.sum([v['test']['confusion_matrix'] for v in results if v['variant']==name],0).tolist()) for name in variants}
 old.save(WORK/'calibration_audit.json',dict(results=results,summary=summary,limits='Fixed ablations chosen before results, reused development rooms. Altering bias/temperature also changes protected routing. Source-uniform assumption unverified; no target prior inferred. This is not actual054 evaluation.'))
 print(summary)
if __name__=='__main__':main()
