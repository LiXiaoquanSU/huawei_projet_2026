"""One documented follow-up after catastrophic e1 stress failure."""
import json
import numpy as np
import torch
from train import WORK,read,save,load,fit,summary,rank

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    original=read(WORK/'selection.json')
    if not (WORK/'initial_selection.json').exists():save(WORK/'initial_selection.json',original)
    d=load();v='event_residual'
    for s in (42,2026):
        for f in ('a1','c1','g1'):fit(d,v,s,f)
    fit(d,v,42,'e1')
    initial=read(WORK/'initial_selection.json');rep=initial['replicates']+[summary(v,s) for s in (42,2026)]
    base_e1=read(WORK/'fits/baseline_s42_e1/metrics.json')['validation']['accuracy'];candidates=[]
    for a in ('events',v):
        rr=[r for r in rep if r['architecture']==a];e1=read(WORK/'fits'/f'{a}_s42_e1/metrics.json')['validation']['accuracy']
        candidates.append(dict(architecture=a,passes_e1_gate=e1>=base_e1,mean_selection=float(np.mean([r['selection_score'] for r in rr])),mean_worst=float(np.mean([r['worst'] for r in rr])),mean_nll=float(np.mean([r['nll'] for r in rr])),e1_accuracy=e1))
    chosen=max(candidates,key=lambda r:(r['passes_e1_gate'],r['mean_selection'],r['mean_worst'],-r['mean_nll']));best=max([r for r in rep if r['architecture']==chosen['architecture']],key=rank)
    selection=dict(architecture=chosen['architecture'],seed=best['seed'],screen=initial['screen'],replicates=rep,followup=candidates,stress_gate_passed=chosen['passes_e1_gate'])
    save(WORK/'selection.json',selection);fit(d,selection['architecture'],selection['seed'],'full');print('FOLLOWUP_COMPLETE',json.dumps(selection),flush=True)
if __name__=='__main__':main()
