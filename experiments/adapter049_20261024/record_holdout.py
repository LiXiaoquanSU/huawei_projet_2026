"""Conditional adapter test: whole records excluded from adapter fit only.

The inherited backbone saw all source records during historical training. These
are NOT end-to-end independent accuracy estimates and must never be called so.
"""
import hashlib,gc
import numpy as np
import torch
from run import *

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    cache=torch.load(Path('D:/huawei2026-experiments/evidence049_research_20261024/frozen_latents.pt'),weights_only=True)
    rows=cache['rows'];unique={r['identifier']:r for r in rows};assignment={};offset=0
    groups=sorted({(int(r['headcount']),r['room']) for r in unique.values()})
    for label,room in groups:
        ids=sorted([k for k,r in unique.items() if int(r['headcount'])==label and r['room']==room],key=lambda s:hashlib.sha256(('adapter-record42:'+s).encode()).hexdigest())
        for j,identifier in enumerate(ids):assignment[identifier]=(j+offset)%3
        offset=(offset+len(ids))%3
    save(WORK/'record_holdout_plan.json',dict(assignment=assignment,method='calibrated_consistent',steps=75,residualscale=.25,mix=1.,selection='Fixed from earlier development, no per-fold threshold or epoch choice.',
      caveat='Only adapter fit excludes these records. Historical frozen backbone saw all960 windows; conditional ablation, NOT end-to-end independent validation.'))
    checkpoint=ROOT/'models/best_037/final.pt';model,b=build(checkpoint);del model
    reports=[];allz=np.empty((960,5),np.float32)
    for fold in range(3):
        fit_ids={k for k,v in assignment.items() if v!=fold};test_ids=set(assignment)-fit_ids
        assert fit_ids.isdisjoint(test_ids)
        head,states,history,_,base=fit(cache,checkpoint,fit_ids,set(),'calibrated_consistent','actual_record',b,1.,steps=75)
        head.load_state_dict(states[75],strict=False)
        with torch.no_grad():
            z,_=head(cache['latents'].cuda().flatten(0,1),cache['summaries'].cuda().flatten(0,1),.25)
            z=policy(z.reshape(-1,4,5),b.cuda());indices=[i for i,r in enumerate(rows) if r['identifier'] in test_ids]
            allz[indices]=z[indices].cpu().numpy();report=dict(fold=fold,fit_ids=sorted(fit_ids),test_ids=sorted(test_ids),metrics=metrics(z,cache['labels'].cuda(),base,indices))
        reports.append(report);print('RECORD_HOLDOUT',report['fold'],report['metrics'],flush=True)
        del head;gc.collect();torch.cuda.empty_cache()
    baseline=policy(cache['logits'],b);combined=metrics(torch.from_numpy(allz),cache['labels'],baseline,list(range(960)))
    save(WORK/'record_holdout_results.json',dict(folds=reports,combined=combined,independent_backbone_test=False))
    print('RECORD_HOLDOUT_COMBINED',combined,flush=True)

if __name__=='__main__':main()
