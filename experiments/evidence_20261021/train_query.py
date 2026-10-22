"""Independent second-stage class-query test; fixed before inspecting its results."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
import train as base
from query_network import build
WORK=Path('D:/huawei2026-experiments/query_20261021')
def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    base.ref.WORK=WORK;base.ref.build=build;WORK.mkdir(parents=True,exist_ok=True)
    snap=WORK/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    d=base.load();results=[]
    for s in (42,2026):
        for f in base.FOLDS:base.fit(d,'query_context',s,f)
        results.append(base.summary('query_context',s,WORK))
    base.fit(d,'query_context',42,'e1')
    controls=[base.summary(v,s,base.OLD) for v in ('cross_events','context_events') for s in (42,2026)]
    baseline=max(np.mean([r['selection_score'] for r in controls if r['architecture']==v]) for v in ('cross_events','context_events'))
    score=float(np.mean([r['selection_score'] for r in results]));improved=bool(score>baseline)
    item=dict(architecture='query_context',seed=max(results,key=base.rank)['seed'],two_seed_selection=score,two_seed_room_mean=float(np.mean([r['room_mean'] for r in results])),two_seed_worst=float(np.mean([r['worst'] for r in results])),two_seed_nll=float(np.mean([r['nll'] for r in results])),results=results,e1=base.read(WORK/'fits/query_context_s42_e1/metrics.json')['validation'])
    base.save(WORK/'selection.json',dict(candidates=[item],screen=results[:1],controls=controls,improved_development=improved,platform_best=.39))
    if improved:base.fit(d,'query_context',item['seed'],'full')
    print('QUERY_COMPLETE',json.dumps(dict(improved=improved,candidate=item)),flush=True)
if __name__=='__main__':main()
