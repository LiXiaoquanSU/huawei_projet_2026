import gc,importlib.util,json,shutil
from pathlib import Path
import numpy as np
import torch
from network import ROOT,build
spec=importlib.util.spec_from_file_location('training_reference',ROOT/'experiments/structure_events_20261021/train.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
WORK=Path('D:/huawei2026-experiments/evidence_20261021');OLD=Path('D:/huawei2026-experiments/global_local_20261021');ref.WORK=WORK;ref.build=build
read,save,sha,seed,load,fit,evaluate=ref.read,ref.save,ref.sha,ref.seed,ref.load,ref.fit,ref.evaluate
FOLDS=['a1','c1','g1']
def summary(v,s,root=WORK):
    rr=[read(root/'fits'/f'{v}_s{s}_{f}/metrics.json')['validation'] for f in FOLDS]
    cm=np.asarray([r['confusion_matrix'] for r in rr]).sum(0);mean=float(np.mean([r['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    return dict(architecture=v,seed=s,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst=min(r['accuracy'] for r in rr),nll=float(np.mean([r['nll'] for r in rr])),rooms={f:r['accuracy'] for f,r in zip(FOLDS,rr)},confusion_matrix=cm.tolist())
def rank(r):return r['selection_score'],r['worst'],-r['nll']
def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;assert torch.cuda.is_available();WORK.mkdir(parents=True,exist_ok=True)
    interrupted=WORK/'fits/aligned_events_s42_g1'
    if interrupted.exists() and not (interrupted/'metrics.json').exists() and not (WORK/'interrupted_before_timed_fix').exists():
        shutil.copytree(interrupted,WORK/'interrupted_before_timed_fix')
    snap=WORK/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    d=load();controls=[summary('cross_events',s,OLD) for s in (42,2026)];save(WORK/'controls.json',controls)
    for v in ('cross_events','aligned_events','timed_events','aligned_timed'):
        for f in FOLDS:fit(d,v,42,f)
    check=summary('cross_events',42)
    assert check['confusion_matrix']==controls[0]['confusion_matrix'],(check,controls[0])
    screen=[summary(v,42) for v in ('aligned_events','timed_events','aligned_timed')];save(WORK/'screen.json',screen)
    top=sorted(screen,key=rank,reverse=True)[:2];final=[]
    for item in top:
        v=item['architecture']
        for f in FOLDS:fit(d,v,2026,f)
        fit(d,v,42,'e1');rr=[summary(v,s) for s in (42,2026)]
        final.append(dict(architecture=v,seed=max(rr,key=rank)['seed'],two_seed_selection=float(np.mean([r['selection_score'] for r in rr])),two_seed_room_mean=float(np.mean([r['room_mean'] for r in rr])),two_seed_worst=float(np.mean([r['worst'] for r in rr])),two_seed_nll=float(np.mean([r['nll'] for r in rr])),results=rr,e1=read(WORK/'fits'/f'{v}_s42_e1/metrics.json')['validation']))
    final.sort(key=lambda r:(r['two_seed_selection'],r['two_seed_worst'],-r['two_seed_nll']),reverse=True)
    improved=final[0]['two_seed_selection']>float(np.mean([r['selection_score'] for r in controls]))
    save(WORK/'selection.json',dict(candidates=final,screen=screen,controls=controls,improved_development=improved,platform_best=.39))
    if improved:fit(d,final[0]['architecture'],final[0]['seed'],'full')
    print('COMPLETE',json.dumps(dict(improved=improved,candidates=final)),flush=True)
if __name__=='__main__':main()
