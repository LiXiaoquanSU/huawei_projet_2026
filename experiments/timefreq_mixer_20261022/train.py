import shutil,json
from pathlib import Path
import numpy as np
import torch
from engine import ROOT,WORK,read,save,sha,load,fit,build
FOLDS=('a1','c1','g1')
def summary(v,s):
    rr=[read(WORK/'fits'/f'{v}_s{s}_{f}/metrics.json')['validation'] for f in FOLDS];cm=np.asarray([r['confusion_matrix'] for r in rr]).sum(0)
    mean=float(np.mean([r['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    return dict(architecture=v,seed=s,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst=min(r['accuracy'] for r in rr),nll=float(np.mean([r['nll'] for r in rr])),rooms={f:r['accuracy'] for f,r in zip(FOLDS,rr)},confusion_matrix=cm.tolist())
def rank(r):return r['selection_score'],r['worst'],-r['nll']
def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;assert torch.cuda.is_available()
    snap=WORK/'source_snapshot';snap.mkdir(parents=True,exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    d=load();variants=('conv_coarse','conv_fine','axial_fine')
    for v in variants:
        for f in FOLDS:fit(d,v,42,f)
    screen=[summary(v,42) for v in variants];save(WORK/'screen.json',screen);print('SCREEN',json.dumps(screen),flush=True)
    old=Path('D:/huawei2026-experiments/fresh_backbones_20261022');controls=[]
    for seed in (42,2026):
        rr=[read(old/'fits'/f'mixer_s{seed}_{f}/metrics.json')['validation'] for f in FOLDS];cm=np.asarray([r['confusion_matrix'] for r in rr]).sum(0)
        mean=float(np.mean([r['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
        controls.append(dict(architecture='mixer',seed=seed,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst=min(r['accuracy'] for r in rr),nll=float(np.mean([r['nll'] for r in rr])),confusion_matrix=cm.tolist()))
    from engine import evaluate
    m=build('mixer').cuda();cp=torch.load(old/'fits/mixer_s42_a1/final.pt',map_location='cpu',weights_only=True);m.load_state_dict(cp['model_state'])
    metric,_=evaluate(m,d,[i for i,r in enumerate(d['rows']) if r['room']=='a1']);assert metric['confusion_matrix']==read(old/'fits/mixer_s42_a1/metrics.json')['validation']['confusion_matrix'];save(WORK/'control_parity.json',metric);del m,cp;torch.cuda.empty_cache()
    save(WORK/'controls.json',controls)
    final=[]
    for entry in sorted(screen,key=rank,reverse=True)[:2]:
        v=entry['architecture']
        for f in FOLDS:fit(d,v,2026,f)
        fit(d,v,42,'e1');rr=[summary(v,s) for s in (42,2026)]
        final.append(dict(architecture=v,seed=max(rr,key=rank)['seed'],two_seed_selection=float(np.mean([r['selection_score'] for r in rr])),two_seed_room_mean=float(np.mean([r['room_mean'] for r in rr])),two_seed_worst=float(np.mean([r['worst'] for r in rr])),two_seed_nll=float(np.mean([r['nll'] for r in rr])),results=rr,e1=read(WORK/'fits'/f'{v}_s42_e1/metrics.json')['validation']))
    final.sort(key=lambda r:(r['two_seed_selection'],r['two_seed_worst'],-r['two_seed_nll']),reverse=True)
    save(WORK/'selection.json',dict(candidates=final,screen=screen,controls=controls,improved_development=final[0]['two_seed_selection']>float(np.mean([r['selection_score'] for r in controls])),target_60_reached=final[0]['two_seed_room_mean']>=.6,platform_best=.39))
    fit(d,final[0]['architecture'],final[0]['seed'],'full');print('COMPLETE',json.dumps(final),flush=True)
if __name__=='__main__':main()
