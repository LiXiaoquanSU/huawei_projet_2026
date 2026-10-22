import gc,shutil
from pathlib import Path
import numpy as np
import torch
from common import ROOT,WORK,FeatureStore,build,old_data,read,save
from engine import fit,evaluate,ordered_indices
FOLDS=('a1','c1','g1')
MODELS=('cross_events','context_events','mixer')
def summary(name,recipe,s):
    rr=[read(WORK/'fits'/f'{name}_{recipe}_s{s}_{f}/metrics.json')['validation'] for f in FOLDS];cm=np.asarray([r['confusion_matrix'] for r in rr]).sum(0)
    mean=float(np.mean([r['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    dense=[read(WORK/'fits'/f'{name}_{recipe}_s{s}_{f}/metrics.json')['dense_validation']['record_mean_accuracy'] for f in FOLDS]
    return dict(architecture=name,recipe=recipe,seed=s,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst=min(r['accuracy'] for r in rr),nll=float(np.mean([r['nll'] for r in rr])),dense_record_room_mean=float(np.mean(dense)),confusion_matrix=cm.tolist())
def rank(r):return r['selection_score'],r['worst'],-r['nll']
def baseline(name,s):
    folder='fresh_backbones_20261022' if name=='mixer' else 'global_local_20261021';base=Path('D:/huawei2026-experiments')/folder/'fits'
    rr=[read(base/f'{name}_s{s}_{f}/metrics.json')['validation'] for f in FOLDS];cm=np.asarray([r['confusion_matrix'] for r in rr]).sum(0);mean=float(np.mean([r['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    return dict(architecture=name,recipe='baseline',seed=s,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst=min(r['accuracy'] for r in rr),nll=float(np.mean([r['nll'] for r in rr])),confusion_matrix=cm.tolist())
def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;assert torch.cuda.is_available();store=FeatureStore()
    snap=WORK/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    controls=[baseline(name,s) for name in MODELS for s in (42,2026)];save(WORK/'controls.json',controls)
    parity=[]
    for name in MODELS:
        base='fresh_backbones_20261022' if name=='mixer' else 'global_local_20261021';folder=Path('D:/huawei2026-experiments')/base/'fits'/f'{name}_s42_a1'
        model=build(name).cuda();cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['model_state']);m,_,_=evaluate(model,store,ordered_indices(store,'baseline','a1',False));assert m['confusion_matrix']==read(folder/'metrics.json')['validation']['confusion_matrix'];parity.append(dict(architecture=name,metrics=m));del model,cp;gc.collect();torch.cuda.empty_cache()
    save(WORK/'control_parity.json',parity)
    for name in MODELS:
        for recipe in ('step_1s','step_0p5s'):
            for f in FOLDS:fit(store,name,recipe,42,f)
    screen=[summary(name,r,42) for name in MODELS for r in ('step_1s','step_0p5s')];save(WORK/'screen.json',screen)
    candidates=[]
    for name in MODELS:
        chosen=max([r for r in screen if r['architecture']==name],key=rank);recipe=chosen['recipe']
        for f in FOLDS:fit(store,name,recipe,2026,f)
        fit(store,name,recipe,42,'e1');rr=[summary(name,recipe,s) for s in (42,2026)];bb=[r for r in controls if r['architecture']==name]
        candidates.append(dict(architecture=name,recipe=recipe,seed=2026 if name=='context_events' else 42,two_seed_selection=float(np.mean([r['selection_score'] for r in rr])),two_seed_room_mean=float(np.mean([r['room_mean'] for r in rr])),baseline_room_mean=float(np.mean([r['room_mean'] for r in bb])),improved_development=float(np.mean([r['selection_score'] for r in rr]))>float(np.mean([r['selection_score'] for r in bb])),results=rr,e1=read(WORK/'fits'/f'{name}_{recipe}_s42_e1/metrics.json')['validation']))
    candidates.sort(key=lambda r:r['two_seed_selection'],reverse=True);save(WORK/'selection.json',dict(candidates=candidates,screen=screen,controls=controls,platform_best=.39,fits_expected=33))
    for item in candidates:fit(store,item['architecture'],item['recipe'],item['seed'],'full')
    print('TRAIN_COMPLETE',[(c['architecture'],c['recipe'],c['two_seed_room_mean'],c['baseline_room_mean']) for c in candidates],flush=True)
if __name__=='__main__':main()
