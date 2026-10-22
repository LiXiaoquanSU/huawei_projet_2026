"""Reuse the exact .38 training loop; all new fits go to a new D folder."""
import gc,importlib.util,json,shutil
from pathlib import Path
import numpy as np
import torch
from network import ROOT,build
spec=importlib.util.spec_from_file_location('training_reference',ROOT/'experiments/structure_events_20261021/train.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
WORK=Path('D:/huawei2026-experiments/global_local_20261021');OLD=Path('D:/huawei2026-experiments/structure_events_20261021');ref.WORK=WORK;ref.build=build
read,save,sha,seed,load,fit,evaluate=ref.read,ref.save,ref.sha,ref.seed,ref.load,ref.fit,ref.evaluate
VARIANTS=['context_events','cross_events','band_events'];FOLDS=['a1','c1','g1']
def summary(v,s,root=WORK):
    rr=[read(root/'fits'/f'{v}_s{s}_{f}/metrics.json')['validation'] for f in FOLDS];cm=np.asarray([r['confusion_matrix'] for r in rr]).sum(0);mean=float(np.mean([r['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    return dict(architecture=v,seed=s,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst=min(r['accuracy'] for r in rr),nll=float(np.mean([r['nll'] for r in rr])),rooms={f:r['accuracy'] for f,r in zip(FOLDS,rr)},confusion_matrix=cm.tolist())
def rank(r):return r['selection_score'],r['worst'],-r['nll']
def controls(d):
    out=[]
    for v in ('baseline','events'):
        for s in (42,2026):
            for f in FOLDS:
                m=read(OLD/'fits'/f'{v}_s{s}_{f}/metrics.json');c=m['config'];assert c['epochs']==5 and c['lr']==.0004 and c['batch_size']==32 and c['external_data'] is False
            out.append(summary(v,s,OLD))
    parity={}
    for v in ('baseline','events'):
        folder=OLD/'fits'/f'{v}_s42_a1';cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=True);model=build(v).cuda();model.load_state_dict(cp['model_state']);ix=[i for i,r in enumerate(d['rows']) if r['room']=='a1'];m,p=evaluate(model,d,ix)
        assert m['confusion_matrix']==read(folder/'metrics.json')['validation']['confusion_matrix'];parity[v]=dict(accuracy=m['accuracy'],checkpoint_sha256=sha(folder/'final.pt'));del model,cp;gc.collect();torch.cuda.empty_cache()
    save(WORK/'reused_controls.json',dict(results=out,prediction_recomputed_a1=parity,training_reference_sha256=sha(ROOT/'experiments/structure_events_20261021/train.py'),control_source='previous matched official-only folds; not new fits'));return out
def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;assert torch.cuda.is_available();WORK.mkdir(parents=True,exist_ok=True);snap=WORK/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    d=load();controls(d)
    for v in VARIANTS:
        for f in FOLDS:fit(d,v,42,f)
    screen=[summary(v,42) for v in VARIANTS];save(WORK/'screen_results.json',screen);top=sorted(screen,key=rank,reverse=True)[:2];print('SCREEN',json.dumps(screen),flush=True)
    for item in top:
        v=item['architecture']
        for f in FOLDS:fit(d,v,2026,f)
        fit(d,v,42,'e1')
    final=[]
    for item in top:
        v=item['architecture'];rr=[summary(v,s) for s in (42,2026)];best=max(rr,key=rank)
        final.append(dict(architecture=v,seed=best['seed'],two_seed_selection=float(np.mean([r['selection_score'] for r in rr])),two_seed_room_mean=float(np.mean([r['room_mean'] for r in rr])),two_seed_worst=float(np.mean([r['worst'] for r in rr])),two_seed_nll=float(np.mean([r['nll'] for r in rr])),results=rr,e1=read(WORK/'fits'/f'{v}_s42_e1/metrics.json')['validation']))
    final.sort(key=lambda r:(r['two_seed_selection'],r['two_seed_worst'],-r['two_seed_nll']),reverse=True);save(WORK/'selection.json',dict(candidates=final,platform_best=.38,screen=screen))
    for item in final:fit(d,item['architecture'],item['seed'],'full')
    print('ALL_NEW_TRAINING_COMPLETE',json.dumps(final),flush=True)
if __name__=='__main__':main()
