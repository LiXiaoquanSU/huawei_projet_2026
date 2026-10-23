"""Paired loss ablation on the successful context architecture; no window changes."""
import gc
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('D:/huawei2026-experiments/training_diversity_20261023')
OLD = Path('D:/huawei2026-experiments/global_local_20261021')
sys.path.insert(0, str(ROOT/'experiments/global_local_20261021'))
spec = importlib.util.spec_from_file_location('paired_reference', ROOT/'experiments/structure_events_20261021/train.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
network_build = ref.build
original_loss = ref.task_loss


class Model(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.base = network_build('context_events')
        self.recipe = recipe

    def preprocess(self, x, dt):
        return self.base.preprocess(x, dt)

    def forward_tasks(self, t, s):
        tasks = self.base.forward_tasks(t, s)
        if self.recipe == 'direct':
            tasks['logits'] = tasks['direct'].clamp_min(1e-8).log()
        return tasks

    def forward_features(self, t, s):
        return self.forward_tasks(t, s)['logits']


CURRENT = 'physical'


def loss(tasks, y, target, mode):
    # direct retains expert supervision; excludes ordinal, JS and signal losses.
    return original_loss(tasks, y, target, 'direct' if CURRENT == 'direct' else 'physical')


def fit(data, recipe, seed, fold):
    global CURRENT
    CURRENT = recipe
    ref.WORK = WORK
    ref.build = lambda _: Model(recipe)
    ref.task_loss = loss
    result = ref.fit(data, recipe, seed, fold)
    return result


def prediction(path):
    rows = ref.readrows(path) if hasattr(ref, 'readrows') else None
    if rows is None:
        import csv
        with path.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
    p = np.exp([[float(r[f'log_probability_{i}']) for i in range(5)] for r in rows])
    return rows, p/p.sum(1, keepdims=True)


def compare(recipe, seed):
    results = []
    for fold in ('a1', 'c1', 'g1'):
        rows, candidate = prediction(WORK/'fits'/f'{recipe}_s{seed}_{fold}/predictions.csv')
        r1, cross = prediction(OLD/'fits'/f'cross_events_s42_{fold}/predictions.csv')
        r2, context = prediction(OLD/'fits'/f'context_events_s2026_{fold}/predictions.csv')
        assert len(rows) == len(r1) == len(r2)
        assert all(tuple(a[k] for k in ('identifier','start','stop','headcount')) ==
                   tuple(b[k] for k in ('identifier','start','stop','headcount'))
                   for other in (r1,r2) for a,b in zip(rows,other))
        y = np.array([int(r['headcount']) for r in rows])
        baseline = ((cross+context)*.5).argmax(1)
        # Prespecified replacement: keep original cross .39 and replace context.
        blended = ((cross+candidate)*.5).argmax(1)
        cm = np.zeros((5,5), int)
        np.add.at(cm, (y,blended), 1)
        results.append(dict(room=fold, single_accuracy=float((candidate.argmax(1)==y).mean()),
                            baseline_accuracy=float((baseline==y).mean()),
                            blend_accuracy=float((blended==y).mean()),
                            corrected=int(((baseline!=y)&(blended==y)).sum()),
                            broken=int(((baseline==y)&(blended!=y)).sum()),
                            confusion_matrix=cm.tolist()))
    cm = np.sum([x['confusion_matrix'] for x in results],axis=0)
    return dict(recipe=recipe, seed=seed, rooms=results,
                room_mean=float(np.mean([x['blend_accuracy'] for x in results])),
                baseline_mean=float(np.mean([x['baseline_accuracy'] for x in results])),
                macro_recall=float(np.mean(np.diag(cm)/cm.sum(1))),
                worst_room=min(x['blend_accuracy'] for x in results))


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available()
    WORK.mkdir(parents=True,exist_ok=True)
    snapshot=WORK/'source_snapshot';snapshot.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),snapshot/'train.py')
    shutil.copy2(ROOT/'experiments/structure_events_20261021/train.py',snapshot/'reference_train.py')
    data=ref.load()
    # Same new initialization, same samples/optimizer/epochs; only loss/readout differ.
    for recipe in ('physical','direct'):
        for fold in ('a1','c1','g1'):
            fit(data,recipe,314,fold)
    first=[compare(recipe,314) for recipe in ('physical','direct')]
    ref.save(WORK/'screen.json',first)
    # Fixed second initialization for both arms, not just the apparent winner.
    for recipe in ('physical','direct'):
        for fold in ('a1','c1','g1'):
            fit(data,recipe,2026,fold)
    comparisons=[compare(recipe,seed) for recipe in ('physical','direct') for seed in (314,2026)]
    summary=[]
    for recipe in ('physical','direct'):
        rr=[x for x in comparisons if x['recipe']==recipe]
        summary.append(dict(recipe=recipe,room_mean=float(np.mean([x['room_mean'] for x in rr])),
                            macro_recall=float(np.mean([x['macro_recall'] for x in rr])),
                            worst_room=min(x['worst_room'] for x in rr)))
    selected=max(summary,key=lambda x:(min(x['room_mean'],x['macro_recall']),x['worst_room']))
    # Seed for the deployment candidate fixed in advance, not selected by accuracy.
    selection=dict(recipe=selected['recipe'],seed=314,comparisons=comparisons,summary=summary,
                   checkpoint_rule='fixed final epoch; deployment seed314 fixed in advance',
                   unchanged='old960 nonoverlap windows, class-room sampler, AdamW .0004, bs32, 5epochs',
                   direct_loss='direct CE + .15 expert CE; direct readout, no untrained ordinal averaging',
                   baseline='original fixed cross42/context2026 .40 ensemble; all development historical reused folds')
    ref.save(WORK/'selection.json',selection)
    # Stress check is reported after selection, never used to retune the loss.
    for recipe in ('physical','direct'):
        fit(data,recipe,314,'e1')
    fit(data,selected['recipe'],314,'full')
    ref.save(WORK/'complete.json',dict(fits=15,selection=selection,platform_score=None))
    print('COMPLETE',json.dumps(selection),flush=True)
    del data
    gc.collect()


if __name__=='__main__':
    main()
