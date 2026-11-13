"""Planned single-variable neighbour of scored .49; total1 bias .50, not .85."""
import gc
import importlib.util
import shutil
from pathlib import Path
import numpy as np
import onnx

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('bias050_shared_builder',ROOT/'experiments/final_bias_20261024/build.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
WORK=Path('D:/huawei2026-experiments/bias050_20261024')
PREVIOUS=Path('D:/huawei2026-experiments/final_bias_20261024')
BEST=ROOT/'models/final_bias_20261024/01_final_bias_one035.onnx'
BEST_HASH='d083e3bb36f522e29b45acb88e41e5577013f9140b2f59dd2f9741c476f839df'


def main():
    assert shared.sha(BEST)==BEST_HASH
    WORK.mkdir(parents=True,exist_ok=True)
    for name in ('rows.json','cached_exact048.npz'):
        source=PREVIOUS/name;target=WORK/name
        if target.exists():assert shared.sha(target)==shared.sha(source)
        else:shutil.copy2(source,target)
    shared.save(WORK/'summary.json',dict(chosen_delta=[0.,.5,0.,0.,0.],chosen_method='planned_neighbour_after_scored049',
        transformation='Frozen048 final output + [0,0.50,0,0,0]. Relative to scored049 total0.35 this adds only0.15. No rerouting.',
        provenance='No new training or calibration fit. User-authorized single-variable .50 follow-up after scored .35 reached0.49.',
        best049_sha256=BEST_HASH,target_platform_accuracy='exceed0.52 is goal, not prediction',
        bias_meaning='logit shift, not probability increment',
        limitations='Source windows seen by backbone; local ladder already inspected, not independent validation. Empty false positives may increase.'))
    shared.WORK=WORK;shared.EXPORT=WORK/'export';shared.MAX_SHIFT=.500001
    shared.main()
    # Actual scored049 graph differs in exactly one residual float, no stacked Add.
    old=onnx.load(BEST);new=onnx.load(WORK/'export/model.onnx')
    assert len(old.graph.node)==len(new.graph.node)
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(old.graph.node,new.graph.node))
    assert len(old.graph.initializer)==len(new.graph.initializer)
    changed=[]
    for a,b in zip(old.graph.initializer,new.graph.initializer):
        assert a.name==b.name
        if a.SerializeToString()!=b.SerializeToString():
            changed.append(a.name)
            assert a.name=='__final_bias_delta'
            x=onnx.numpy_helper.to_array(a);y=onnx.numpy_helper.to_array(b)
            assert np.array_equal(x,np.array([0,.35,0,0,0],np.float32))
            assert np.array_equal(y,np.array([0,.50,0,0,0],np.float32))
    assert changed==['__final_bias_delta']
    assert old.graph.input==new.graph.input and old.graph.output==new.graph.output
    del old,new;gc.collect()
    with np.load(WORK/'cached_exact048.npz') as d:
        z=d['logits'];labels=d['labels']
    before=(z+np.array([0,.35,0,0,0],np.float32)).argmax(1)
    after=(z+np.array([0,.50,0,0,0],np.float32)).argmax(1)
    changes=np.flatnonzero(before!=after)
    proof=dict(best049_sha256=BEST_HASH,candidate_sha256=shared.sha(WORK/'export/model.onnx'),
        every_computation_node_identical_to_scored049=True,only_changed_initializer=changed[0],
        learned_weights_and_internal_bias_unchanged=True,total1_bias=.5,relative_extra_vs049=.15,
        new_add_nodes_vs049=0,source_windows=960,source_prediction_changes_vs049=len(changes),
        corrected_vs049=int(((before!=labels)&(after==labels)).sum()),broken_vs049=int(((before==labels)&(after!=labels)).sum()),
        correct_vs049=int((before==labels).sum()),candidate_correct=int((after==labels).sum()),independent_test=False)
    shared.save(WORK/'export/comparison049.json',proof)
    rows=shared.read(WORK/'rows.json')
    shared.save(WORK/'export/changes_vs049.json',[dict(**rows[i],truth=int(labels[i]),old_prediction=int(before[i]),prediction=int(after[i])) for i in changes])
    assert shared.sha(BEST)==BEST_HASH
    print('COMPARISON049',proof,flush=True)


if __name__=='__main__':main()
