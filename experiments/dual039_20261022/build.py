"""Fixed two-model probability ensemble; half storage, float32 CPU computation.

No fitting, label-based routing, blend-weight search, or new training is used.
Run on the project Windows machine; source models and data stay local.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import blosc2
import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('D:/huawei2026-experiments/global_local_20261021')
SPECS = [
    ('cross_events', 42, '01_cross_events_seed42.onnx', '5553bdec10f5faeab55759aab5750c522634ae149a4730bbe185cf26710ea24d'),
    ('context_events', 2026, '02_context_events_seed2026.onnx', 'f2ed6fb013e74cdf8d22f3597d11250df517f73a852048603efa64993b50cfb8'),
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def probability(x):
    p = np.exp(np.asarray(x, dtype=np.float64) - np.max(x))
    return p / p.sum()


def development():
    result = []
    for seeds in ((42, 2026), (2026, 42)):
        rooms = []
        for fold in ('a1', 'c1', 'g1'):
            a, b = [rows(SOURCE/'fits'/f'{name}_s{seed}_{fold}/predictions.csv')
                    for (name, *_), seed in zip(SPECS, seeds)]
            assert len(a) == len(b)
            cm = [np.zeros((5, 5), dtype=int) for _ in range(3)]
            disagreement = 0
            for x, y in zip(a, b):
                assert all(x[k] == y[k] for k in ('identifier', 'start', 'stop', 'headcount'))
                pa, pb = [probability([float(z[f'log_probability_{i}']) for i in range(5)]) for z in (x, y)]
                preds = [pa.argmax(), pb.argmax(), (pa+pb).argmax()]
                for matrix, pred in zip(cm, preds):
                    matrix[int(x['headcount']), pred] += 1
                disagreement += preds[0] != preds[1]
            rooms.append(dict(room=fold, cross_accuracy=float(np.trace(cm[0])/cm[0].sum()),
                              context_accuracy=float(np.trace(cm[1])/cm[1].sum()),
                              ensemble_accuracy=float(np.trace(cm[2])/cm[2].sum()),
                              confusion_matrix=cm[2].tolist(), disagreement_windows=int(disagreement)))
        result.append(dict(seeds=seeds, rooms=rooms, **{k:float(np.mean([v[k] for v in rooms]))
                      for k in ('cross_accuracy', 'context_accuracy', 'ensemble_accuracy')}))
    return dict(method='fixed 0.5 probability average; no parameter search; reused historical development folds',
                results=result, note='Not an independent new test; not a forecast of platform accuracy.')


def assemble(work):
    models = []
    compression = []
    for index, (name, seed, filename, expected) in enumerate(SPECS):
        path = ROOT/'models/global_local_20261021'/filename
        assert sha(path) == expected
        graph = compose.add_prefix(onnx.load(path), f'branch{index}_', rename_inputs=False)
        cast_nodes, replaced = [], []
        count = 0
        for tensor in graph.graph.initializer:
            if tensor.data_type == TensorProto.FLOAT and np.prod(tensor.dims) >= 1024:
                array = numpy_helper.to_array(tensor)
                half = array.astype(np.float16)
                assert np.isfinite(half).all()
                stored_name = tensor.name + '_storage16'
                replaced.append(numpy_helper.from_array(half, stored_name))
                cast_nodes.append(helper.make_node('Cast', [stored_name], [tensor.name],
                                                  to=TensorProto.FLOAT, name=stored_name+'_cast'))
                count += int(array.size)
            else:
                replaced.append(tensor)
        del graph.graph.initializer[:]
        graph.graph.initializer.extend(replaced)
        original_nodes = list(graph.graph.node)
        del graph.graph.node[:]
        graph.graph.node.extend(cast_nodes + original_nodes)
        compression.append(dict(architecture=name, seed=seed, source_sha256=expected, half_stored_values=count))
        models.append(graph)
    a, b = models
    assert [(x.name, x.type.SerializeToString()) for x in a.graph.input] == [(x.name, x.type.SerializeToString()) for x in b.graph.input]
    assert [(x.domain, x.version) for x in a.opset_import] == [(x.domain, x.version) for x in b.opset_import]
    assert not a.functions and not b.functions
    nodes = list(a.graph.node) + list(b.graph.node)
    for i, model in enumerate(models):
        nodes.append(helper.make_node('Softmax', [model.graph.output[0].name], [f'prob{i}'], axis=0))
    nodes.extend([
        helper.make_node('Add', ['prob0', 'prob1'], ['sum_prob']),
        helper.make_node('Mul', ['sum_prob', 'half'], ['mean_prob']),
        helper.make_node('Clip', ['mean_prob', 'floor'], ['safe_prob']),
        helper.make_node('Log', ['safe_prob'], ['logits']),
    ])
    initializers = list(a.graph.initializer) + list(b.graph.initializer) + [
        numpy_helper.from_array(np.array(.5, np.float32), 'half'),
        numpy_helper.from_array(np.array(1e-12, np.float32), 'floor'),
    ]
    graph = helper.make_graph(nodes, 'dual039_equal_probability', list(a.graph.input),
                              [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [5])],
                              initializers, value_info=list(a.graph.value_info)+list(b.graph.value_info))
    model = helper.make_model(graph, opset_imports=list(a.opset_import), producer_name='huawei2026-dual039')
    model.ir_version = a.ir_version
    onnx.checker.check_model(model)
    onnx.save_model(model, work/'model.onnx', save_as_external_data=False)
    return compression


def references(work):
    # Existing independent FP32 exports provide exact reference predictions.
    both = [json.loads((SOURCE/'exports'/f'{name}_s{seed}'/'references.json').read_text()) for name, seed, *_ in SPECS]
    assert len(both[0]) == len(both[1])
    refs = []
    for x, y in zip(*both):
        assert x['input'] == y['input']
        refs.append(dict(input=x['input'], logits=np.log(np.maximum((probability(x['logits'])+probability(y['logits']))*.5, 1e-12)).tolist()))
    save(work/'references.json', refs)


def real_input_audit(work):
    """Compare compressed ONNX with original full precision on real CSI.

    Labels only select coverage probes; these all-data checkpoints cannot yield
    an unbiased accuracy estimate on these samples.
    """
    spec = importlib.util.spec_from_file_location('dual_reference', ROOT/'experiments/global_local_20261021/network.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    models = []
    for name, seed, *_ in SPECS:
        model = module.build(name).eval().cuda()
        cp = torch.load(SOURCE/'fits'/f'{name}_s{seed}_full/final.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(cp['model_state'])
        models.append(model)
    raw = Path('D:/huawei2026-data/labeled')
    selected, rooms_seen, labels_seen = [], set(), set()
    for row in rows(raw/'samples.csv'):
        if row['room'] not in rooms_seen or row['headcount'] not in labels_seen:
            selected.append(row)
            rooms_seen.add(row['room']); labels_seen.add(row['headcount'])
    options = ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
    session = ort.InferenceSession(str(work/'model.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    results = []
    for row in selected:
        csi = blosc2.load_array(str(raw/f"CSI_{row['identifier']}.b2"))
        ati = blosc2.load_array(str(raw/f"ATI_{row['identifier']}.b2"))
        for start in (0, 5700, 11400):
            x = csi[start:start+600].view(np.float32).reshape(600,1,3,1024,2).copy()
            dt = ati[start:start+600].copy(); dt[-1]=0
            with torch.no_grad():
                tx, td = torch.from_numpy(x).cuda(), torch.from_numpy(dt).cuda()
                t, s = models[0].preprocess(tx[None].double(),td[None].double())
                p = sum(m.forward_features(t.float(),s.float()).softmax(-1) for m in models)*.5
                reference = p[0].cpu().numpy()
            actual = session.run(['logits'], dict(frames=x,intervals=dt))[0]
            predicted = probability(actual)
            error = float(np.max(np.abs(reference-predicted)))
            margin = float(np.sort(reference)[-1]-np.sort(reference)[-2])
            same = int(reference.argmax()) == int(predicted.argmax())
            assert error < .002 and (same or margin < .002), (row,error,margin)
            results.append(dict(identifier=row['identifier'], start=start, headcount=row['headcount'],
                                probability_max_error=error, argmax_matches=same, reference_margin=margin))
        del csi, ati
        print('RAW_CHECK', row['identifier'], flush=True)
    del session, models
    gc.collect(); torch.cuda.empty_cache()
    return dict(status='pass', windows=len(results), rooms=sorted(rooms_seen), labels=sorted(labels_seen),
                cases=results, note='Numerical probes only; not held-out accuracy.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=Path('D:/huawei2026-experiments/dual039_20261022'))
    args = parser.parse_args(); work=args.work
    work.mkdir(parents=True, exist_ok=True)
    dev = development(); save(work/'development.json', dev)
    print('DEVELOPMENT', json.dumps(dev), flush=True)
    compression = assemble(work); references(work)
    print('EXPORTED', (work/'model.onnx').stat().st_size, flush=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'tools/check_candidate.py'), str(work)], check=True)
    audit = real_input_audit(work); save(work/'real_input_audit.json', audit)
    resource = json.loads((work/'resource_check.json').read_text())
    meta = dict(architecture='dual039_equal_probability', method='0.5 cross42 probability + 0.5 context2026 probability',
                platform_score=None, status='research_candidate_platform_unverified', bytes=resource['bytes'],
                sha256=resource['sha256'], compression=compression, development=dev, resource=resource,
                numerical_audit=audit, training='No new training. Reuses two original platform .39 checkpoints.',
                storage='Large float32 initializers rounded to float16, cast back to float32 before computation.',
                limitations='Development gain is small and inconsistent across seeds; +0.05 platform gain is not established.')
    save(work/'candidate.json', meta)
    print('COMPLETE', meta['sha256'], flush=True)


if __name__ == '__main__':
    main()
