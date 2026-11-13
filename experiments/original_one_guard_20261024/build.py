"""Append one input-dependent guard to the scored49130 ONNX; no retraining.

Original-view044 predicts1 -> original044 scores, else untouched49130 scores.
All learned weights, bias, input transforms and old guards remain byte-identical.
"""
import csv
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import blosc2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('D:/huawei2026-experiments/original_one_guard_20261024')
AUDIT = Path('D:/huawei2026-experiments/actual047_audit_20261024')
DATA = Path('D:/huawei2026-data/labeled')
SOURCE = ROOT / 'models/best_047_light/model.onnx'
BASE = ROOT / 'models/best_044/model.onnx'
HASHES = {
    'models/best_047_light/model.onnx': 'd41c3d7cff9a31cde80635ea40df30b4f432c9029b50b40430fff7af00ee62ad',
    'models/best_044/model.onnx': 'd2c21c3161b3838247d73b4a23d9ba6091090be39a55627683a41604d04d9063',
}
ORIGINAL = '__rxf_v0___tta_v0_logits'
LEGACY = '__original_one_guard_legacy_logits'
PREFIX = '__original_one_guard_'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def probability(z):
    z = np.asarray(z, dtype=np.float64)
    p = np.exp(z - z.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def session(path):
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def assemble():
    source = onnx.load(SOURCE); model = onnx.ModelProto(); model.CopyFrom(source)
    assert list(model.graph.node[-1].output) == ['logits'] and model.graph.node[-1].op_type == 'Where'
    assert not any(PREFIX in value for node in model.graph.node for value in [*node.input, *node.output])
    assert any(ORIGINAL in node.output for node in model.graph.node)
    assert not any('logits' in node.input for node in model.graph.node)
    model.graph.node[-1].output[0] = LEGACY
    model.graph.node.extend([
        helper.make_node('ArgMax', [ORIGINAL], [PREFIX + 'original_class'], axis=0, keepdims=0,
                         select_last_index=0, name=PREFIX + 'argmax'),
        helper.make_node('Equal', [PREFIX + 'original_class', PREFIX + 'one'], [PREFIX + 'use_original'], name=PREFIX + 'equal'),
        helper.make_node('Where', [PREFIX + 'use_original', ORIGINAL, LEGACY], ['logits'], name=PREFIX + 'select'),
    ])
    model.graph.initializer.append(numpy_helper.from_array(np.array(1, np.int64), PREFIX + 'one'))
    props = {item.key: item.value for item in source.metadata_props}
    props.update(base_sha256=HASHES['models/best_047_light/model.onnx'], base_submission='49130',
        base_accuracy='0.47', platform_accuracy='not evaluated', target_platform_accuracy='0.54 (goal, not prediction)',
        transformation='Preserve original untransformed044 argmax1; otherwise preserve49130 scores. Can override its0/2 guard only on original argmax1.',
        learned_weights='Every49130 initializer preserved byte for byte; only one int64 routing constant added.',
        new_training='false', bias='Original044 fixed bias unchanged; no fitting or threshold search',
        local_evidence='Post-hoc training-source error audit; not independent validation or hidden-score prediction.')
    helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    # Exact preservation proof, including every Constant attribute in each body.
    for old, new in zip(source.graph.node, model.graph.node):
        copied = onnx.NodeProto(); copied.CopyFrom(new)
        if LEGACY in copied.output:
            copied.output[0] = 'logits'
        assert old.SerializeToString() == copied.SerializeToString()
    for old, new in zip(source.graph.initializer, model.graph.initializer):
        assert old.SerializeToString() == new.SerializeToString()
    assert len(model.graph.node) == len(source.graph.node) + 3
    assert len(model.graph.initializer) == len(source.graph.initializer) + 1
    assert model.graph.input == source.graph.input and model.graph.output == source.graph.output
    assert not any(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer)
    destination = WORK / 'model.onnx'
    if destination.exists():
        assert destination.read_bytes() == model.SerializeToString(), 'Do not overwrite a different candidate.'
    else:
        onnx.save_model(model, destination, save_as_external_data=False)
    proof = dict(status='pass', source_sha256=sha(SOURCE), candidate_sha256=sha(destination),
        source_nodes=len(source.graph.node), added_nodes=3, all_original_nodes_preserved=True,
        original_initializers=len(source.graph.initializer), original_initializers_byte_identical=True,
        extra_initializers='One int64 scalar1, routing only', learned_parameters_changed=False,
        original_bias_unchanged=True, same_interface=True, embedded_weights=True)
    save(WORK / 'structure_proof.json', proof)
    return proof


def fixture_references():
    path = WORK / 'references.json'
    if path.exists():
        return
    cases = []
    fixtures = [ROOT / 'checks/smoke_input.npz', ROOT / 'checks/case_gap.npz', 'zero']
    # Isolated engines prevent idle model thread pools competing during checks.
    a = session(BASE)
    for source in fixtures:
        if source == 'zero':
            inputs = dict(frames=np.zeros((600, 1, 3, 1024, 2), np.float32), intervals=np.zeros(600, np.float32))
        else:
            with np.load(source) as d:
                inputs = {k: d[k].copy() for k in ('frames', 'intervals')}
        cases.append(dict(input=str(source), base_logits=a.run(['logits'], inputs)[0].tolist()))
    del a; gc.collect()
    b = session(SOURCE)
    for case in cases:
        if case['input'] == 'zero':
            inputs = dict(frames=np.zeros((600, 1, 3, 1024, 2), np.float32), intervals=np.zeros(600, np.float32))
        else:
            with np.load(case['input']) as d:
                inputs = {k: d[k].copy() for k in ('frames', 'intervals')}
        legacy = b.run(['logits'], inputs)[0]
        original = np.array(case['base_logits'])
        case['logits'] = original.tolist() if int(original.argmax()) == 1 else legacy.tolist()
        case['route'] = 'original_one' if int(original.argmax()) == 1 else 'legacy047'
    del b; gc.collect(); save(path, cases)


def evaluate_raw():
    digest = sha(WORK / 'model.onnx')
    cache = WORK / 'validation.json'
    if cache.exists():
        result = read(cache)
        assert result['candidate_sha256'] == digest
        return result
    actual = rows(AUDIT / 'all_predictions.csv')
    audit = read(AUDIT / 'summary.json')
    assert audit['model_hashes']['base044'] == HASHES['models/best_044/model.onnx']
    assert audit['model_hashes']['light047'] == HASHES['models/best_047_light/model.onnx']
    assert len(actual) == 960
    sources = {item['identifier']: item for item in audit['source_files']}
    by_record = {}
    for row in actual:
        by_record.setdefault(row['identifier'], []).append(row)
    engine = session(WORK / 'model.onnx')
    all_checks = []; started = time.perf_counter()
    for identifier, selected in by_record.items():
        record_path = WORK / 'validation_records' / (identifier + '.json')
        if record_path.exists():
            value = read(record_path)
            assert value['candidate_sha256'] == digest and len(value['windows']) == 20
            all_checks.extend(value['windows'])
            continue
        csi_path, ati_path = [DATA / f'{prefix}_{identifier}.b2' for prefix in ('CSI', 'ATI')]
        assert sha(csi_path) == sources[identifier]['csi_sha256']
        assert sha(ati_path) == sources[identifier]['ati_sha256']
        csi = blosc2.load_array(str(csi_path)); ati = blosc2.load_array(str(ati_path)).reshape(-1)
        checks = []
        for row in selected:
            start, stop = int(row['start']), int(row['stop'])
            x = np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600, 1, 3, 1024, 2)
            dt = ati[start:stop].astype(np.float32, copy=True); dt[-1] = 0
            original = np.array([float(row[f'base044_logit_{i}']) for i in range(5)])
            legacy = np.array([float(row[f'light047_logit_{i}']) for i in range(5)])
            trigger = int(original.argmax()) == 1
            expected = original if trigger else legacy
            actual_z = engine.run(['logits'], dict(frames=x, intervals=dt))[0]
            assert actual_z.shape == (5,) and actual_z.dtype == np.float32 and np.isfinite(actual_z).all()
            error = float(np.max(np.abs(probability(actual_z) - probability(expected))))
            logit_error = float(np.max(np.abs(actual_z - expected)))
            assert error < 1e-4 and logit_error < .001, (identifier, start, error, logit_error)
            assert int(actual_z.argmax()) == int(expected.argmax()), (identifier, start, 'policy mismatch')
            checks.append(dict(identifier=identifier, room=row['room'], start=start, stop=stop,
                truth=int(row['truth']), route='original_one' if trigger else 'legacy047',
                prediction=int(actual_z.argmax()), old_prediction=int(legacy.argmax()),
                probability_max_error=error, logit_max_error=logit_error))
        save(record_path, dict(candidate_sha256=digest, windows=checks))
        all_checks.extend(checks)
        print('VERIFIED', identifier, len(all_checks), '/960', 'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
        del csi, ati; gc.collect()
    del engine; gc.collect()
    assert len(all_checks) == 960
    y = np.array([r['truth'] for r in all_checks]); p = np.array([r['prediction'] for r in all_checks])
    old = np.array([r['old_prediction'] for r in all_checks])
    cm = np.bincount(y * 5 + p, minlength=25).reshape(5, 5)
    result = dict(status='pass', candidate_sha256=digest, windows=960, records=48,
        argmax_matches_expected_policy=960, max_probability_error=max(r['probability_max_error'] for r in all_checks),
        max_logit_error=max(r['logit_max_error'] for r in all_checks),
        triggered=sum(r['route'] == 'original_one' for r in all_checks),
        unchanged_route_checks=sum(r['route'] == 'legacy047' for r in all_checks),
        correct=int((p == y).sum()), in_sample_accuracy=float((p == y).mean()),
        corrected=int(((old != y) & (p == y)).sum()), broken=int(((old == y) & (p != y)).sum()),
        confusion_matrix=cm.tolist(), recall=(cm.diagonal() / cm.sum(1)).tolist(),
        independent_test=False, new_training=False,
        note='Actual exported ONNX on all960 original raw windows. Rule selected post-hoc on training-source labels; no platform score is claimed.')
    assert result['correct'] == 654 and result['corrected'] == 137 and result['broken'] == 2
    save(cache, result)
    return result


def deliver(proof, validation, resource):
    target = ROOT / 'models/original_one_guard_20261024/01_original_one_guard.onnx'
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(WORK / 'model.onnx')
    assert resource['status'] == 'pass' and resource['sha256'] == digest == validation['candidate_sha256']
    if target.exists():
        assert sha(target) == digest
    else:
        shutil.copy2(WORK / 'model.onnx', target)
    if target.with_suffix('.json').exists():
        recorded = read(target.with_suffix('.json'))
        if recorded.get('platform_score') is not None:
            assert recorded['sha256'] == digest
            print('VERIFIED_EXISTING_SCORED_ARTIFACT', str(target), recorded['platform_score'], flush=True)
            return  # Never replace subsequent user-reported scores with build-time unknowns.
    metadata = dict(path=target.relative_to(ROOT).as_posix(), sha256=digest, bytes=target.stat().st_size,
        platform_score=None, submission_id=None, target_platform_score=.54,
        status='candidate_platform_unverified', owner='self_policy_on_teammate047',
        method='Original untransformed044 argmax1 -> original044 logits; otherwise unchanged49130 logits.',
        source_model_sha256=HASHES['models/best_047_light/model.onnx'], source_submission='49130',
        original_bias_unchanged=True, learned_parameters_changed=False, new_training_fits=0,
        structure_proof=proof, validation=validation, resource=resource,
        limitations=['Rule was designed after looking at original training-source errors.',
            'Two formerly correct empty windows became1 in local audit.',
            'The new outer rule can override old0/2 protection when original view predicts1.',
            'No independent or hidden accuracy is available; .54 is a goal only.'],
        report='docs/ORIGINAL_ONE_GUARD_20261024.md')
    save(target.with_suffix('.json'), metadata); save(WORK / 'candidate.json', metadata)
    save(ROOT / 'checks/original_one_guard_20261024.json', metadata)
    registry_path = ROOT / 'models/registry.json'; registry = read(registry_path)
    item = {k: metadata[k] for k in ('path', 'sha256', 'bytes', 'platform_score', 'status', 'method', 'report')}
    registry['research_candidates'] = [r for r in registry.get('research_candidates', []) if r['path'] != item['path']] + [item]
    registry['next_submission'] = item['path']
    registry['target_platform_score'] = .54
    registry['research_status'] = 'original_one_guard_ready_for_platform_test'
    registry['candidate_note'] = 'Frozen49130 plus original-view1 guard; actual960-window numerical verification and local resources pass. In-sample68.125%, platform unknown. Best remains .47.'
    save(registry_path, registry)
    path = ROOT / 'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; scores = list(reader)
    if not any(row['sha256'] == digest for row in scores):
        scores.append(dict(model='原视角1人保护，四视角047底座', platform_score='', path=item['path'], sha256=digest,
            source='post-hoc training-source error analysis; platform untested', status='candidate_platform_unverified',
            submission_id='', availability='research_candidate', original_path=item['path'], archive_manifest=''))
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(scores)
    for relative, expected in HASHES.items():
        assert sha(ROOT / relative) == expected
    print('DELIVERED', json.dumps(dict(path=metadata['path'], bytes=metadata['bytes'], sha256=digest,
        in_sample_accuracy=validation['in_sample_accuracy'], platform_score=None)), flush=True)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    for relative, digest in HASHES.items():
        assert sha(ROOT / relative) == digest
    save(WORK / 'plan.json', dict(method='original_one_argmax_guard', source_hashes=HASHES,
        new_training=False, no_threshold_search=True, preserve_all_original_weights=True,
        validation='Actual exported model on960 raw windows against cached independent original044/49130 rule outputs.',
        resources='Isolated tools/check_candidate.py; <100million bytes, <.7s CPU mean, <6GB.',
        scientific_scope='Post-hoc in-sample candidate, platform score unknown.'))
    shutil.copy2(Path(__file__), WORK / 'build_source.py')
    proof = assemble(); print('STRUCTURE_PASS', json.dumps(proof), flush=True)
    fixture_references()
    validation = evaluate_raw()
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'tools/check_candidate.py'), str(WORK)], check=True)
    resource = read(WORK / 'resource_check.json')
    deliver(proof, validation, resource)


if __name__ == '__main__':
    main()
