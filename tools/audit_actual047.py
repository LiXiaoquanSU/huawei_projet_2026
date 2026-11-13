"""Run the unchanged scored ONNX files on all labelled nonoverlap windows.

This is an in-sample error audit, not held-out accuracy. No fitting or export.
"""
import csv
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import blosc2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/huawei2026-data/labeled')
WORK = Path('D:/huawei2026-experiments/actual047_audit_20261024')
SPECS = {
    'base044': ('models/best_044/model.onnx', 'd2c21c3161b3838247d73b4a23d9ba6091090be39a55627683a41604d04d9063'),
    'light047': ('models/best_047_light/model.onnx', 'd41c3d7cff9a31cde80635ea40df30b4f432c9029b50b40430fff7af00ee62ad'),
    'full047': ('models/best_047_full/model.onnx', '3e0da273c26ad3f966b2610a6bca351c365e111487dcbe6625dc30f90160fc33'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def table(path, values, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in values for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(values)


def probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    p = np.exp(values - values.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def session(path):
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    # Several resident models must not busy-spin while another model computes.
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def metrics(rows, name):
    cm = np.zeros((5, 5), dtype=np.int64)
    for row in rows:
        cm[int(row['truth']), int(row[name + '_prediction'])] += 1
    support = cm.sum(1); predicted = cm.sum(0)
    present = support > 0
    recalls = np.divide(cm.diagonal(), support, out=np.zeros(5, dtype=float), where=present)
    precisions = np.divide(cm.diagonal(), predicted, out=np.zeros(5, dtype=float), where=predicted > 0)
    errors = [row for row in rows if row[name + '_prediction'] != row['truth']]
    return dict(windows=len(rows), correct=int(cm.trace()), errors=len(errors),
        in_sample_accuracy=float(cm.trace() / len(rows)), confusion_matrix=cm.tolist(),
        support=support.tolist(), predicted_counts=predicted.tolist(), recall=recalls.tolist(),
        precision=precisions.tolist(), macro_recall_present=float(recalls[present].mean()),
        errors_with_softmax_ge_080=sum(row[name + '_softmax_max'] >= .8 for row in errors),
        note='Softmax maximum is a score diagnostic, not a verified correctness probability.')


def transition(rows, before, after):
    a = np.array([r[before + '_prediction'] == r['truth'] for r in rows])
    b = np.array([r[after + '_prediction'] == r['truth'] for r in rows])
    changed = np.array([r[before + '_prediction'] != r[after + '_prediction'] for r in rows])
    return dict(corrected=int((~a & b).sum()), broken=int((a & ~b).sum()),
        both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()),
        different_predictions=int(changed.sum()), changed_but_both_wrong=int((changed & ~a & ~b).sum()))


def summarize(samples):
    rows = []; source_hashes = []
    for sample in samples:
        path = WORK / 'records' / (sample['identifier'] + '.json')
        value = json.loads(path.read_text(encoding='utf-8'))
        assert value['model_hashes'] == {k: v[1] for k, v in SPECS.items()}
        rows.extend(value['windows']); source_hashes.append(value['source'])
    assert len(rows) == 960 and len({(r['identifier'], r['start']) for r in rows}) == 960
    summary = dict(scope='Actual scored ONNX models on their original labelled source data; in-sample error audit only.',
        windows=len(rows), records=len(samples), model_hashes={k: v[1] for k, v in SPECS.items()},
        windowing='20 non-overlapping600-frame windows per12000-frame record; original order and padded ATI last value0',
        models={name: metrics(rows, name) for name in SPECS},
        by_room={room: {name: metrics([r for r in rows if r['room'] == room], name) for name in SPECS}
            for room in sorted({r['room'] for r in rows})},
        transitions={a + '_to_' + b: transition(rows, a, b) for a, b in
            [('base044', 'light047'), ('base044', 'full047'), ('light047', 'full047')]},
        transitions_by_class={str(y): transition([r for r in rows if r['truth'] == y], 'light047', 'full047') for y in range(5)},
        full_execution_counts=dict(
            original_onnx=sum(r.get('full047_inference_mode', 'original_onnx') == 'original_onnx' for r in rows),
            proven_protected_branch_reuse=sum(r.get('full047_inference_mode') == 'proven_protected_branch_reuse' for r in rows)),
        source_files=source_hashes,
        platform_scores_unchanged=dict(base044=.44, light047=.47, full047=.47),
        limitations=['These source records were used to fit the underlying network. This is not validation or hidden accuracy.',
            'All20 windows from a record share one occupancy label and are correlated.',
            'Unknown room metadata is retained as unknown, not invented.',
            'Confusion here does not identify the teacher hidden-set error distribution.'])
    records = []
    for sample in samples:
        selected = [r for r in rows if r['identifier'] == sample['identifier']]
        record = dict(**sample, windows=len(selected))
        for name in SPECS:
            mm = metrics(selected, name)
            record[name + '_correct'] = mm['correct']
            record[name + '_accuracy'] = mm['in_sample_accuracy']
            record[name + '_predictions'] = '/'.join(str(v) for v in mm['predicted_counts'])
        records.append(record)
    records.sort(key=lambda r: (r['light047_correct'], r['full047_correct'], r['identifier']))
    errors = [r for r in rows if r['light047_prediction'] != r['truth'] or r['full047_prediction'] != r['truth']]
    table(WORK / 'all_predictions.csv', rows)
    table(WORK / 'errors.csv', errors, list(dict.fromkeys(k for row in rows for k in row)))
    table(WORK / 'record_summary.csv', records)
    summary['worst_records'] = records[:12]
    save(WORK / 'summary.json', summary)
    save(ROOT / 'checks/actual047_in_sample_audit.json', summary)
    return summary


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    samples = read(DATA / 'samples.csv')
    assert len(samples) == 48 and len({s['identifier'] for s in samples}) == 48
    for path, digest in SPECS.values():
        assert sha(ROOT / path) == digest
    plan = dict(scope='in-sample diagnostic of actual submitted artifacts, not held-room proxies',
        samples_csv_sha256=sha(DATA / 'samples.csv'), model_hashes={k: v[1] for k, v in SPECS.items()},
        inference='Unmodified original ONNX files, ONNX Runtime CPU, 4 threads, idle thread spinning disabled; all inputs from original CSI. Light/base execute every window. Full reuses the mathematically identical light output only on its fixed outer0/2 guard with margin>.005, after direct guard verification per source record; all other full outputs run original ONNX. Per-window timings are diagnostic, not a controlled runtime benchmark.',
        labels='Only supplied record headcount labels; no inference of movement or human identities.',
        windows_per_record=20, frames_per_window=600, stride_frames=600,
        new_training=False, parameters_changed=False, source_script_sha256=sha(Path(__file__)))
    save(WORK / 'plan.json', plan)
    shutil.copy2(Path(__file__), WORK / 'audit_actual047_source.py')
    if all((WORK / 'records' / (sample['identifier'] + '.json')).exists() for sample in samples):
        summary = summarize(samples)
        save(WORK / 'complete.json', dict(status='completed', windows=960, records=48,
            new_training=False, original_model_hashes_unchanged=True,
            resumed_completed_records=True, full_execution_counts=summary['full_execution_counts']))
        print('COMPLETE_FROM_CHECKPOINTS', json.dumps(dict(models=summary['models'], transitions=summary['transitions'])), flush=True)
        return
    sessions = {name: session(ROOT / path) for name, (path, _) in SPECS.items()}
    start_time = time.perf_counter(); total = 0; correct = {name: 0 for name in SPECS}
    for sample in samples:
        identifier = sample['identifier']; path = WORK / 'records' / (identifier + '.json')
        label = int(sample['headcount'])
        assert label in range(5) and int(identifier.rsplit('_', 1)[1]) == label
        if path.exists():
            value = json.loads(path.read_text(encoding='utf-8'))
            assert value['model_hashes'] == plan['model_hashes'] and len(value['windows']) == 20
            assert all(r['truth'] == label for r in value['windows'])
            for name in SPECS:
                correct[name] += sum(r[name + '_prediction'] == label for r in value['windows'])
            total += 20
            print('RESUME', identifier, total, flush=True)
            continue
        csi_path, ati_path = [DATA / f'{prefix}_{identifier}.b2' for prefix in ('CSI', 'ATI')]
        source = dict(identifier=identifier, csi_sha256=sha(csi_path), ati_sha256=sha(ati_path))
        csi = blosc2.load_array(str(csi_path)); ati = blosc2.load_array(str(ati_path))
        assert csi.shape == (12000, 1, 3, 1024) and csi.dtype == np.complex64, (identifier, csi.shape, csi.dtype)
        ati = np.asarray(ati).reshape(-1)
        assert len(ati) == 12000
        windows = []; guard_verified = False
        for start in range(0, 12000, 600):
            stop = start + 600
            x = np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600, 1, 3, 1024, 2)
            dt = ati[start:stop].astype(np.float32, copy=True); dt[-1] = 0
            assert np.isfinite(x).all() and np.isfinite(dt).all() and (dt >= 0).all()
            amp = np.abs(csi[start:stop]).astype(np.float64)
            scale = float(np.mean(amp ** 2))
            row = dict(identifier=identifier, room=sample['room'], layout=sample['layout'], truth=label,
                start=start, stop=stop, window_index=start // 600,
                duration_s=float(dt[:-1].sum(dtype=np.float64) / 1e6), max_interval_us=float(dt.max()),
                amplitude_rms=float(np.sqrt(scale)),
                relative_frame_change=float(np.mean(np.diff(amp, axis=0) ** 2) / max(scale, 1e-20)),
                zero_complex_fraction=float(np.mean(csi[start:stop] == 0)))
            outputs = {}
            for name, engine in sessions.items():
                use_guard = name == 'full047' and row['light047_prediction'] in (0, 2) and row['light047_margin'] > .005
                if use_guard and guard_verified:
                    z = outputs['light047'].copy(); elapsed = 0.
                    row['full047_inference_mode'] = 'proven_protected_branch_reuse'
                else:
                    tick = time.perf_counter()
                    z = engine.run(['logits'], dict(frames=x, intervals=dt))[0]
                    elapsed = time.perf_counter() - tick
                    if name == 'full047':
                        row['full047_inference_mode'] = 'original_onnx'
                    if use_guard:
                        assert np.allclose(z, outputs['light047'], atol=2e-5, rtol=2e-5)
                        assert int(z.argmax()) == row['light047_prediction']
                        guard_verified = True
                outputs[name] = z
                assert z.shape == (5,) and z.dtype == np.float32 and np.isfinite(z).all()
                p = probabilities(z); order = np.argsort(p)
                row[name + '_prediction'] = int(z.argmax())
                row[name + '_softmax_max'] = float(p.max())
                row[name + '_margin'] = float(p[order[-1]] - p[order[-2]])
                row[name + '_cpu_s'] = elapsed
                for i, score in enumerate(z):
                    row[name + f'_logit_{i}'] = float(score)
                correct[name] += int(z.argmax() == label)
            windows.append(row); total += 1
        save(path, dict(model_hashes=plan['model_hashes'], source=source, windows=windows))
        progress = dict(completed_windows=total, total_windows=960, last_record=identifier,
            in_sample_correct_counts=correct, elapsed_s=time.perf_counter() - start_time,
            note='partial in-sample counts, no generalization claim')
        save(WORK / 'progress.json', progress)
        print('RECORD', json.dumps(progress), flush=True)
        del csi, ati, amp; gc.collect()
    del sessions; gc.collect()
    summary = summarize(samples)
    for path, digest in SPECS.values():
        assert sha(ROOT / path) == digest
    save(WORK / 'complete.json', dict(status='completed', windows=960, records=48,
        new_training=False, original_model_hashes_unchanged=True,
        inference_wall_seconds=time.perf_counter() - start_time))
    print('COMPLETE', json.dumps(dict(models=summary['models'], transitions=summary['transitions'])), flush=True)


if __name__ == '__main__':
    main()
