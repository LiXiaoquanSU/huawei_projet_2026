"""Three-way record-disjoint calibration experiment; no platform submission.

Each fresh model sees only the fit records. Five whole records (one/class)
calibrate that exact model; the entire outer room remains unseen by both.
"""
import csv
import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import blosc2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
WORK = Path('D:/huawei2026-experiments/protected_context_20261024')
CACHE = Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/tta_policy_20261024'))
from model import signal_targets
from utils import task_loss
from common import permutation_indices

spec = importlib.util.spec_from_file_location('protected_context_network', ROOT / 'experiments/global_local_20261021/network.py')
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def split(rows, room):
    test = [i for i, row in enumerate(rows) if row['room'] == room]
    eligible = [i for i, row in enumerate(rows) if row['room'] not in (room, 'unknown')]
    records = {}
    for i in eligible:
        row = rows[i]; records[row['identifier']] = int(row['headcount'])
    chosen = []
    for label in range(5):
        options = sorted([identifier for identifier, y in records.items() if y == label],
                         key=lambda identifier: hashlib.sha256(('calibration17:' + identifier).encode()).hexdigest())
        assert len(options) >= 2, (room, label, 'not enough independent fit/calibration records')
        chosen.append(options[0])
    calibration = [i for i in eligible if rows[i]['identifier'] in chosen]
    fit = [i for i in eligible if rows[i]['identifier'] not in chosen]
    groups = [{rows[i]['identifier'] for i in part} for part in (fit, calibration, test)]
    assert all(groups[a].isdisjoint(groups[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    for part in (fit, calibration):
        assert {int(rows[i]['headcount']) for i in part} == set(range(5))
    return fit, calibration, test


def reverse_features(data):
    path = WORK / 'reverse_features.pt'
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=True)
        assert value['cache_sha256'] == sha(CACHE)
        return value['t'], value['s']
    model = network.build('baseline').cuda().eval()
    tt = torch.empty_like(data['t']); ss = torch.empty_like(data['s'])
    byid = {}
    for i, row in enumerate(data['rows']):
        byid.setdefault(row['identifier'], []).append(i)
    reverse_dt = torch.cat([torch.arange(598, -1, -1), torch.tensor([599])]).cuda()
    raw = Path('D:/huawei2026-data/labeled')
    for identifier, ix in byid.items():
        csi = blosc2.load_array(str(raw / f'CSI_{identifier}.b2'))
        ati = blosc2.load_array(str(raw / f'ATI_{identifier}.b2'))
        for offset in range(0, len(ix), 2):
            chosen = ix[offset:offset + 2]
            xs, ds = [], []
            for i in chosen:
                row = data['rows'][i]; start, stop = int(row['start']), int(row['stop'])
                xs.append(csi[start:stop].view(np.float32).reshape(600, 1, 3, 1024, 2).copy())
                dt = ati[start:stop].copy(); dt[-1] = 0; ds.append(dt)
            x = torch.from_numpy(np.stack(xs)).cuda().double()
            dt = torch.from_numpy(np.stack(ds)).cuda().double()
            with torch.no_grad():
                t, s = model.preprocess(x.flip(1), dt.index_select(1, reverse_dt))
                tt[chosen] = t.float().cpu(); ss[chosen] = s.float().cpu()
                if offset == 0 and identifier == next(iter(byid)):
                    original_t, original_s = model.preprocess(x, dt)
                    assert float((original_t.float().cpu() - data['t'][chosen]).abs().max()) < .001
                    assert float((original_s.float().cpu() - data['s'][chosen]).abs().max()) < .001
        del csi, ati
        print('REVERSE_CACHE', identifier, flush=True)
    torch.save(dict(t=tt, s=ss, cache_sha256=sha(CACHE), rows=data['rows']), path)
    del model; gc.collect(); torch.cuda.empty_cache()
    return tt, ss


def fit(data, architecture, room, initialization, fit_ix):
    folder = WORK / 'fits' / f'{architecture}_s{initialization}_{room}'
    checkpoint = folder / 'final.pt'
    model = network.build(architecture).cuda()
    if checkpoint.exists():
        cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert cp['fit_identifiers'] == sorted({data['rows'][i]['identifier'] for i in fit_ix})
        model.load_state_dict(cp['model_state']); return model.eval()
    seed(initialization)
    model = network.build(architecture).cuda()
    rows = data['rows']; groups = {}
    for i in fit_ix:
        key = int(data['y'][i]), rows[i]['room']; groups[key] = groups.get(key, 0) + 1
    domains = {label: len({room for y, room in groups if y == label}) for label in range(5)}
    weights = [1 / (groups[(int(data['y'][i]), rows[i]['room'])] * domains[int(data['y'][i])]) for i in fit_ix]
    sampler = WeightedRandomSampler(weights, len(fit_ix), generator=torch.Generator().manual_seed(42))
    loader = DataLoader(TensorDataset(data['t'][fit_ix], data['s'][fit_ix], data['y'][fit_ix],
                                    signal_targets(data['t'][fit_ix])), batch_size=32, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0004, weight_decay=.01)
    history = []; seed(42)
    for epoch in range(5):
        model.train(); losses = []; start = time.perf_counter()
        for t, summary, y, targets in loader:
            t, summary, y, targets = [v.cuda() for v in (t, summary, y, targets)]
            optimizer.zero_grad(set_to_none=True)
            tasks = model.forward_tasks(t, summary)
            loss, _ = task_loss(tasks, y, targets, 'physical')
            assert torch.isfinite(loss); loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 3))
            optimizer.step(); losses.append(float(loss))
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), seconds=time.perf_counter() - start))
        save(folder / 'history.json', history)
        print('EPOCH', architecture, room, initialization, history[-1], flush=True)
    torch.save(dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    fit_identifiers=sorted({rows[i]['identifier'] for i in fit_ix}), seed=initialization,
                    architecture=architecture, epochs=5), checkpoint)
    del optimizer, loader; gc.collect(); torch.cuda.empty_cache()
    return model.eval()


def predict(model, data, reverse, indices, views):
    # Match the eventual half-storage/float-compute precision for both models.
    with torch.no_grad():
        for value in model.state_dict().values():
            if value.is_floating_point() and value.numel() >= 1024:
                value.copy_(value.half().float())
    result = []
    index, signs, summary_index = [v.cuda() for v in permutation_indices((2, 1, 0))]
    with torch.no_grad():
        for offset in range(0, len(indices), 8):
            ix = indices[offset:offset + 8]
            t, s = data['t'][ix].cuda(), data['s'][ix].cuda()
            if views:
                rt, rs = reverse[0][ix].cuda(), reverse[1][ix].cuda()
                pairs = [(t, s), (rt, rs), (t.index_select(1, index) * signs, s.index_select(1, summary_index)),
                         (rt.index_select(1, index) * signs, rs.index_select(1, summary_index))]
                z = torch.stack([model.forward_features(a, b) for a, b in pairs], 1).reshape(-1, 2, 2, 5)
            else:
                z = model.forward_features(t, s)
            result.append(z.cpu())
    return torch.cat(result)


def calibrate(logp, labels):
    # Balanced calibration contains exactly one whole source record per class.
    assert torch.bincount(labels, minlength=5).min() > 0
    raw_bias = torch.zeros(5, requires_grad=True)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([raw_bias, log_temperature], lr=.03)
    for _ in range(150):
        bias = 2 * raw_bias.tanh(); bias = bias - bias.mean()
        temperature = log_temperature.exp().clamp(.5, 3.)
        loss = torch.nn.functional.cross_entropy(logp / temperature + bias, labels)
        loss = loss + .05 * bias.square().mean() + .05 * log_temperature.square()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    bias = 2 * raw_bias.detach().tanh(); bias = bias - bias.mean()
    return dict(bias=bias.tolist(), temperature=float(log_temperature.detach().exp().clamp(.5, 3.)),
                samples=len(labels), method='fixed150step regularized balanced calibration; no outer-label fitting')


def corrected(logp, calibration):
    return (logp / calibration['temperature'] + torch.tensor(calibration['bias'])).softmax(-1)


def baseline(raw, calibration):
    identity = corrected(raw[:, 0].exp().mean(1).clamp_min(1e-12).log(), calibration)
    average = corrected(raw.exp().mean((1, 2)).clamp_min(1e-12).log(), calibration)
    labels = identity.argmax(-1)
    return torch.where(((labels == 0) | (labels == 2))[:, None], identity, average)


def metric(probabilities, labels, indices, rows, reference):
    prediction = probabilities.argmax(-1); base = reference.argmax(-1)
    cm = torch.bincount(labels * 5 + prediction, minlength=25).reshape(5, 5)
    present = cm.sum(1) > 0
    counts = torch.bincount(prediction, minlength=5)
    records = []
    for identifier in sorted({rows[i]['identifier'] for i in indices}):
        local = torch.tensor([j for j, i in enumerate(indices) if rows[i]['identifier'] == identifier])
        records.append(dict(identifier=identifier, label=int(labels[local[0]]), windows=len(local),
            accuracy=float((prediction[local] == labels[local]).float().mean()),
            corrected=int(((base[local] != labels[local]) & (prediction[local] == labels[local])).sum()),
            broken=int(((base[local] == labels[local]) & (prediction[local] != labels[local])).sum())))
    return dict(accuracy=float((prediction == labels).float().mean()),
        macro_recall_present=float((cm.diag()[present] / cm.sum(1)[present]).float().mean()),
        confusion_matrix=cm.tolist(), predicted_counts=counts.tolist(), records=records,
        corrected=int(((base != labels) & (prediction == labels)).sum()),
        broken=int(((base == labels) & (prediction != labels)).sum()),
        single_class_collapse=bool(counts.max() == len(labels)))


def fold(data, reverse, room, initialization):
    folder = WORK / 'results' / f's{initialization}_{room}'
    path = folder / 'metrics.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    fit_ix, cal_ix, test_ix = split(data['rows'], room)
    save(folder / 'manifest.json', dict(fit=[data['rows'][i] for i in fit_ix],
        calibration=[data['rows'][i] for i in cal_ix], test=[data['rows'][i] for i in test_ix],
        disjoint_source_records=True, unknown_excluded=True))
    predictions = {}
    for architecture in ('baseline', 'context_events'):
        model = fit(data, architecture, room, initialization, fit_ix)
        predictions[architecture] = [predict(model, data, reverse, ix, architecture == 'baseline') for ix in (cal_ix, test_ix)]
        del model; gc.collect(); torch.cuda.empty_cache()
    raw_cal, raw_test = predictions['baseline']
    context_cal, context_test = predictions['context_events']
    cal_labels, labels = data['y'][cal_ix], data['y'][test_ix]
    ca = calibrate(raw_cal.exp().mean((1, 2)).clamp_min(1e-12).log(), cal_labels)
    cb = calibrate(context_cal, cal_labels)
    zero = dict(bias=[0.] * 5, temperature=1.)
    a, b = baseline(raw_test, ca), corrected(context_test, cb)
    protect = (a.argmax(-1) == 0) | (a.argmax(-1) == 2)
    mixture = .75 * a + .25 * b
    guarded = torch.where(protect[:, None], a, mixture)
    agree = a.argmax(-1) == b.argmax(-1)
    agreement_control = torch.where(agree[:, None], mixture, a)
    assert torch.equal(agreement_control.argmax(-1), a.argmax(-1))
    methods = dict(baseline_zero=baseline(raw_test, zero), baseline_calibrated=a,
        context_zero=context_test.softmax(-1), context_calibrated=b,
        protected_context25=guarded, unprotected_context25_diagnostic=mixture,
        agreement_control=agreement_control)
    result = dict(room=room, seed=initialization, calibration=dict(baseline=ca, context=cb),
        methods={name: metric(p, labels, test_ix, data['rows'], a) for name, p in methods.items()},
        complementary_oracle_accuracy=float(((a.argmax(-1) == labels) | (b.argmax(-1) == labels)).float().mean()),
        disagreement_windows=int((a.argmax(-1) != b.argmax(-1)).sum()), guard_windows=int(protect.sum()))
    save(path, result)
    np.savez_compressed(folder / 'predictions.npz', labels=labels.numpy(), baseline_raw=raw_test.numpy(),
        context_raw=context_test.numpy(), calibration_baseline=raw_cal.numpy(), calibration_context=context_cal.numpy(),
        calibration_labels=cal_labels.numpy(), **{name: p.numpy() for name, p in methods.items()})
    print('FOLD_COMPLETE', room, initialization, {name: value['accuracy'] for name, value in result['methods'].items()}, flush=True)
    return result


def assess(results):
    development = [r for r in results if r['room'] != 'e1']
    stress = next(r for r in results if r['room'] == 'e1')
    methods = {}
    for name in development[0]['methods']:
        cm = np.sum([r['methods'][name]['confusion_matrix'] for r in development], axis=0)
        recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
        methods[name] = dict(room_mean=float(np.mean([r['methods'][name]['accuracy'] for r in development])),
            macro_recall=float(recall.mean()), recall=recall.tolist(), confusion_matrix=cm.tolist())
    base, candidate = methods['baseline_calibrated'], methods['protected_context25']
    reasons = []
    if candidate['room_mean'] < base['room_mean'] + .01:
        reasons.append('room mean improvement below1 percentage point')
    if candidate['macro_recall'] <= base['macro_recall']:
        reasons.append('pooled macro recall not improved')
    if any(a < b - 1e-8 for a, b in zip(candidate['recall'], base['recall'])):
        reasons.append('at least one class recall regressed')
    for r in development:
        a, b = r['methods']['protected_context25'], r['methods']['baseline_calibrated']
        if a['accuracy'] < b['accuracy'] - .02:
            reasons.append(f"room {r['room']} regressed over2 points")
        if a['single_class_collapse']:
            reasons.append(f"room {r['room']} collapsed to one predicted class")
    a, b = stress['methods']['protected_context25'], stress['methods']['baseline_calibrated']
    if a['accuracy'] < b['accuracy'] or a['accuracy'] == 0 or a['predicted_counts'][0] == 0 or a['predicted_counts'][1] == 0:
        reasons.append('e1 stress failed to retain accuracy and predict both present classes')
    return dict(methods=methods, eligible_for_second_seed=not reasons, rejection_reasons=reasons,
                stress=stress, platform_prediction=None)


def main():
    torch.set_num_threads(4); torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available()
    WORK.mkdir(parents=True, exist_ok=True)
    plan = dict(method='0.47-style four-view proxy plus single-view context expert; protected fixed75/25 fusion',
        provenance='Fresh proxy training, not actual .47 full-data weights.',
        split='outer entire room; inner calibration one source record/class; fit excludes both',
        fixed_fit=dict(epochs=5, learning_rate=.0004, batch_size=32, weight_decay=.01),
        development=['a1', 'c1', 'g1'], stress='e1', seed=42, second_seed=2026,
        selection='Predeclared gates in assess; no blend/temperature/bias hyperparameter search on outer rooms.',
        policy='No candidate export on failure. A second seed is run only if seed42 passes all gates.',
        limitations=['Few confounded rooms, repeatedly used outer development records.',
            'Calibration holds out five original records from each fit; fewer training records than historical full-fold fits.',
            'One calibration record per class cannot establish room-invariant calibration.',
            'Proxy results cannot be assigned to actual .47 or other full-data checkpoints.'])
    save(WORK / 'plan.json', plan)
    data = torch.load(CACHE, map_location='cpu', weights_only=True)
    data.pop('absolute', None)
    for name, digest in data['source_hashes'].items():
        assert sha(ROOT / 'src' / name) == digest
    for room in ('a1', 'c1', 'g1', 'e1'):
        split(data['rows'], room)
    save(WORK / 'source_manifest.json', dict(cache_sha256=sha(CACHE),
        files={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/model.py', ROOT / 'src/features.py',
            ROOT / 'src/utils.py', ROOT / 'experiments/global_local_20261021/network.py',
            ROOT / 'experiments/structure_events_20261021/network.py']}))
    reverse = reverse_features(data)
    results = [fold(data, reverse, room, 42) for room in ('a1', 'c1', 'g1', 'e1')]
    assessment = assess(results)
    save(WORK / 'seed42_assessment.json', assessment)
    if assessment['eligible_for_second_seed']:
        replica = [fold(data, reverse, room, 2026) for room in ('a1', 'c1', 'g1', 'e1')]
        save(WORK / 'seed2026_assessment.json', assess(replica))
    save(WORK / 'complete.json', dict(status='completed', assessment=assessment,
        new_training_fits=8 * (2 if assessment['eligible_for_second_seed'] else 1),
        candidate_exported=False, next_action='Review full per-room evidence; actual .47 remains best; no automatic export or submission.'))
    save(ROOT / 'checks/protected_context_20261024.json', dict(artifact_root=str(WORK), **assessment,
        new_training_fits=8 * (2 if assessment['eligible_for_second_seed'] else 1), candidate_exported=False))
    print('EXPERIMENT_COMPLETE', json.dumps(dict(summary=assessment['methods'], rejected=assessment['rejection_reasons'])), flush=True)


if __name__ == '__main__':
    main()
