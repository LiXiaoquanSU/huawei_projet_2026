"""Record the reported score and diagnose frozen held-room proxy failures.

No training, bias fitting, hidden-label access or submission is performed.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/tta_policy_20261024'))
from common import WORK, scores, decide, precision_spec


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    relative = 'models/tta_policy_20261024/01_symmetric_unprotected.onnx'
    model_path = ROOT / relative
    with model_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == '0d641389b68a658bfcd6f53bf2b535bf08ec33f953768d2ef4a080fcaf993b98'
    source = 'user-reported platform result 2026-10-24'
    status = 'evaluated_below_best'
    sidecar = model_path.with_suffix('.json')
    meta = load(sidecar)
    meta.update(platform_score=0.41, score_source=source, status=status, submission_id=None)
    save(sidecar, meta)

    registry_path = ROOT / 'models/registry.json'
    registry = load(registry_path)
    matched = [item for item in registry['research_candidates'] if item['sha256'] == digest]
    assert len(matched) == 1
    matched[0].update(platform_score=0.41, source=source, status=status, submission_id=None)
    registry['next_submission'] = None
    registry['candidate_note'] = 'Unprotected twelve-view policy scored .41, below both .47 guarded models; local proxy ranking failed to transfer. No new candidate selected.'
    save(registry_path, registry)

    table_path = ROOT / 'models/platform_scores.csv'
    with table_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        table = list(reader)
    matched = [item for item in table if item['sha256'] == digest]
    assert len(matched) == 1
    matched[0].update(platform_score='0.41', source=source, status=status, availability='retained_evaluated_research')
    with table_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    torch.set_num_threads(4)
    bias = precision_spec()[1]
    folds = [('a1', 42), ('c1', 42), ('g1', 42), ('a1', 2026), ('c1', 2026), ('g1', 2026), ('e1', 42)]
    diagnostics = []
    for room, seed in folds:
        with np.load(WORK / 'predictions' / f'{room}_s{seed}.npz') as data:
            raw = torch.from_numpy(data['log_probability'].copy())
            labels = torch.from_numpy(data['labels'].copy())
        q = scores(raw, bias)
        calibration_audit = {}
        for name, fixed_bias in [('teammate044_fixed', bias), ('zero_bias_diagnostic', torch.zeros_like(bias))]:
            qq = scores(raw, fixed_bias)
            calibration_audit[name] = {}
            for policy in ('full', 'unprotected'):
                prediction = decide(qq, policy).argmax(-1)
                calibration_audit[name][policy] = dict(
                    accuracy=float((prediction == labels).float().mean()),
                    predicted_counts=torch.bincount(prediction, minlength=5).tolist())
        full = decide(q, 'full').argmax(-1)
        unprotected = decide(q, 'unprotected').argmax(-1)
        light = decide(q, 'light').argmax(-1)
        protected = (light == 0) | (light == 2)
        changed = full != unprotected
        # With identical views/bias, the two rules only disagree inside the guard.
        assert not bool((changed & ~protected).any())
        correct_full = full == labels
        correct_new = unprotected == labels
        transition = torch.zeros(5, 5, dtype=torch.int64)
        for a, b in zip(full[changed], unprotected[changed]):
            transition[a, b] += 1
        histograms = {}
        for name, vec in [('full', full), ('unprotected', unprotected), ('truth', labels)]:
            histograms[name] = torch.bincount(vec, minlength=5).tolist()
        diagnostics.append(dict(room=room, seed=seed, windows=len(labels), guard_windows=int(protected.sum()),
            changed=int(changed.sum()), corrected=int((~correct_full & correct_new).sum()),
            broken=int((correct_full & ~correct_new).sum()),
            changed_but_both_wrong=int((changed & ~correct_full & ~correct_new).sum()),
            full_to_unprotected_transition=transition.tolist(), class_histograms=histograms,
            calibration_audit=calibration_audit))
    result = dict(platform_score=0.41, platform_score_source=source, sha256=digest,
        compared_best_score=0.47, platform_difference_percentage_points=-6,
        diagnostics=diagnostics,
        limitations=['Held-room predictions use historical proxy weights, not the actual scored full-data weights.',
            'Two seeds reuse the same physical validation records; windows are correlated.',
            'No hidden per-class scores or predictions are available; these diagnostics cannot locate hidden errors.',
            'Candidate implementation also batches/preprocesses differently; previous numerical probes reduce but do not globally exclude implementation differences.'],
        new_training_fits=0)
    save(ROOT / 'checks/unprotected041_review.json', result)
    for item in diagnostics:
        print(json.dumps({k:item[k] for k in ('room','seed','changed','corrected','broken','calibration_audit')}, ensure_ascii=False))
    print('RECORDED', relative, '.41; best remains .47')


if __name__ == '__main__':
    main()
