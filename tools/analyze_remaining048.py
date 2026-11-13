"""Diagnose the306 remaining errors from cached actual ONNX outputs; no fitting."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('D:/huawei2026-experiments/remaining048_20261024')
AUDIT = Path('D:/huawei2026-experiments/actual047_audit_20261024')
NEW = Path('D:/huawei2026-experiments/original_one_guard_20261024')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    with (AUDIT / 'all_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        old = list(csv.DictReader(stream))
    current = {}
    for path in (NEW / 'validation_records').glob('*.json'):
        for row in read(path)['windows']:
            current[row['identifier'], row['start']] = row
    assert len(old) == len(current) == 960
    registry = read(ROOT / 'models/registry.json')
    bias = np.array(next(m['bias'] for m in registry['retained_models'] if m['id'] == 'best_044'))
    combined = []
    for row in old:
        actual = current[row['identifier'], int(row['start'])]
        assert actual['truth'] == int(row['truth'])
        base = np.array([float(row[f'base044_logit_{i}']) for i in range(5)])
        combined.append(dict(identifier=row['identifier'], room=row['room'], truth=int(row['truth']),
            start=int(row['start']), stop=int(row['stop']), prediction=actual['prediction'],
            original044=int(row['base044_prediction']), light047=int(row['light047_prediction']),
            full047=int(row['full047_prediction']), original_unbiased_diagnostic=int((base - bias).argmax()),
            amplitude_rms=float(row['amplitude_rms']), relative_frame_change=float(row['relative_frame_change']),
            duration_s=float(row['duration_s']), max_interval_us=float(row['max_interval_us'])))
    errors = [row for row in combined if row['truth'] != row['prediction']]
    assert len(errors) == 306
    by_class = {}
    for label in range(5):
        all_class = [r for r in combined if r['truth'] == label]
        bad_class = [r for r in errors if r['truth'] == label]
        by_class[str(label)] = dict(total=len(all_class), errors=len(bad_class),
            correct=len(all_class) - len(bad_class),
            error_destinations={str(k): v for k, v in sorted(Counter(r['prediction'] for r in bad_class).items())},
            rooms=dict(sorted(Counter(r['room'] for r in bad_class).items())))
    by_record = []
    for identifier in sorted({r['identifier'] for r in combined}):
        rr = [r for r in combined if r['identifier'] == identifier]
        bad = [r for r in rr if r['truth'] != r['prediction']]
        by_record.append(dict(identifier=identifier, room=rr[0]['room'], truth=rr[0]['truth'],
            errors=len(bad), windows=len(rr),
            predictions=dict(sorted(Counter(r['prediction'] for r in rr).items()))))
    by_record.sort(key=lambda r: (-r['errors'], r['identifier']))
    y = np.array([r['truth'] for r in errors])
    oracle_candidates = ('original044', 'full047', 'original_unbiased_diagnostic')
    rescue = {name: int(sum(r[name] == r['truth'] for r in errors)) for name in oracle_candidates}
    rescue['any_of_these_label_oracle'] = int(sum(any(r[name] == r['truth'] for name in oracle_candidates) for r in errors))
    result = dict(platform_score=.48, local_correct=654, local_total=960, local_error_count=306,
        in_sample_error_rate=306/960, independent_test=False,
        by_class=by_class, by_record=by_record,
        augmentation_residual=dict(original044_correct_but048_wrong=rescue['original044'],
            original044_also_wrong=306-rescue['original044']),
        diagnostic_rescuable_errors=rescue,
        joint_bias_diagnostic_by_true_class={str(k): sum(r['original_unbiased_diagnostic']==r['truth'] for r in errors if r['truth']==k) for k in range(5)},
        notes=['All counts describe training-source data, not hidden errors.',
            'Oracle rescue counts use known labels and cannot be deployed as a routing rule.',
            'Unbiased diagnostic subtracts044 final bias only; not exact original037 file.',
            'Signal change is not a true speed, movement or static-person label.'])
    WORK.mkdir(parents=True, exist_ok=True)
    save(WORK / 'summary.json', result)
    save(ROOT / 'checks/remaining048_summary.json', result)
    for name, data in [('remaining_errors.csv', errors), ('record_errors.csv', by_record)]:
        with (WORK / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    save(WORK / 'joined_windows.json', combined)
    print(json.dumps({k: result[k] for k in ('by_class','augmentation_residual','diagnostic_rescuable_errors','joint_bias_diagnostic_by_true_class')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
