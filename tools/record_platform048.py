"""Record the user's .48 platform result without changing the ONNX artifact."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = 'models/original_one_guard_20261024/01_original_one_guard.onnx'
DIGEST = '021769f76948def0ec0d53cbe54d779e84beef925226a3e6498c526d18901324'
SOURCE = 'user-reported platform result 2026-10-24'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    path = ROOT / RELATIVE
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == DIGEST
    meta = read(path.with_suffix('.json'))
    meta.update(id='best_048_original_one', name='原视角1人保护版', platform_score=.48,
        score_source=SOURCE, source=SOURCE, status='current_best', submission_id=None)
    save(path.with_suffix('.json'), meta)
    save(ROOT / 'checks/original_one_guard_20261024.json', meta)

    registry_path = ROOT / 'models/registry.json'; registry = read(registry_path)
    for item in registry['retained_models']:
        if item['id'] in ('best_047_light', 'best_047_full'):
            item['status'] = 'retained_former_best'
    previous = registry.get('tied_current_best', [])
    if previous:
        registry['previous_tied_best_047'] = previous
    registry['tied_current_best'] = []
    best = {key: meta[key] for key in ('id', 'name', 'path', 'sha256', 'bytes', 'platform_score',
        'owner', 'status', 'source', 'submission_id', 'report')}
    registry['current_best'] = best
    registry['best_own'] = dict(best, ownership_note='Our inference policy on teammate047; not a newly trained independent backbone.')
    registry['retained_models'] = [item for item in registry['retained_models'] if item['sha256'] != DIGEST] + [best]
    registry['research_candidates'] = [item for item in registry.get('research_candidates', []) if item['sha256'] != DIGEST]
    registry['next_submission'] = None
    registry['research_status'] = 'original_one_guard_scored048'
    registry['target_platform_score'] = .54
    registry['candidate_note'] = 'Original-view1 guard reported .48, now best; no training or bias changes. .54 remains an unmet platform goal.'
    save(registry_path, registry)

    path = ROOT / 'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; table = list(reader)
    matching = [row for row in table if row['sha256'] == DIGEST]
    assert len(matching) == 1
    matching[0].update(platform_score='0.48', source=SOURCE, status='current_best', availability='retained')
    for row in table:
        if row['path'] in ('models/best_047_light/model.onnx', 'models/best_047_full/model.onnx'):
            row['status'] = 'retained_former_best'
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(table)
    for variant in ('light', 'full'):
        p = ROOT / f'models/best_047_{variant}/model.json'
        older = read(p); older['status'] = 'retained_former_best'; save(p, older)
    print('Recorded .48 as current best; original artifact unchanged; submission id unknown; target remains .54.')


if __name__ == '__main__':
    main()
