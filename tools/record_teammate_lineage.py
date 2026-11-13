"""Record the teammate explanation supplied by the user; leave ONNX bytes intact."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    registry_path = ROOT / 'models/registry.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    ids = {'best_047_light': '49130', 'best_047_full': '49131'}
    for collection in ('retained_models', 'tied_current_best'):
        for model in registry[collection]:
            if model['id'] in ids:
                model['submission_id'] = ids[model['id']]
    registry['current_best']['submission_id'] = '49130'
    registry['teammate_lineage_user_confirmed'] = [
        dict(stage='original Udirect', platform_score=0.44, path='models/best_044/model.onnx'),
        dict(stage='original plus time reversal', platform_score=0.45, submission_id='49108', artifact_received=False),
        dict(stage='four guarded views', platform_score=0.47, submission_id='49130', path='models/best_047_light/model.onnx'),
        dict(stage='twelve guarded views', platform_score=0.47, submission_id='49131', path='models/best_047_full/model.onnx')]
    registry['lineage_source'] = 'User relayed teammate explanation 2026-10-24; original structure/weights unchanged, inference transforms, fusion and protection added.'
    save(registry_path, registry)
    for name, sid in ids.items():
        path = ROOT / f'models/{name}/model.json'
        meta = json.loads(path.read_text(encoding='utf-8-sig'))
        meta.update(submission_id=sid, lineage_source=registry['lineage_source'])
        save(path, meta)
    path = ROOT / 'models/platform_scores.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; table = list(reader)
    for item in table:
        for name, sid in ids.items():
            if item['path'] == f'models/{name}/model.onnx':
                item['submission_id'] = sid
    if not any(item.get('submission_id') == '49108' for item in table):
        table.append(dict(model='队友原模型加时间反转，双视角融合', platform_score='0.45', path='', sha256='',
            source='user relayed teammate explanation 2026-10-24', status='reported_intermediate_artifact_not_received',
            submission_id='49108', availability='not_received', original_path='', archive_manifest=''))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(table)
    print('Recorded user-confirmed lineage: .44 -> 49108/.45 -> 49130/.47 -> 49131/.47. Intermediate binary not received.')


if __name__ == '__main__':
    main()
