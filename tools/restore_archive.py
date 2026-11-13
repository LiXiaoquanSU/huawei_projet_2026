"""Preview or restore an explicitly selected archive path without overwriting files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=Path('D:/huawei2026-experiments/project_archive_20261023_044')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=DEFAULT)
    parser.add_argument('--path',help='Original relative file or directory; preview by default')
    parser.add_argument('--apply',action='store_true',help='Restore missing files after all safety checks')
    args=parser.parse_args();archive=args.archive.resolve()
    manifest=json.loads((archive/'manifest.json').read_text(encoding='utf-8'))
    prefix=args.path.replace('\\','/').rstrip('/') if args.path else ''
    selected=[e for e in manifest['entries'] if not prefix or e['path']==prefix or e['path'].startswith(prefix+'/')]
    if not selected:raise ValueError('No matching archived path')
    if args.apply and not prefix:raise ValueError('--apply requires a specific --path')
    plan=[];conflicts=[]
    for entry in selected:
        target=(ROOT/entry['path']).resolve();blob=(archive/entry['blob']).resolve()
        if not target.is_relative_to(ROOT) or not blob.is_relative_to(archive/'blobs'):
            raise ValueError('Archive path escapes the intended directory')
        if target.exists():
            if not target.is_file() or sha(target)!=entry['sha256']:
                conflicts.append(entry['path'])
            continue
        plan.append((entry,target,blob))
    if not args.apply:
        print(json.dumps(dict(mode='preview',matching_files=len(selected),missing_files=len(plan),
                              examples=[e['path'] for e,_,_ in plan[:40]],
                              conflicting_files=conflicts[:40],
                              note='Add --apply to restore a specific selected path.'),indent=2))
        return
    if conflicts:raise FileExistsError(f'Will not overwrite changed files: {conflicts[:10]}')
    for entry,_,blob in plan:
        if not blob.is_file() or sha(blob)!=entry['sha256']:raise ValueError('Missing or corrupted archive blob')
    for entry,target,blob in plan:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(blob,target)
        assert sha(target)==entry['sha256']
    print(json.dumps(dict(mode='restored',files=len(plan),path=prefix)))


if __name__=='__main__':main()
