"""Read public Zenodo metadata and selectively download ZIP members via HTTP ranges."""
import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

WORK = Path('D:/huawei2026-experiments/motion_external_20261021')
RECORD = 'https://zenodo.org/api/records/17225694'

def fetch(url, headers=None):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'RoomSense-research/1.0', **(headers or {})})
            return urllib.request.urlopen(req, timeout=90)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

class RemoteFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.cache = {}
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0: return b''
        start, stop = self.pos, self.pos + n - 1
        key = (start, stop)
        if key not in self.cache:
            # Cache-busting is required for CDNs that otherwise cache the first range.
            u = self.url + ('&' if '?' in self.url else '?') + f'range_start={start}&range_end={stop}'
            with fetch(u, {'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity'}) as r:
                if r.status != 206:
                    raise RuntimeError(f'HTTP Range unsupported: {r.status}; refusing a silent multi-GB download')
                content_range = r.headers.get('Content-Range', '')
                if not content_range.startswith(f'bytes {start}-'):
                    raise RuntimeError(f'Wrong range: {content_range}')
                value = r.read()
            if len(value) != n: raise RuntimeError(f'Short range: {len(value)} != {n}')
            if n < 1024 * 1024: self.cache[key] = value
        else: value = self.cache[key]
        self.pos += n
        return value

def main(args):
    WORK.mkdir(parents=True, exist_ok=True)
    api_path = WORK / 'external/zenodo_record.json'
    if api_path.exists(): record = json.loads(api_path.read_text(encoding='utf-8'))
    else:
        with fetch(RECORD) as r: record = json.load(r)
        save(api_path, record)
    print('LICENSE', record['metadata'].get('license'), flush=True)
    for file in record['files']:
        key = file['key']
        if not key.startswith('80MHz:') or not key.endswith('people.zip'): continue
        count = int(key.split(':')[1].split('people')[0])
        if count not in args.counts: continue
        index_path = WORK / f'external/index_{count}.json'
        url = file['links'].get('self') or file['links']['download']
        stream = RemoteFile(url, file['size'])
        with zipfile.ZipFile(stream) as z:
            entries = [dict(path=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size,
                            crc32=f'{i.CRC:08x}') for i in z.infolist() if not i.is_dir()]
            save(index_path, dict(archive=key, archive_checksum=file['checksum'], members=entries))
            selected = [i for i in z.infolist() if '__MACOSX' not in i.filename and
                        not Path(i.filename).name.startswith('._') and
                        (i.filename.endswith('config.json') or
                         (args.mats and i.filename.endswith('.mat')))]
            print('ARCHIVE', key, 'members', len(entries), 'selected', len(selected),
                  'compressed_MB', round(sum(i.compress_size for i in selected)/1e6, 2), flush=True)
            for i in selected:
                relative = Path(i.filename)
                if relative.is_absolute() or '..' in relative.parts: raise ValueError(i.filename)
                target = WORK / 'external/extracted' / relative
                if target.exists() and target.stat().st_size == i.file_size: continue
                value = z.read(i)  # ZIP CRC verified by zipfile.
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value)
                if i.filename.endswith('config.json'):
                    print(i.filename, value.decode('utf-8-sig')[:1600], flush=True)
                else: print('MAT', i.filename, len(value), flush=True)
        print('INDEX_DONE', count, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mats', action='store_true')
    p.add_argument('--counts', type=int, nargs='+', default=[0,1,2,3,4])
    main(p.parse_args())
