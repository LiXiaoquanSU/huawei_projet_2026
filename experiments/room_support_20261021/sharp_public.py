"""Inspect/download the author's related public SHARP data, not the gated full dataset."""
import argparse,hashlib,html,io,json,re,sys,urllib.request,zipfile,zlib,struct,subprocess,email
from pathlib import Path
WORK=Path('D:/huawei2026-experiments/room_support_20261021/sharp_public')
PAGE='https://researchdata.cab.unipd.it/624/'
class Response(io.BytesIO):
    pass
def request(url,headers=None):
    # Windows curl/Schannel verifies the university certificate using the OS
    # trust store. Do not disable TLS verification for a missing OpenSSL chain.
    hp=WORK/'last_http_headers.txt';args=['curl.exe','--fail','--location','--silent','--show-error','--max-time','90','-D',str(hp)]
    limit=10_000_000
    for k,v in (headers or {}).items():
        args+=['-H',f'{k}: {v}']
        if k=='Range':a,b=map(int,v.removeprefix('bytes=').split('-'));limit=b-a+1
    args+=['--max-filesize',str(limit),url]
    payload=subprocess.run(args,check=True,capture_output=True).stdout
    parts=[p for p in hp.read_text().split('\n\n') if p.strip()];last=parts[-1].split('\n',1)
    r=Response(payload);r.status=int(last[0].split()[1]);r.headers=email.message_from_string(last[1]);return r
class Remote(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,x,whence=0):self.pos=x if whence==0 else self.pos+x if whence==1 else self.size+x;return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        with request(self.url,{'Range':f'bytes={self.pos}-{self.pos+n-1}'}) as r:
            if r.status!=206:raise ValueError('Server does not honor ranges; refusing unintended full archive download')
            payload=r.read(n+1)
        assert len(payload)==n;self.pos+=n;return payload
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');args=ap.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    s=request(PAGE).read().decode();(WORK/'source_page.html').write_text(s,encoding='utf-8')
    paper=WORK/'SHARP.pdf'
    if not paper.exists():paper.write_bytes(request('https://arxiv.org/pdf/2103.09924').read())
    source=WORK/'source_code';source.mkdir(exist_ok=True)
    if not (source/'tree.json').exists():
        repo=json.loads(request('https://api.github.com/repos/francescamen/SHARP').read());branch=repo['default_branch']
        tree=json.loads(request(f'https://api.github.com/repos/francescamen/SHARP/git/trees/{branch}?recursive=1').read())
        (source/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        for entry in tree['tree']:
            name=entry['path']
            if entry['type']=='blob' and ('preprocessing.py' in name or name.lower() in ('readme.md','license')):
                (source/Path(name).name).write_bytes(request(f"https://raw.githubusercontent.com/francescamen/SHARP/{tree['sha']}/{name}").read())
    urls=[html.unescape(u) for u in re.findall(r'href="([^"]+)"',s) if u.endswith('/dataset.zip')]
    url=next(u for u in urls if not u.endswith('/indexcodes/dataset.zip'))
    if url.startswith('/'):url='https://researchdata.cab.unipd.it'+url
    with request(url,{'Range':'bytes=0-0'}) as r:
        print('RANGE',r.status,r.headers.get('Content-Range'),flush=True)
        if r.status!=206:raise ValueError('Archive ranges unavailable')
        size=int(r.headers['Content-Range'].split('/')[-1])
    z=zipfile.ZipFile(Remote(url,size));entries=[]
    for info in z.infolist():
        if info.is_dir():continue
        entries.append(dict(name=info.filename,bytes=info.file_size,compressed=info.compress_size,crc32=info.CRC))
        print(info.filename,info.file_size,flush=True)
    (WORK/'archive_index.json').write_text(json.dumps(dict(url=url,archive_bytes=size,members=entries),indent=2),encoding='utf-8')
    if args.download:
        selected=[]
        for info in z.infolist():
            if info.is_dir() or '__MACOSX' in info.filename:continue
            # Specific filenames selected only after inspecting the archive index.
            if info.filename.split('/')[0] not in {'S1a','S6a','S7a'}:continue
            if not re.search(r'_(?:E|L[12]?|W|R)\.mat$',Path(info.filename).name):continue
            selected.append(info)
        if not selected:raise ValueError('No exact selected names; inspect archive index before changing selection')
        if sum(i.file_size for i in selected)>2_000_000_000:raise ValueError('Selected extraction unexpectedly large')
        manifest=[]
        for info in selected:
            dest=WORK/'raw'/info.filename;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists() or dest.stat().st_size!=info.file_size:
                with z.open(info) as r:payload=r.read()
                dest.with_suffix('.part').write_bytes(payload)
                dest.with_suffix('.part').replace(dest)
            payload=dest.read_bytes();assert zlib.crc32(payload)&0xffffffff==info.CRC
            manifest.append(dict(path=str(dest),member=info.filename,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),crc32=info.CRC))
            print('DOWNLOADED',info.filename,flush=True)
        (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
