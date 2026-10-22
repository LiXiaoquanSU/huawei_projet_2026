"""Inspect a new public multi-environment counting dataset, selectively extract ZIP."""
import argparse,hashlib,io,json,re,urllib.request,zipfile,zlib,struct,subprocess
from pathlib import Path
WORK=Path('D:/huawei2026-experiments/data_revision_20261021')
def fetch(url,headers=None):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RoomSense-research',**(headers or {})}),timeout=90)
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
class Remote(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,x,whence=0):self.pos=x if whence==0 else self.pos+x if whence==1 else self.size+x;return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        with fetch(self.url,{'Range':f'bytes={self.pos}-{self.pos+n-1}'}) as r:
            if r.status!=206:raise ValueError('Server ignored byte ranges; not downloading entire archive')
            payload=r.read(n+1)
        assert len(payload)==n,(len(payload),n);self.pos+=n;return payload
def read_member(z,info,root):
    if info.compress_type in (0,8,12,14):return z.read(info)
    # EHUNAM uses Deflate64, unsupported by Python's zipfile. Fetch only this
    # member, reconstruct a one-member ZIP and let installed7-Zip decode it.
    r=z.fp;r.seek(info.header_offset);header=r.read(30);assert header[:4]==b'PK\x03\x04'
    n,e=struct.unpack_from('<HH',header,26);r.seek(n+e,1);compressed=r.read(info.compress_size)
    name=info.filename.encode('utf-8');crc=info.CRC;cs=info.compress_size;us=info.file_size;method=info.compress_type
    local=struct.pack('<IHHHHHIIIHH',0x04034b50,21,2048,method,0,33,crc,cs,us,len(name),0)+name+compressed
    central=struct.pack('<IHHHHHHIIIHHHHHII',0x02014b50,20,21,2048,method,0,33,crc,cs,us,len(name),0,0,0,0,0,0)+name
    end=struct.pack('<IHHHHIIH',0x06054b50,0,0,1,1,len(central),len(local),0)
    temp=root/'selected_member.zip';temp.write_bytes(local+central+end)
    output=subprocess.run(['C:/Program Files/7-Zip/7z.exe','e','-so',str(temp)],check=True,capture_output=True).stdout
    assert len(output)==us and zlib.crc32(output)&0xffffffff==crc;temp.unlink();return output
def main(args):
    root=WORK/'ehunam';root.mkdir(parents=True,exist_ok=True)
    article=json.load(fetch('https://api.figshare.com/v2/articles/28541225'));save(root/'figshare_article.json',article)
    print('TITLE',article['title'],'LICENSE',article.get('license'),flush=True)
    for f in article['files']:print('FILE',f,flush=True)
    for f in article['files']:
        if f['name'] in ('Summary.mat','Summary.xlsx'):
            payload=fetch(f['download_url']).read();assert hashlib.md5(payload).hexdigest()==f['computed_md5'];(root/f['name']).write_bytes(payload)
    for name in ('README.md','Subcarrier.m'):
        (root/name).write_bytes(fetch(f'https://raw.githubusercontent.com/GuillermoDiazSM/EHUNAM-dataset/main/{name}').read())
    for f in article['files']:
        if not f['name'].lower().endswith('.zip'):continue
        z=zipfile.ZipFile(Remote(f['download_url'],f['size']));index=[dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size,crc32=i.CRC) for i in z.infolist() if not i.is_dir()]
        save(root/'archive_index.json',dict(source=f,members=index));print('ZIP_INDEX',len(index),'BYTES',sum(i['bytes'] for i in index),flush=True)
        names=[i for i in z.infolist() if not i.is_dir() and (i.filename.lower().endswith('summary.mat') or any(s in i.filename.lower() for s in ('readme','license','summary.csv')))]
        if args.sample:
            candidates=[i for i in z.infolist() if 'MC1_' in i.filename and '_PC_' in i.filename and i.filename.endswith('.mat')]
            print('CANDIDATE_EXAMPLE',[(i.filename,i.file_size,i.compress_type) for i in candidates[:25]],flush=True)
            names+=candidates[:1]
        if args.selected:
            wanted=set(json.loads(Path(args.selected).read_text())['members']);names+=[i for i in z.infolist() if i.filename in wanted]
        manifest=[]
        for info in dict((i.filename,i) for i in names).values():
            dest=root/'raw'/info.filename
            assert dest.resolve().is_relative_to((root/'raw').resolve())
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists() or dest.stat().st_size!=info.file_size:
                payload=read_member(z,info,root);assert zlib.crc32(payload)&0xffffffff==info.CRC
                temp=dest.with_suffix(dest.suffix+'.part');temp.write_bytes(payload);temp.replace(dest)
            with dest.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
            manifest.append(dict(member=info.filename,path=str(dest),bytes=info.file_size,crc32=info.CRC,sha256=digest));print('DOWNLOADED',info.filename,info.file_size,flush=True)
        previous=root/'download_manifest.json'
        allitems={i['member']:i for i in json.loads(previous.read_text())} if previous.exists() else {}
        allitems.update({i['member']:i for i in manifest});save(previous,list(allitems.values()))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sample',action='store_true');ap.add_argument('--selected');main(ap.parse_args())
