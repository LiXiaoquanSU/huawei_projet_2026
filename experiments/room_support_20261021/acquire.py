"""Acquire public source metadata and only verified 0--4 RadioPoints captures."""
import argparse, hashlib, json, re, urllib.request
from pathlib import Path
WORK=Path('D:/huawei2026-experiments/room_support_20261021')
REPO='RadioPoints/Device-free_RF_Human_Sensing_Datasets'
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RoomSense-research'}),timeout=90)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');args=ap.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    tree=json.load(fetch(f'https://api.github.com/repos/{REPO}/git/trees/master?recursive=1'))
    (WORK/'github_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
    entries=[x for x in tree['tree'] if x['type']=='blob' and ('WiFi-CrowdCounting/' in x['path'] or x['path'].lower() in ('readme.md','license','license.txt'))]
    for x in entries:print(x['path'],x.get('size'),flush=True)
    for x in entries:
        if 'WiFi-CrowdCounting/' in x['path']:continue
        (WORK/Path(x['path']).name).write_bytes(fetch(f"https://raw.githubusercontent.com/{REPO}/{tree['sha']}/{x['path']}").read())
    for name,url in {
        'dataport.html':'https://ieee-dataport.org/documents/csi-dataset-wireless-human-sensing-80-mhz-wi-fi-channels',
        'read_bfee.c':'https://raw.githubusercontent.com/dhalperi/linux-80211n-csitool-supplementary/master/matlab/read_bfee.c',
        'get_scaled_csi.m':'https://raw.githubusercontent.com/dhalperi/linux-80211n-csitool-supplementary/master/matlab/get_scaled_csi.m',
        'read_bf_file.m':'https://raw.githubusercontent.com/dhalperi/linux-80211n-csitool-supplementary/master/matlab/read_bf_file.m'
    }.items():
        try:
            with fetch(url) as r:
                payload=r.read();(WORK/name).write_bytes(payload)
                print('SOURCE',name,r.status,r.url,len(payload),flush=True)
        except Exception as e:print('SOURCE_ERROR',name,str(e),flush=True)
    if args.download:
        manifest=[]
        for x in entries:
            path=x['path']
            if 'WiFi-CrowdCounting/' not in path or Path(path).name.startswith('Setup_'):continue
            # Selection must follow the published names, inspected in metadata first.
            match=re.search(r'(?:^|/)([0-8])(?:people|person|persons|p|_people)?(?:[._/-]|$)',path,re.I)
            if not match:raise ValueError(f'Unrecognized count label: {path}')
            if int(match[1])>4:continue
            dest=WORK/'radiopoints'/path
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists() or dest.stat().st_size!=x['size']:
                temp=dest.with_suffix(dest.suffix+'.part')
                with fetch(f"https://raw.githubusercontent.com/{REPO}/{tree['sha']}/{path}") as r,temp.open('wb') as f:
                    while chunk:=r.read(1024*1024):f.write(chunk)
                assert temp.stat().st_size==x['size'];temp.replace(dest)
            digest=hashlib.sha256(dest.read_bytes()).hexdigest()
            manifest.append(dict(path=str(dest),source_path=path,room=path.split('/')[1],count=int(match[1]),bytes=x['size'],sha256=digest,git_blob=x['sha']))
            print('DOWNLOADED',path,x['size'],flush=True)
        (WORK/'radiopoints_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
