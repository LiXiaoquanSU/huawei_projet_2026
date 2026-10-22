import csv,hashlib,importlib.util,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
WORK=Path('D:/huawei2026-experiments/sliding_windows_20261022')
OLD=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
RAW=Path('D:/huawei2026-data/labeled')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def old_data():
    d=torch.load(OLD,map_location='cpu',weights_only=True,mmap=True);d.pop('absolute',None)
    for k,h in d['source_hashes'].items():assert sha(ROOT/'src'/k)==h
    return d
def build(name):
    folder='fresh_backbones_20261022' if name=='mixer' else 'global_local_20261021'
    spec=importlib.util.spec_from_file_location('sliding_'+folder,ROOT/'experiments'/folder/'network.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.build(name)
class FeatureStore:
    def __init__(self):
        self.index=read(WORK/'cache/index.json');self.rows=self.index['rows'];self.arrays={}
        for k,h in self.index['source_hashes'].items():assert sha(ROOT/'src'/k)==h
    def get(self,i):
        r=self.rows[i];ident=r['identifier']
        if ident not in self.arrays:
            folder=WORK/'cache/shards'/ident
            self.arrays[ident]=(np.load(folder/'t.npy',mmap_mode='r'),np.load(folder/'s.npy',mmap_mode='r'))
        t,s=self.arrays[ident];j=r['local_index']
        return torch.from_numpy(t[j].copy()),torch.from_numpy(s[j].copy())
    def indices(self,recipe,fold,training):
        return [i for i,r in enumerate(self.rows) if r[recipe] and (fold=='full' or (r['room'] not in ('unknown',fold) if training else r['room']==fold))]
class WindowDataset(Dataset):
    def __init__(self,store,indices):self.store=store;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,j):
        i=self.indices[j];t,s=self.store.get(i);return t,s,int(self.store.rows[i]['headcount']),i
