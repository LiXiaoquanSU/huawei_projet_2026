import hashlib
from pathlib import Path
import blosc2,numpy as np,torch
from tf_model import TFPreprocessor
from train_baseline import read_csv,window

torch.set_num_threads(4)
out=Path('artifacts/tf_cache');out.mkdir(parents=True,exist_ok=True)
model=TFPreprocessor().cuda().eval()
for split in ['train','val']:
    rows=read_csv(f'artifacts/baseline_a1_seed42/{split}_windows.csv')
    groups={}
    for j,r in enumerate(rows):groups.setdefault(r['identifier'],[]).append((j,r))
    features=[];order=[]
    with torch.no_grad():
        for n,(ident,group) in enumerate(groups.items()):
            csi=blosc2.load_array(group[0][1]['csi_path']);ati=blosc2.load_array(group[0][1]['ati_path'])
            for start in range(0,len(group),4):
                chunk=group[start:start+4];arrays=[window(csi,ati,r) for _,r in chunk]
                x=torch.from_numpy(np.stack([a[0] for a in arrays])).cuda();dt=torch.from_numpy(np.stack([a[1] for a in arrays])).cuda()
                features.append(model(x,dt).cpu());order.extend(j for j,_ in chunk)
            del csi,ati
            print(split,n+1,'/',len(groups),flush=True)
    t=torch.cat(features)[np.argsort(order)].contiguous()
    torch.save(dict(temporal=t,summary=torch.zeros(len(rows),17),labels=torch.tensor([int(r['headcount']) for r in rows]),
                    source_hash=hashlib.sha256(Path('tf_model.py').read_bytes()).hexdigest()),out/f'{split}_features.pt')
