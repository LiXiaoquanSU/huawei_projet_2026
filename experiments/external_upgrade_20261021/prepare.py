"""Prepare real-channel external sequences; no antenna/subcarrier replication."""
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/external_upgrade_20261021')
DATA=Path('D:/huawei2026-experiments/room_support_20261021')
def save(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def temporal(c,t):
    a=.5*np.log1p(np.abs(c).astype(np.float64)**2)
    if a.shape[-1]>32:
        # Group actual neighboring observed bins, rather than filling new bins.
        groups=np.array_split(np.arange(a.shape[-1]),32)
        a=np.stack([a[...,g].mean(-1) for g in groups],-1)
    a=a.reshape(len(a),-1);hz=1/np.median(np.diff(t));grid=np.arange(0,3,1/hz)
    v=np.stack([np.interp(grid,t,a[:,k]) for k in range(a.shape[1])],1)
    ratio=Fraction(50/hz).limit_denominator(1000);v=resample_poly(v,ratio.numerator,ratio.denominator,axis=0)
    v=np.stack([np.interp(np.arange(150)/50,np.arange(len(v))/50,v[:,k]) for k in range(v.shape[1])])
    v=(v-v.mean())/np.sqrt(np.mean((v-v.mean())**2)+1e-4)
    return np.clip(v,-6,6).astype(np.float32)
def main():
    WORK.mkdir(parents=True,exist_ok=True);dest=WORK/'external_sequences.pt'
    if dest.exists():print('EXISTING',dest,flush=True);return
    result={}
    rr=rows(DATA/'radiopoints_windows.csv');images=[]
    for record in dict.fromkeys(r['record'] for r in rr):
        d=np.load(DATA/'clean'/f'{record}.npz');c,t=d['csi'],d['time_s']
        for r in [r for r in rr if r['record']==record]:
            start=float(r['start_s']);take=(t>=start)&(t<start+3)
            images.append(temporal(c[take],t[take]-start))
        print('RADIO_SEQUENCE',record,flush=True)
    result['radio']=dict(x=torch.from_numpy(np.stack(images)),y=torch.tensor([int(r['headcount']) for r in rr]),rows=rr)
    sr=rows(DATA/'sharp_public/windows.csv');images=[]
    for record in dict.fromkeys(r['record'] for r in sr):
        path=DATA/'sharp_public/raw'/record;a=loadmat(path)['csi_buff'];a=np.fft.fftshift(a,axes=1)
        index=np.load(DATA/'sharp_public/clean'/f'{path.stem}_index.npz')
        a=a[:len(a)//4*4].reshape(-1,1,4,256)[index['packet_index']][...,index['valid_subcarriers']];t=index['time_s']
        for r in [r for r in sr if r['record']==record]:
            start=float(r['start_s']);take=(t>=start)&(t<start+3)
            images.append(temporal(a[take],t[take]-start))
        print('SHARP_SEQUENCE',record,flush=True)
    result['sharp']=dict(x=torch.from_numpy(np.stack(images)),y=torch.tensor([int(r['headcount']) for r in sr]),activity=torch.tensor([['empty','sitting','walking','running'].index(r['activity']) for r in sr]),rows=sr)
    b=torch.load('D:/huawei2026-experiments/motion_external_20261021/external/single_rx_features_v2.pt',weights_only=True,map_location='cpu')
    ix=[i for i,r in enumerate(b['rows']) if r['split']=='external_train']
    result['brescia']=dict(x=b['x'][ix].reshape(-1,32,4,150).mean(2),y=b['y'][ix],rows=[dict(b['rows'][i],room='brescia_office') for i in ix])
    for source,d in result.items():
        assert torch.isfinite(d['x']).all() and len(d['x'])==len(d['rows'])==len(d['y'])
        print('PREPARED',source,list(d['x'].shape),torch.bincount(d['y'],minlength=5).tolist(),flush=True)
    torch.save(result,dest)
    save(WORK/'external_preparation.json',{k:dict(shape=list(v['x'].shape),records=len({r['record'] for r in v['rows']}),class_counts=torch.bincount(v['y'],minlength=5).tolist(),nominal_time_windows=sum('nominal' in r.get('timing','') for r in v['rows'])) for k,v in result.items()})
if __name__=='__main__':main()
