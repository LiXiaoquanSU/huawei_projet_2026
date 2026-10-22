"""Descriptive diagnostics only: room/count support, no fitting or selection."""
import json
import numpy as np
import torch
from train import load
from network import ROOT,SpectralDiversity
torch.set_num_threads(4);d=load();collected=[]
with torch.no_grad():
    for start in range(0,len(d['t']),32):
        t=d['t'][start:start+32]
        a=t[:,384:768].reshape(-1,3,8,16,150).mean(3).reshape(-1,24,150)
        p=t[:,768:960].reshape(-1,6,8,4,150).mean(3).reshape(-1,48,150)
        collected.append(torch.stack([SpectralDiversity.moments(x)[:,-1] for x in (a,p)],1))
features=torch.cat(collected).numpy();rows=[]
for room in sorted({r['room'] for r in d['rows']}):
    for count in range(5):
        ix=[i for i,r in enumerate(d['rows']) if r['room']==room and int(d['y'][i])==count]
        if not ix:continue
        x=features[ix]
        rows.append(dict(room=room,count=count,records=len({d['rows'][i]['identifier'] for i in ix}),windows=len(ix),amplitude_participation_mean=float(x[:,0].mean()),phase_participation_mean=float(x[:,1].mean())))
out=dict(note='Within-window normalized participation proxies; not people counts. Group means only; correlated windows, sparse room/count coverage.',groups=rows)
(ROOT/'checks/spectral_feature_diagnosis_20261022.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
