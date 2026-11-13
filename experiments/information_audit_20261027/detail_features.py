"""Extract lost temporal-scale and phase-coherence summaries from labeled raw CSI."""
import json
from pathlib import Path
import numpy as np
import blosc2
ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/detail_features_20261027')
def pool(x):
    # x: receiver/pair, frequency. Symmetric pooling over receiver identities.
    return np.concatenate([x.mean(0),x.std(0)])
def extract(a):
    power=np.abs(a)**2;valid=power>1e-8;amp=.5*np.log1p(power)
    amp=(amp*valid).reshape(600,3,32,32).sum(-1)/np.maximum(valid.reshape(600,3,32,32).sum(-1),1)
    d=amp-amp.mean(0);coarse=np.repeat(d.reshape(150,4,3,32).mean(1),4,axis=0);energy=(d*d).mean(0)
    temporal=np.concatenate([pool(((d-coarse)**2).mean(0)/(energy+1e-8)),pool(((d[1:]-d[:-1])**2).mean(0)/(energy+1e-8))])
    unit=a/np.sqrt(power+1e-4);phase=[]
    for u,v in ((0,1),(0,2),(1,2)):
        mask=valid[:,u]*valid[:,v];q=(unit[:,u]*unit[:,v].conj()*mask).reshape(600,32,32).sum(-1)/np.maximum(mask.reshape(600,32,32).sum(-1),1)
        # Magnitudes invariant to a constant pair-phase offset and pair reversal.
        phase.append(np.stack([np.abs(q.mean(0)),np.abs(q).mean(0),np.sqrt((np.abs(q-q.mean(0))**2).mean(0))]))
    phase=np.stack(phase);ph=np.concatenate([pool(phase[:,k]) for k in range(3)])
    return temporal,ph
def main():
    WORK.mkdir(parents=True,exist_ok=True);rows=json.loads(Path('D:/huawei2026-experiments/next053_20261025/rows.json').read_text());time=np.zeros((len(rows),128),np.float32);phase=np.zeros((len(rows),192),np.float32)
    for n,ident in enumerate(sorted({r['identifier'] for r in rows})):
        raw=blosc2.load_array(str(Path('D:/huawei2026-data/labeled')/f'CSI_{ident}.b2'))
        for j,r in enumerate(rows):
            if r['identifier']!=ident:continue
            a=np.asarray(raw[r['start']:r['stop']]).reshape(600,3,1024);time[j],phase[j]=extract(a)
        if n%8==7:print('EXTRACTED',n+1,'/48',flush=True)
    assert np.isfinite(time).all() and np.isfinite(phase).all()
    np.savez_compressed(WORK/'features.npz',temporal=time,phase=phase)
    (WORK/'rows.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8');print('DONE',time.shape,phase.shape,flush=True)
if __name__=='__main__':main()
