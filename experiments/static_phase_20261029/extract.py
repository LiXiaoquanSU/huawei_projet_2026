"""Static frequency curvature of inter-RX phase, invariant to phase affine in frequency."""
import sys
from pathlib import Path
import numpy as np
import blosc2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/static_phase_20261029');CACHE=Path('D:/huawei2026-experiments/next053_20261025');RAW=Path('D:/huawei2026-data/labeled')

def descriptor(c):
    c=np.asarray(c[:,0],dtype=np.complex64);power=np.abs(c);valid=power>1e-4;u=c/np.maximum(power,1e-4)
    values=[]
    for step in (8,16):
        pair_stats=[]
        for a,b in ((0,1),(0,2),(1,2)):
            r=u[:,a]*u[:,b].conj();mask=valid[:,a]&valid[:,b]
            cur=r[:,2*step:]*r[:,:-2*step]*np.square(r[:,step:-step].conj())
            good=mask[:,2*step:]&mask[:,:-2*step]&mask[:,step:-step]
            z=np.zeros_like(r);w=np.zeros_like(mask);z[:,step:-step]=cur*good;w[:,step:-step]=good
            # Aggregate time and32 adjacent subcarriers without aligning away static phase.
            denom=w.reshape(len(c),32,32).sum((0,2)).clip(1)
            mean=z.reshape(len(c),32,32).sum((0,2))/denom
            pair_stats.append(np.stack([mean.real,np.abs(mean.imag),np.abs(mean)],0))
        v=np.stack(pair_stats);values.extend([v.mean(0).ravel(),v.std(0).ravel()])
    return np.concatenate(values).astype(np.float32)

def checks():
    rng=np.random.default_rng(29);c=(rng.normal(size=(48,1,3,1024))+1j*rng.normal(size=(48,1,3,1024))).astype(np.complex64);base=descriptor(c)
    f=np.arange(1024,dtype=np.float32);alpha=np.array([.3,-1.2,2.1],np.float32);beta=np.array([.002,-.003,.005],np.float32)
    transformed=c*np.exp(1j*(alpha[:,None]+beta[:,None]*f))[None,None]
    errors=dict(affine_phase=float(np.max(np.abs(base-descriptor(transformed)))),receiver_permutation=float(np.max(np.abs(base-descriptor(c[:,:,[2,0,1]])))) ,time_reversal=float(np.max(np.abs(base-descriptor(c[::-1])))))
    assert max(errors.values())<1e-5;assert np.isfinite(descriptor(np.zeros_like(c))).all();return errors

def main():
    WORK.mkdir(exist_ok=True);test=checks();rows=old.read(CACHE/'rows.json');features=np.zeros((len(rows),384),np.float32)
    for ident in sorted({r['identifier'] for r in rows}):
        p=RAW/f'CSI_{ident}.b2';a=blosc2.load_array(str(p))
        for i,r in enumerate(rows):
            if r['identifier']==ident:features[i]=descriptor(a[r['start']:r['stop']])
        del a;print('EXTRACTED',ident,flush=True)
    np.savez_compressed(WORK/'features.npz',static_phase=features)
    old.save(WORK/'protocol.json',dict(rows_sha256=old.sha(CACHE/'rows.json'),features_shape=list(features.shape),checks=test,raw_directory=str(RAW),formula='Inter-RX unit phasor r(f); r(f+d)r(f-d)conj(r(f))^2, d8/16. Mask invalid subcarriers, temporal/frequency mean, real/absoluteimag/magnitude, symmetricpairmean/std.',limits='Invariant to ideal affine phase in frequency, not all hardware distortion. Static room geometry may dominate; not known human-specific. Never interpreted as recovered persons.',code_sha256=old.sha(Path(__file__))))
if __name__=='__main__':main()
