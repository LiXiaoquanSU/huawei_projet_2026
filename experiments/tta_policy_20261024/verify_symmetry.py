from pathlib import Path
"""Check the exported unprotected model under all receiver/time relabelings."""
import itertools
import numpy as np
import blosc2
from common import ROOT,WORK,read,save,sha
from deliver import session,probability


def main():
    meta=read(WORK/'candidate.json');assert meta['policy']=='unprotected'
    path=ROOT/meta['path'];assert sha(path)==meta['sha256'];runtime=session(path)
    inputs=[]
    for fixture in (ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz'):
        with np.load(fixture) as data:inputs.append((str(fixture),data['frames'].copy(),data['intervals'].copy()))
    inputs.append(('zero',np.zeros((600,1,3,1024,2),np.float32),np.zeros(600,np.float32)))
    raw=Path('D:/huawei2026-data/labeled');identifier='a_a1_hash813694_1'
    csi=blosc2.load_array(str(raw/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(raw/f'ATI_{identifier}.b2'))
    x=csi[:600].view(np.float32).reshape(600,1,3,1024,2).copy();dt=ati[:600].copy();dt[-1]=0
    inputs.append((identifier+':0',x,dt));del csi,ati
    reverse=np.r_[np.arange(598,-1,-1),599];cases=[]
    for name,frames,intervals in inputs:
        reference=probability(runtime.run(['logits'],dict(frames=frames,intervals=intervals))[0])
        for p in itertools.permutations(range(3)):
            for backwards in (False,True):
                x=frames[:,:,list(p),:,:].copy();dt=intervals.copy()
                if backwards:x=x[::-1].copy();dt=dt[reverse].copy()
                actual=probability(runtime.run(['logits'],dict(frames=x,intervals=dt))[0])
                error=float(np.abs(actual-reference).max());same=int(actual.argmax())==int(reference.argmax())
                assert error<.001 and same,(name,p,backwards,error)
                cases.append(dict(input=name,permutation=p,time_reversed=backwards,probability_max_error=error,argmax_matches=same))
        print('SYMMETRY',name,flush=True)
    result=dict(status='pass',cases=cases,note='4 inputs x all6 receiver orders x both time directions; invariance test only, not accuracy')
    save(WORK/'symmetry_audit.json',result);meta['symmetry_audit']=result
    save(WORK/'candidate.json',meta);save(path.with_suffix('.json'),meta)
    save(ROOT/'checks/symmetric_policy_20261024.json',dict(status='pass',sha256=meta['sha256'],cases=len(cases),max_probability_error=max(x['probability_max_error'] for x in cases)))
    print('SYMMETRY_PASS',len(cases),flush=True)


if __name__=='__main__':main()
