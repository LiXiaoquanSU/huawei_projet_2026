from pathlib import Path
"""Compare exported ONNX and its Torch reference on raw real windows."""
import numpy as np
import blosc2
import torch
from common import ROOT,WORK,Submission,build,rows,read,save,sha
from deliver import session,probability


def main():
    torch.set_num_threads(4);meta=read(WORK/'candidate.json');path=ROOT/meta['path'];assert sha(path)==meta['sha256']
    model,bias=build(ROOT/'models/best_037/final.pt',full=True)
    wrapper=Submission(model,bias,meta['policy']).cuda().eval();runtime=session(path)
    raw=Path('D:/huawei2026-data/labeled');chosen=[];rooms=set();labels=set();cases=[]
    for row in rows(raw/'samples.csv'):
        if row['room'] not in rooms or row['headcount'] not in labels:
            chosen.append(row);rooms.add(row['room']);labels.add(row['headcount'])
    for row in chosen:
        x=blosc2.load_array(str(raw/f"CSI_{row['identifier']}.b2"));dt=blosc2.load_array(str(raw/f"ATI_{row['identifier']}.b2"))
        for start in (0,5700,11400):
            frames=x[start:start+600].view(np.float32).reshape(600,1,3,1024,2).copy();intervals=dt[start:start+600].copy();intervals[-1]=0
            with torch.no_grad():reference=probability(wrapper(torch.from_numpy(frames).cuda(),torch.from_numpy(intervals).cuda()).cpu().numpy())
            actual=probability(runtime.run(['logits'],dict(frames=frames,intervals=intervals))[0])
            error=float(np.abs(reference-actual).max());same=int(reference.argmax())==int(actual.argmax());margin=float(np.sort(reference)[-1]-np.sort(reference)[-2])
            assert error<.001 and (same or margin<.001)
            cases.append(dict(identifier=row['identifier'],start=start,probability_max_error=error,argmax_matches=same))
        print('EXPORT_NUMERICAL',row['identifier'],flush=True)
        del x,dt
    result=dict(status='pass',cases=cases,description='actual exported ONNX versus Torch candidate, raw-input numerical equivalence only, not accuracy')
    save(WORK/'export_raw_audit.json',result);meta['export_raw_audit']=result
    save(WORK/'candidate.json',meta);save(path.with_suffix('.json'),meta)
    print('PASSED',len(cases),'max_error',max(x['probability_max_error'] for x in cases),flush=True)


if __name__=='__main__':main()
