"""Factorized/ordinal agreement pooling of frozen direct and CORN heads."""
import json,sys,importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('adapter049_shared',ROOT/'experiments/adapter049_20261024/run.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
policy,read,save=adapter.policy,adapter.read,adapter.save
WORK=Path('D:/huawei2026-experiments/ordinal_consensus_20261024')
EVIDENCE=Path('D:/huawei2026-experiments/presence_rescue_20261024')
OLD=Path('D:/huawei2026-experiments/protected_context_20261024')
METHODS=('geometric','conditional_arithmetic','conditional_geometric','ordinal_logodds')

def fuse(d,o,method,strength):
    p=(d+o)*.5
    if method=='geometric':
        a=(d.clamp_min(1e-8)*o.clamp_min(1e-8)).sqrt();a=a/a.sum(-1,keepdim=True)
    elif method.startswith('conditional'):
        dd=d[...,1:]/d[...,1:].sum(-1,keepdim=True).clamp_min(1e-8)
        oo=o[...,1:]/o[...,1:].sum(-1,keepdim=True).clamp_min(1e-8)
        pos=(dd+oo)*.5 if method=='conditional_arithmetic' else (dd.clamp_min(1e-8)*oo.clamp_min(1e-8)).sqrt()
        pos=pos/pos.sum(-1,keepdim=True).clamp_min(1e-8)
        a=torch.cat([p[...,:1],(1-p[...,:1])*pos],-1)
    elif method=='ordinal_logodds':
        dc=d.flip(-1).cumsum(-1).flip(-1)[...,1:].clamp(1e-7,1-1e-7)
        oc=o.flip(-1).cumsum(-1).flip(-1)[...,1:].clamp(1e-7,1-1e-7)
        q=((dc.log()-torch.log1p(-dc)+oc.log()-torch.log1p(-oc))*.5).sigmoid()
        a=torch.cat([1-q[...,:1],q[...,:-1]-q[...,1:],q[...,-1:]],-1).clamp_min(0)
        a=a/a.sum(-1,keepdim=True)
    else:raise ValueError(method)
    return ((1-strength)*p+strength*a).clamp_min(1e-8).log()

def metric(z,y,base,indices):
    ix=torch.tensor(indices);pred=z[ix].argmax(-1);bp=base[ix].argmax(-1);yy=y[ix]
    cm=torch.bincount(yy*5+pred,minlength=25).reshape(5,5)
    return dict(correct=int((pred==yy).sum()),total=len(ix),corrected=int(((pred==yy)&(bp!=yy)).sum()),broken=int(((pred!=yy)&(bp==yy)).sum()),confusion_matrix=cm.tolist())

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(methods=list(METHODS),strengths=[.25,.5,1.],training=False,
      rationale='Current equal probability mixture conflates occupancy and conditional count confidence. Compare factorized conditional count pooling and ordered cumulative log-odds pooling; existing CORN is not new.',
      control='Actual scored049 same original four views, rules and fixed bias. No extra backbone, no source label fitted parameters.',
      selection='Positive independent-room aggregate; no room losing>2 percentage points; all source classes nondecreasing. Multiple fixed recipes are development comparisons, not independent test selection.',
      caveat='Existing biases calibrated for original arithmetic mixture, not guaranteed calibrated for new pooling. No hidden uniform-prior assumption.'))
    cache=torch.load(Path('D:/huawei2026-experiments/evidence049_research_20261024/frozen_latents.pt'),weights_only=True);rows=cache['rows'];y=cache['labels'];reports=[]
    for room in ('actual','a1','c1','g1','e1'):
        actual=room=='actual';e=torch.load(EVIDENCE/('actual_evidence.pt' if actual else f'proxy_evidence_{room}.pt'),weights_only=True)
        f=e['features'][:,:172].reshape(-1,4,43);d=f[:,:,:5];o=f[:,:,5:10]
        assert float((d.sum(-1)-1).abs().max())<1e-5 and float((o.sum(-1)-1).abs().max())<1e-5
        assert float((((d+o)*.5)-e['logits'].exp()).abs().max())<1e-6
        if actual:b=e['bias'];T=1.;ix=list(range(960))
        else:
            cal=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline'];b=torch.tensor(cal['bias']);T=cal['temperature']
            manifest=read(OLD/'results'/f's42_{room}'/'manifest.json');ids={r['identifier'] for r in manifest['test']};ix=[i for i,r in enumerate(rows) if r['identifier'] in ids]
        base=policy(e['logits'],b,T)
        for method in METHODS:
            for strength in (.25,.5,1.):
                raw=fuse(d,o,method,strength);z=policy(raw,b,T)
                r=dict(room=room,method=method,strength=strength,metrics=metric(z,y,base,ix));reports.append(r)
                np.savez_compressed(WORK/f'{room}_{method}_{strength}.npz',raw=raw.numpy(),logits=z.numpy(),labels=y.numpy(),test_indices=ix)
                print('POOL',room,method,strength,r['metrics']['corrected'],r['metrics']['broken'],flush=True)
    save(WORK/'results.json',reports);summary=[]
    for method in METHODS:
        for strength in (.25,.5,1.):
            rr=[r for r in reports if r['room']!='actual' and r['method']==method and r['strength']==strength]
            actual=next(r for r in reports if r['room']=='actual' and r['method']==method and r['strength']==strength)
            worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in rr)
            sourceok=all(actual['metrics']['confusion_matrix'][k][k]>=old for k,old in enumerate([206,198,66,78,140]))
            net=sum(r['metrics']['corrected']-r['metrics']['broken'] for r in rr)
            summary.append(dict(method=method,strength=strength,proxy_net=net,worst_room_delta=worst,
              room_net={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in rr},source=actual['metrics'],eligible=net>0 and worst>=-.02 and sourceok))
    save(WORK/'summary.json',summary);save(WORK/'rows.json',rows)
    print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
