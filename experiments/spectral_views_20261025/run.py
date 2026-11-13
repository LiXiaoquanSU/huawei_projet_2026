"""Fixed spectral-view experiment; frozen original054 protections and weights.

These feature-space transformations are hypotheses, not guaranteed CSI symmetries.
Selection uses reused development rooms, not a fresh blind test.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/protection_arbiter_20261025'))
from arbiter_common import *
WORK=Path('D:/huawei2026-experiments/spectral_views_20261025')
KINDS=('frequency_reverse','frequency_smooth')

def transform(t,kind):
    a=t[:,:768].reshape(-1,6,128,150)
    p=t[:,768:960].reshape(-1,6,32,150)
    if kind=='frequency_reverse':a=a.flip(2);p=p.flip(2)
    elif kind=='frequency_smooth':
        # Convex 1/4,1/2,1/4 smoothing; replicated endpoints, no frequency wrap.
        def smooth(x):return .5*x+.25*torch.cat([x[:,:,:1],x[:,:,:-1]],2)+.25*torch.cat([x[:,:,1:],x[:,:,-1:]],2)
        a=smooth(a);p=smooth(p)
    else:raise ValueError(kind)
    return torch.cat([a.flatten(1,2),p.flatten(1,2),t[:,960:]],1)

def policy(raw,extra,b,T,amount):
    original=raw.exp().mean(2).clamp_min(1e-12).log()/T+b
    new=extra.exp().mean(2).clamp_min(1e-12).log()/T+b
    # Extra views: identity and reverse Rx, each with both time directions.
    q=(1-amount)*original.exp().mean(1)+amount*new.exp().mean(1)
    return old.finish(raw,b,T,q.clamp_min(1e-12).log())

def infer():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(kinds=KINDS,amounts=[.25,.5],selection='development net >= 9/880; >=2 rooms improve; no room net below -2 windows; source >=676/960; record bootstrap lower >0',frozen_guards=True,prior_free_arbiter_used=False))
    t=np.load(PREV/'t.npy',mmap_mode='r');s=np.load(PREV/'s.npy',mmap_mode='r')
    for room in (*ROOMS,'actual'):
        d=torch.load(EVID/f'{room}.pt',weights_only=True);ix=old.split(room,d['rows'])[2]
        if all((WORK/f'{room}_{kind}.pt').exists() for kind in KINDS):continue
        model,_=old.build(old.checkpoint(room),full=room=='actual');model=model.cuda().eval()
        idx,sign,si=old.permutation_indices((2,1,0));idx=idx.cuda();sign=sign.cuda();si=si.cuda()
        for kind in KINDS:
            path=WORK/f'{room}_{kind}.pt'
            if path.exists():continue
            out=[]
            with torch.inference_mode():
                for start in range(0,len(ix),12):
                    jj=ix[start:start+12];n=len(jj)
                    tt=torch.from_numpy(np.array(t[jj])).cuda().flatten(0,1);ss=torch.from_numpy(np.array(s[jj])).cuda().flatten(0,1)
                    tt=transform(tt,kind)
                    tt=torch.cat([tt,tt.index_select(1,idx)*sign]);ss=torch.cat([ss,ss.index_select(1,si)])
                    z=model.forward_features(tt,ss).reshape(2,n,2,5).permute(1,0,2,3)
                    out.append(z.cpu())
                    if start%120==0:print('INFER',room,kind,start,len(ix),flush=True)
            torch.save(dict(indices=ix,raw=torch.cat(out)),path)
        del model;torch.cuda.empty_cache()

def evaluate():
    spec=importlib.util.spec_from_file_location('spectral_bootstrap',ROOT/'experiments/protection_arbiter_20261025/evaluate.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    results=[]
    for kind in KINDS:
        for amount in [.25,.5]:
            result=dict(kind=kind,amount=amount,rooms={},records=[])
            for room in (*ROOMS,'actual'):
                d=torch.load(EVID/f'{room}.pt',weights_only=True);x=torch.load(WORK/f'{room}_{kind}.pt',weights_only=True);ix=x['indices'];b,T=old.calibration(room)
                raw=d['raw'][ix];y=d['labels'][ix];base=old.policy(raw,b,T);z=policy(raw,x['raw'],b,T,amount)
                m=old.metrics(z,y,base,list(range(len(ix))));bm=old.metrics(base,y,base,list(range(len(ix))))
                m['baseline_correct']=bm['correct'];m['net']=m['correct']-bm['correct'];result['rooms'][room]=m
                torch.save(dict(logits=z,baseline=base,labels=y,indices=ix),WORK/f'eval_{kind}_{amount}_{room}.pt')
                if room=='actual':continue
                for ident in sorted({d['rows'][i]['identifier'] for i in ix}):
                    jj=[k for k,i in enumerate(ix) if d['rows'][i]['identifier']==ident]
                    c=int((z[jj].argmax(-1)==y[jj]).sum());bc=int((base[jj].argmax(-1)==y[jj]).sum())
                    result['records'].append(dict(room=room,identifier=ident,windows=len(jj),correct=c,baseline_correct=bc,delta=c-bc))
            result['net']=sum(r['delta'] for r in result['records']);result['ci95']=ev.bootstrap(result['records'])
            gains=[result['rooms'][room]['net'] for room in ROOMS]
            result['passed']=result['net']>=9 and sum(g>0 for g in gains)>=2 and min(gains)>=-2 and result['rooms']['actual']['net']>=0 and result['ci95'][0]>0
            results.append(result)
            print('RESULT',kind,amount,'net',result['net'],'rooms',gains,'source',result['rooms']['actual']['net'],'CI',result['ci95'],'pass',result['passed'],flush=True)
    save(WORK/'results.json',results)
    passed=sorted([r for r in results if r['passed']],key=lambda r:(r['net'],r['rooms']['actual']['net']),reverse=True)
    save(WORK/'selection.json',dict(selected=passed[0] if passed else None,platform_score=None,reused_development=True))

if __name__=='__main__':
    infer();evaluate()
