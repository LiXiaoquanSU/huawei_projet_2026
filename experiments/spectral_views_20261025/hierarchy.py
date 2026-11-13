"""Separate group mass from conditional count agreement across receiver views."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('spectral_fixed',Path(__file__).with_name('run.py'))
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
from arbiter_common import *
WORK=run.WORK/'hierarchy'
METHODS=('occupied_arithmetic','occupied_geometric','two_plus_arithmetic','two_plus_geometric')

def policy(raw,b,T,kind,amount):
    logq,q,p,_=old.components(raw,b,T)
    original=q.mean(1);original=original/original.sum(-1,keepdim=True)
    boundary=1 if kind.startswith('occupied') else 2
    conditional=p[:,:,boundary:]/p[:,:,boundary:].sum(-1,keepdim=True).clamp_min(1e-10)
    if kind.endswith('geometric'):conditional=conditional.clamp_min(1e-10).log().mean(1).softmax(-1)
    else:conditional=conditional.mean(1)
    mass=original[:,boundary:].sum(-1,keepdim=True)
    alternative=torch.cat([original[:,:boundary],mass*conditional],-1)
    fallback=((1-amount)*original+amount*alternative).clamp_min(1e-12).log()
    return old.finish(raw,b,T,fallback)

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(methods=METHODS,amounts=[.25,.5,1.],rule='same gate as spectral experiment; all comparisons reused development, not blind'))
    spec=importlib.util.spec_from_file_location('hierarchy_bootstrap',ROOT/'experiments/protection_arbiter_20261025/evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    cache={room:torch.load(EVID/f'{room}.pt',weights_only=True) for room in (*ROOMS,'actual')};cal={room:old.calibration(room) for room in cache}
    results=[]
    for kind in METHODS:
        for amount in [.25,.5,1.]:
            r=dict(kind=kind,amount=amount,rooms={},records=[])
            for room,d in cache.items():
                ix=old.split(room,d['rows'])[2];b,T=cal[room];raw=d['raw'][ix];y=d['labels'][ix]
                base=old.policy(raw,b,T);z=policy(raw,b,T,kind,amount);m=old.metrics(z,y,base,list(range(len(ix))));m['net']=m['corrected']-m['broken'];r['rooms'][room]=m
                torch.save(dict(logits=z,baseline=base,labels=y,indices=ix),WORK/f'eval_{kind}_{amount}_{room}.pt')
                if room=='actual':continue
                for ident in sorted({d['rows'][i]['identifier'] for i in ix}):
                    jj=[k for k,i in enumerate(ix) if d['rows'][i]['identifier']==ident];c=int((z[jj].argmax(-1)==y[jj]).sum());bc=int((base[jj].argmax(-1)==y[jj]).sum())
                    r['records'].append(dict(room=room,identifier=ident,windows=len(jj),correct=c,baseline_correct=bc,delta=c-bc))
            r['net']=sum(x['delta'] for x in r['records']);r['ci95']=ev.bootstrap(r['records']);g=[r['rooms'][room]['net'] for room in ROOMS]
            r['passed']=r['net']>=9 and sum(x>0 for x in g)>=2 and min(g)>=-2 and r['rooms']['actual']['net']>=0 and r['ci95'][0]>0
            results.append(r);print(kind,amount,r['net'],g,'source',r['rooms']['actual']['net'],'ci',r['ci95'],'pass',r['passed'],flush=True)
    save(WORK/'results.json',results);passed=sorted([r for r in results if r['passed']],key=lambda r:(r['net'],r['rooms']['actual']['net']),reverse=True)
    save(WORK/'selection.json',dict(selected=passed[0] if passed else None,platform_score=None,reused_development=True))

if __name__=='__main__':main()
