"""Predefined four-view unanimous confirmation for protected low-count outputs.
No gate fitted across different seeds. Reused development screening is explicit.
"""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('spectral_base',Path(__file__).with_name('run.py'))
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
from arbiter_common import *
WORK=run.WORK/'consistency'

def policy(raw,extra,b,T,amount,scope):
    base=old.policy(raw,b,T);bp=base.softmax(-1);offset=raw.new_tensor([0,.35,0,0,0])
    opinions=(extra/T+b+offset).softmax(-1).flatten(1,2)
    alt=opinions.mean(1);winner=alt.argmax(-1);orig=bp.argmax(-1)
    votes=(opinions.argmax(-1)==winner[:,None]).all(1)
    confidence=alt.gather(1,winner[:,None]).squeeze(1)>=2*alt.gather(1,orig[:,None]).squeeze(1)
    v0=(raw[:,0,0]/T+b).argmax(-1);v45=(raw[:,0].exp().mean(1).clamp_min(1e-12).log()/T+b).argmax(-1)
    protected=(v0!=1)&((v45==0)|(v45==2))
    eligible=protected&votes&confidence&((winner-orig).abs()==1)
    if scope=='empty':eligible=eligible&(orig==0)
    proposal=((1-amount)*bp+amount*alt).clamp_min(1e-12).log()
    return torch.where(eligible[:,None],proposal,base)

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(kinds=run.KINDS,amounts=[.25,.5],scopes=['empty','protected'],rule='four transformed views unanimous; alternative odds at least2:1; adjacent count; original1 retained; same publication gate',training=False))
    spec=importlib.util.spec_from_file_location('consistency_bootstrap',ROOT/'experiments/protection_arbiter_20261025/evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    cache={room:torch.load(EVID/f'{room}.pt',weights_only=True) for room in (*ROOMS,'actual')};cal={room:old.calibration(room) for room in cache};results=[]
    for kind in run.KINDS:
      for scope in ['empty','protected']:
       for amount in [.25,.5]:
        r=dict(kind=kind,scope=scope,amount=amount,rooms={},records=[])
        for room,d in cache.items():
            x=torch.load(run.WORK/f'{room}_{kind}.pt',weights_only=True);ix=x['indices'];raw=d['raw'][ix];y=d['labels'][ix];b,T=cal[room]
            base=old.policy(raw,b,T);z=policy(raw,x['raw'],b,T,amount,scope);m=old.metrics(z,y,base,list(range(len(ix))));m['net']=m['corrected']-m['broken'];r['rooms'][room]=m
            torch.save(dict(logits=z,baseline=base,labels=y,indices=ix),WORK/f'eval_{kind}_{scope}_{amount}_{room}.pt')
            if room=='actual':continue
            for ident in sorted({d['rows'][i]['identifier'] for i in ix}):
                jj=[k for k,i in enumerate(ix) if d['rows'][i]['identifier']==ident];c=int((z[jj].argmax(-1)==y[jj]).sum());bc=int((base[jj].argmax(-1)==y[jj]).sum())
                r['records'].append(dict(room=room,identifier=ident,windows=len(jj),correct=c,baseline_correct=bc,delta=c-bc))
        r['net']=sum(x['delta'] for x in r['records']);r['ci95']=ev.bootstrap(r['records']);g=[r['rooms'][room]['net'] for room in ROOMS]
        r['passed']=r['net']>=9 and sum(x>0 for x in g)>=2 and min(g)>=-2 and r['rooms']['actual']['net']>=0 and r['ci95'][0]>0
        results.append(r);print(kind,scope,amount,r['net'],g,'source',r['rooms']['actual']['net'],'ci',r['ci95'],'pass',r['passed'],flush=True)
    save(WORK/'results.json',results);passed=sorted([r for r in results if r['passed']],key=lambda r:(r['net'],r['rooms']['actual']['net']),reverse=True)
    save(WORK/'selection.json',dict(selected=passed[0] if passed else None,platform_score=None,reused_development=True))

if __name__=='__main__':main()
