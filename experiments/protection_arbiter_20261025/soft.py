"""Bounded soft arbitration, selected by proper log loss on source OOF records."""
import learn as gates
from learn import *
assert GUARD_ONLY and not PAIR_ONLY and not PAIR_ONE
GATE_ROOT=WORK;WORK=OOF_ROOT/'guard_soft'
if '--max-half' in sys.argv:WORK=OOF_ROOT/'guard_soft_half'
if '--max-quarter' in sys.argv:WORK=OOF_ROOT/'guard_soft_quarter'
if ERROR_COST!=2:WORK=WORK/f'cost{ERROR_COST}'

def combine(prob,changed,amount):
    return ((1-amount)*prob[:,0]+amount*changed.exp()).clamp_min(1e-12).log()

def select(room):
    path=WORK/f'source_selection_{room}.json'
    if path.exists():return read(path)
    d=torch.load(OOF_ROOT/f'oof_{room}.pt',weights_only=True);p=d['probabilities'];y=d['labels'];w=weights(d['rows'],y)
    baseline=float((-p[:,0].clamp_min(1e-12).log().gather(1,y[:,None]).squeeze(1)*w).sum());candidates=[]
    for profile in PROFILES:
        cp=torch.load(GATE_ROOT/f'selection_{room}_{profile}.pt',weights_only=True);regret=cp['source_oof_regret']
        choices=[]
        for scope in ['group','views']:
            for threshold in [0.,.1,.2,.35,.5]:
                changed,index=action(p,regret,scope,threshold)
                for amount in ([.0625,.125,.25] if '--max-quarter' in sys.argv else [.125,.25,.5] if '--max-half' in sys.argv else [.25,.5,.75,1.]):
                    z=combine(p,changed,amount)
                    nll=float((-z.gather(1,y[:,None]).squeeze(1)*w).sum())
                    choices.append(dict(profile=profile,scope=scope,threshold=threshold,amount=amount,source_nll=nll,
                                        nll_gain=baseline-nll,changes=int((z.argmax(-1)!=p[:,0].argmax(-1)).sum())))
        choices.append(dict(profile=profile,scope='views',threshold=100.,amount=0.,source_nll=baseline,nll_gain=0.,changes=0))
        best=min(choices,key=lambda x:(round(x['source_nll'],7),x['amount'],-x['threshold']))
        candidates.append(best);save(WORK/f'choices_{room}_{profile}.json',choices)
    selected=min(candidates,key=lambda x:(round(x['source_nll'],7),x['amount']))
    result=dict(selected=selected,candidates=candidates,selection='Source record-OOF balanced proper logloss, not outer accuracy. Original protected scope only.')
    save(path,result);print('SOFT_SOURCE_SELECT',room,result,flush=True);return result

def infer(prob,config,room):
    cp=final_models(room,config['profile']);hard,index=gates.apply(prob,cp,config)
    return combine(prob,hard,config['amount']),index

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    if '--actual' in sys.argv:select('actual');return
    for room in ROOMS:select(room)
    # Import evaluation helpers uniquely; no accidental run.py/evaluate.py resolution.
    spec=importlib.util.spec_from_file_location('arbiter_soft_evaluation',ROOT/'experiments/protection_arbiter_20261025/evaluate.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    results=[]
    for method in ['source_selected',*PROFILES]:
        r=dict(method=method,rooms={},records=[])
        for room in ROOMS:
            d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room);p=opinions(d['raw'],b,T)
            if AUX_HEADS:
                a=torch.load(OOF_ROOT/f'target_{room}.pt',weights_only=True)
                p=torch.cat([p,p[:,1:2].expand(-1,6,-1)],1);p[a['indices']]=a['probabilities']
            source=select(room);config=source['selected'] if method=='source_selected' else next(x for x in source['candidates'] if x['profile']==method)
            z,index=infer(p,config,room);base=old.policy(d['raw'],b,T);ix=old.split(room,d['rows'])[2]
            assert torch.equal(z[~protection_mask(p)].argmax(-1),base[~protection_mask(p)].argmax(-1))
            r['rooms'][room]=dict(metrics=old.metrics(z,d['labels'],base,ix),baseline=old.metrics(base,d['labels'],base,ix),source_config=config)
            for identifier in sorted({d['rows'][i]['identifier'] for i in ix}):
                jj=[i for i in ix if d['rows'][i]['identifier']==identifier];y=d['labels'][jj]
                c=int((z[jj].argmax(-1)==y).sum());bc=int((base[jj].argmax(-1)==y).sum())
                r['records'].append(dict(room=room,identifier=identifier,windows=len(jj),correct=c,baseline_correct=bc,delta=c-bc))
            torch.save(dict(logits=z,baseline=base,actions=index,labels=d['labels'],indices=ix,config=config),WORK/f'eval_{method}_{room}.pt')
        r['net']=sum(x['delta'] for x in r['records']);r['total']=sum(x['windows'] for x in r['records']);r['ci95']=ev.bootstrap(r['records'])
        r['room_gains']={room:(v['metrics']['correct']-v['baseline']['correct'])/v['metrics']['total'] for room,v in r['rooms'].items()};r['gain']=r['net']/r['total']
        r['passed']=r['gain']>=.02 and r['ci95'][0]>0 and min(r['room_gains'].values())>=-.02 and sum(g>0 for g in r['room_gains'].values())>=2
        results.append(r)
    save(WORK/'results.json',results)
    chosen=results[0] if results[0]['passed'] else next((x for x in sorted(results[1:],key=lambda r:r['net'],reverse=True) if x['passed']),None)
    save(WORK/'selection.json',dict(selected=chosen['method'] if chosen else None,platform_score=None,reused_development=True,method='Soft protective arbiter; source balanced NLL selection. Iterative outer development, no blindtest claim.'))
    print(json.dumps([dict(method=r['method'],net=r['net'],rooms=r['room_gains'],ci=r['ci95'],passed=r['passed']) for r in results],indent=2),flush=True)

if __name__=='__main__':main()
