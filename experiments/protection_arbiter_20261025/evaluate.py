"""Evaluate all outer rooms only after cross-fitting and source selection finish."""
from learn import *

def bootstrap(records):
    rng=np.random.default_rng(20261025);values=[]
    for _ in range(5000):
        picked=[]
        for room in ROOMS:
            rr=[r for r in records if r['room']==room]
            picked.extend(rr[i] for i in rng.integers(0,len(rr),size=len(rr)))
        values.append(sum(r['delta'] for r in picked)/sum(r['windows'] for r in picked))
    return [float(x) for x in np.percentile(values,[2.5,97.5])]

def main():
    assert (WORK/'proxy_gate_complete.json').exists();out=[]
    for method in ['source_selected',*PROFILES]:
        result=dict(method=method,rooms={},records=[])
        for room in ROOMS:
            d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room)
            config=read(WORK/f'source_selection_{room}.json')
            config=config['selected'] if method=='source_selected' else next(c for c in config['candidates'] if c['profile']==method)
            cp=final_models(room,config['profile']);p=opinions(d['raw'],b,T)
            if AUX_HEADS:
                aux=torch.load(OOF_ROOT/f'target_{room}.pt',weights_only=True)
                p=torch.cat([p,p[:,1:2].expand(-1,6,-1)],1);p[aux['indices']]=aux['probabilities']
            z,actions=apply(p,cp,config)
            base=old.policy(d['raw'],b,T);ix=old.split(room,d['rows'])[2]
            if GUARD_ONLY and not AUX_HEADS:
                outside=~protection_mask(p)
                assert torch.equal(z[outside].argmax(-1),base[outside].argmax(-1))
            if AUX_HEADS:
                changed=(z.argmax(-1)!=base.argmax(-1))&~protection_mask(p)
                assert (z[changed].argmax(-1)<=1).all() and (base[changed].argmax(-1)<=1).all()
                assert (actions[changed]>=20).all()
            if KEEP_ONE:assert torch.equal(z[original_one(p)].argmax(-1),base[original_one(p)].argmax(-1))
            if PAIR_ONE:
                changed=z.argmax(-1)!=base.argmax(-1)
                assert (z[changed].argmax(-1)==1).all()
            metrics=old.metrics(z,d['labels'],base,ix);bm=old.metrics(base,d['labels'],base,ix)
            result['rooms'][room]=dict(metrics=metrics,baseline=bm,source_config=config)
            for identifier in sorted({d['rows'][i]['identifier'] for i in ix}):
                jj=[i for i in ix if d['rows'][i]['identifier']==identifier];y=d['labels'][jj]
                correct=int((z[jj].argmax(-1)==y).sum());bc=int((base[jj].argmax(-1)==y).sum())
                result['records'].append(dict(room=room,identifier=identifier,windows=len(jj),correct=correct,baseline_correct=bc,delta=correct-bc))
            torch.save(dict(logits=z,baseline=base,actions=actions,labels=d['labels'],indices=ix,config=config),WORK/f'eval_{method}_{room}.pt')
        result['net']=sum(x['delta'] for x in result['records']);result['total']=sum(x['windows'] for x in result['records']);result['ci95']=bootstrap(result['records'])
        result['room_gains']={room:(v['metrics']['correct']-v['baseline']['correct'])/v['metrics']['total'] for room,v in result['rooms'].items()}
        result['gain']=result['net']/result['total']
        result['passed']=result['gain']>=.02 and result['ci95'][0]>0 and min(result['room_gains'].values())>=-.02 and sum(g>0 for g in result['room_gains'].values())>=2
        out.append(result)
    save(WORK/'results.json',out)
    # Source-selected pipeline is the predeclared primary method. Other profiles exploratory.
    chosen=out[0] if out[0]['passed'] else next((x for x in sorted(out[1:],key=lambda x:x['net'],reverse=True) if x['passed']),None)
    save(WORK/'selection.json',dict(selected=chosen['method'] if chosen else None,
       reason='Source-only profiles/thresholds; outer development gate>=2pp, record CI>0, no room<-2pp. Fixed profiles exploratory if primary fails.',
       platform_score=None,reused_development=True))
    print(json.dumps([dict(method=r['method'],net=r['net'],room_gains=r['room_gains'],ci=r['ci95'],passed=r['passed']) for r in out],indent=2),flush=True)

if __name__=='__main__':main()
