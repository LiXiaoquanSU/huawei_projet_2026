"""Compare the standalone architecture to frozen054 on identical whole-room windows."""
from train import *

def main():
    spec=importlib.util.spec_from_file_location('shared_hierarchy_bootstrap',ROOT/'experiments/protection_arbiter_20261025/evaluate.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    reports=[]
    for variant in VARIANTS:
        report=dict(variant=variant,rooms={},records=[],epochs=[],seeds={str(seed):dict(correct=0,total=0) for seed in SEEDS})
        for room in ROOMS:
            preds=[]
            for seed in SEEDS:
                folder=WORK/f'{room}_{variant}_s{seed}';r=read(folder/'result.json');d=torch.load(folder/'predictions.pt',weights_only=True);preds.append(d['logits'].exp());report['epochs'].append(r['selected_epoch'])
                for k in ('correct','total'):report['seeds'][str(seed)][k]+=r['metrics'][k]
            z=torch.stack(preds).mean(0).clamp_min(1e-12).log();b,T=old.calibration(room);ref=torch.load(EVID/f'{room}.pt',weights_only=True);base=old.policy(ref['raw'],b,T);ix=d['test_indices']
            m=old.metrics(z,d['labels'],base,ix);m['baseline_correct']=int((base[ix].argmax(-1)==d['labels'][ix]).sum())
            yy=d['labels'][ix];present=yy>0
            # Presence decision uses P(present)>.5, not multiclass argmax!=0.
            m['presence_correct']=int(((z[ix,0].exp()<.5)==present).sum())
            m['positive_count_correct']=int(((z[ix,1:].argmax(-1)+1==yy)&present).sum());m['positive_total']=int(present.sum())
            report['rooms'][room]=m
            torch.save(dict(logits=z,baseline=base,labels=d['labels'],test_indices=ix),WORK/f'ensemble_{variant}_{room}.pt')
            for ident in sorted({d['rows'][i]['identifier'] for i in ix}):
                jj=[i for i in ix if d['rows'][i]['identifier']==ident];yy=d['labels'][jj];c=int((z[jj].argmax(-1)==yy).sum());bc=int((base[jj].argmax(-1)==yy).sum())
                report['records'].append(dict(room=room,identifier=ident,windows=len(jj),correct=c,baseline_correct=bc,delta=c-bc))
        report['correct']=sum(x['correct'] for x in report['records']);report['total']=sum(x['windows'] for x in report['records']);report['baseline_correct']=sum(x['baseline_correct'] for x in report['records']);report['net']=report['correct']-report['baseline_correct'];report['ci95']=ev.bootstrap(report['records'])
        report['mean_single_seed_correct']=float(np.mean([x['correct'] for x in report['seeds'].values()]));reports.append(report)
        print('ARCHITECTURE_RESULT',variant,report['correct'],report['total'],'baseline',report['baseline_correct'],'seeds',report['seeds'],'CI',report['ci95'],flush=True)
    chosen=max(reports,key=lambda r:r['mean_single_seed_correct']);epochs=int(np.median(chosen['epochs']))
    # Follow-up after ensemble failure: select a single seed using SOURCE NLL,
    # not its target accuracy. Keep the originally planned ensemble results.
    source_nll={seed:float(np.mean([read(WORK/f'{room}_{chosen["variant"]}_s{seed}'/'result.json')['source_calibration_nll'] for room in ROOMS])) for seed in SEEDS}
    seed=min(source_nll,key=source_nll.get);single=dict(variant=chosen['variant'],seed=seed,rooms={},records=[])
    for room in ROOMS:
        d=torch.load(WORK/f'{room}_{chosen["variant"]}_s{seed}'/'predictions.pt',weights_only=True);z=d['logits'];ix=d['test_indices'];b,T=old.calibration(room)
        ref=torch.load(EVID/f'{room}.pt',weights_only=True);base=old.policy(ref['raw'],b,T);y=d['labels'];m=old.metrics(z,y,base,ix);m['baseline_correct']=int((base[ix].argmax(-1)==y[ix]).sum())
        yy=y[ix];positive=yy>0;m['presence_correct']=int(((z[ix,0].exp()<.5)==positive).sum());m['positive_count_correct']=int(((z[ix,1:].argmax(-1)+1==yy)&positive).sum());m['positive_total']=int(positive.sum());single['rooms'][room]=m
        for ident in sorted({d['rows'][i]['identifier'] for i in ix}):
            jj=[i for i in ix if d['rows'][i]['identifier']==ident];c=int((z[jj].argmax(-1)==y[jj]).sum());bc=int((base[jj].argmax(-1)==y[jj]).sum());single['records'].append(dict(room=room,identifier=ident,windows=len(jj),correct=c,baseline_correct=bc,delta=c-bc))
    single['correct']=sum(x['correct'] for x in single['records']);single['total']=880;single['baseline_correct']=sum(x['baseline_correct'] for x in single['records']);single['net']=single['correct']-single['baseline_correct'];single['ci95']=ev.bootstrap(single['records'])
    save(WORK/'results.json',reports);save(WORK/'selection.json',dict(variant=chosen['variant'],epochs=epochs,seeds=list(SEEDS),delivery_seed=seed,source_nll=source_nll,selection='Family selected by mean single-seed development; median source epochs. After ensemble failed, delivery seed selected by mean source calibration NLL. This is an exploratory follow-up.',development=single,ensemble_development=chosen,platform_score=None))
    print('SELECTED',chosen['variant'],'epochs',epochs,flush=True)

if __name__=='__main__':main()
