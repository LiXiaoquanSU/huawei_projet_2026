"""Evaluate graph fallback only after all source/target-U fits have completed."""
from graph import *
evspec=importlib.util.spec_from_file_location('environment_eval_graph_shared',ROOT/'experiments/environment_protocol_20261025/evaluate.py')
ev=importlib.util.module_from_spec(evspec);evspec.loader.exec_module(ev)
GWORK=WORK/'graph'

def result(d,b,T,files):
    values=[]
    for f in files:
        cp=torch.load(f,weights_only=True)
        values.append(hybrid(d['raw'],cp['student_logits'],b,T,cp['amount']).softmax(-1))
    return torch.stack(values).mean(0).clamp_min(1e-12).log()

def main():
    assert (GWORK/'complete.json').exists();pro=read(WORK/'protocols.json');results=[]
    for protocol in ['DG','targetU']:
        item=dict(protocol=protocol,rooms={},records=[])
        for room in ROOMS:
            d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room);base=old.policy(d['raw'],b,T)
            dg=result(d,b,T,[GWORK/f'{room}_s{s}_source.pt' for s in SEEDS]);z=dg.clone()
            if protocol=='targetU':
                for fold,ff in enumerate(pro[room]['folds']):
                    proposed=result(d,b,T,[GWORK/f'{room}_s{s}_fold{fold}.pt' for s in SEEDS])
                    z[ff['test_indices']]=proposed[ff['test_indices']]
            ix=old.split(room,d['rows'])[2];metrics=old.metrics(z,d['labels'],base,ix);baseline=old.metrics(base,d['labels'],base,ix)
            item['rooms'][room]=dict(metrics=metrics,baseline=baseline,source_match=old.metrics(z,d['labels'],dg,ix))
            for record in pro[room]['target_records']:
                jj=[i for i in ix if d['rows'][i]['identifier']==record];y=d['labels'][jj]
                cc=int((z[jj].argmax(-1)==y).sum());bc=int((base[jj].argmax(-1)==y).sum());sc=int((dg[jj].argmax(-1)==y).sum())
                item['records'].append(dict(room=room,identifier=record,windows=len(jj),correct=cc,baseline_correct=bc,delta=cc-bc,matched_delta=cc-sc))
            torch.save(dict(logits=z,baseline=base,source_dg=dg,labels=d['labels'],indices=ix),GWORK/f'eval_{protocol}_{room}.pt')
        item['net']=sum(r['delta'] for r in item['records']);item['total']=sum(r['windows'] for r in item['records'])
        item['gain']=item['net']/item['total'];item['ci95']=ev.bootstrap(item['records']);item['matched_source_ci95']=ev.bootstrap(item['records'],'matched_delta')
        item['matched_source_net']=sum(r['matched_delta'] for r in item['records'])
        item['room_gains']={room:(item['rooms'][room]['metrics']['correct']-item['rooms'][room]['baseline']['correct'])/item['rooms'][room]['metrics']['total'] for room in ROOMS}
        item['passed']=item['gain']>=.02 and min(item['room_gains'].values())>=-.02 and sum(g>0 for g in item['room_gains'].values())>=2 and item['ci95'][0]>0
        results.append(item)
    dg,uda=results
    chosen='DG' if dg['passed'] else None
    if uda['passed'] and uda['matched_source_net']>0 and uda['matched_source_ci95'][0]>0 and min(dg['room_gains'].values())>=-.02:chosen='targetU'
    save(GWORK/'results.json',results);save(GWORK/'selection.json',dict(selected=chosen,platform_score=None,
       reason='DGcandidate must pass held-room gate; targetU only preferred if it adds positive paired CI over source-only graph, not merely beats original. Source calibration chose steps/blends.'))
    print(json.dumps([dict(protocol=r['protocol'],net=r['net'],rooms=r['room_gains'],ci=r['ci95'],relative_source=r['matched_source_net'],passed=r['passed']) for r in results],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
