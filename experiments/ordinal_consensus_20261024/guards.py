"""Separate fixed aggregation changes from movement of old protection gates."""
from run import *

def main():
    torch.set_num_threads(4);reports=[]
    save(WORK/'guard_plan.json',dict(guards=['keep0','keep01','legacy'],
      rationale='Successful historical inference preserved selected base outputs. Recomputing all gates with transformed evidence confounds evidence quality and route movement; compare frozen baseline gates.',
      selection='Same source-class and room damage limits as initial plan; guard family added after ungated development results, therefore additional development selection.'))
    multis=Path('D:/huawei2026-experiments/multiscale049_20261024')
    for room in ('actual','a1','c1','g1','e1'):
        actual=room=='actual';e=torch.load(EVIDENCE/('actual_evidence.pt' if actual else f'proxy_evidence_{room}.pt'),weights_only=True)
        if actual:b=e['bias'];T=1.
        else:
            cal=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline'];b=torch.tensor(cal['bias']);T=cal['temperature']
        base=policy(e['logits'],b,T);prediction=base.argmax(-1)
        raw=e['logits'];original=(raw[:,0]/T+b).argmax(-1)
        identity=(raw[:,:2].exp().mean(1).clamp_min(1e-12).log()/T+b).argmax(-1)
        masks=dict(keep0=prediction==0,keep01=prediction<=1,legacy=(original==1)|(identity==0)|(identity==2))
        recipes=[(WORK,m) for m in METHODS]+[(multis,m) for m in ('time_smooth','frequency_smooth','local100','local120')]
        for directory,method in recipes:
            for strength in (.25,.5,1.):
                with np.load(directory/f'{room}_{method}_{strength}.npz') as v:
                    alternate=torch.from_numpy(v['logits'].copy());y=torch.from_numpy(v['labels'].copy());ix=v['test_indices'].tolist()
                for guard,mask in masks.items():
                    z=torch.where(mask[:,None],base,alternate)
                    report=dict(room=room,method=method,strength=strength,guard=guard,metrics=metric(z,y,base,ix));reports.append(report)
    save(WORK/'guard_results.json',reports);summary=[]
    for method in [*METHODS,'time_smooth','frequency_smooth','local100','local120']:
        for strength in (.25,.5,1.):
            for guard in ('keep0','keep01','legacy'):
                rr=[r for r in reports if r['room']!='actual' and r['method']==method and r['strength']==strength and r['guard']==guard]
                source=next(r for r in reports if r['room']=='actual' and r['method']==method and r['strength']==strength and r['guard']==guard)
                net=sum(r['metrics']['corrected']-r['metrics']['broken'] for r in rr)
                worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in rr)
                sourceok=all(source['metrics']['confusion_matrix'][k][k]>=old for k,old in enumerate([206,198,66,78,140]))
                record=dict(method=method,strength=strength,guard=guard,proxy_net=net,worst_room_delta=worst,
                  room_net={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in rr},source=source['metrics'],eligible=net>0 and worst>=-.02 and sourceok)
                summary.append(record)
    save(WORK/'guard_summary.json',summary)
    print('ELIGIBLE',json.dumps([r for r in summary if r['eligible']]),flush=True)
if __name__=='__main__':main()
