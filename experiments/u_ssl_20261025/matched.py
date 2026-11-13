"""Same120-update, same0.5 interpolation comparison, independent of epoch selection."""
from ssl_common import *
def main():
    torch.set_num_threads(4);reports=read(WORK/'matched120.json') if (WORK/'matched120.json').exists() else []
    for room in ('a1','c1','g1','e1','actual'):
        source=torch.load(WORK/'prefixes'/room/'source.pt',weights_only=True);model,b=build(checkpoint(room),full=room=='actual')
        student=Student(model).cuda().eval();del model
        if room=='actual':T=1.;ix=list(range(960))
        else:
            cal=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline'];b=torch.tensor(cal['bias']);T=cal['temperature']
            manifest=read(OLD/'results'/f's42_{room}'/'manifest.json');ids={r['identifier'] for r in manifest['test']};ix=[i for i,r in enumerate(source['rows']) if r['identifier'] in ids]
        base=policy(source['logits'],b,T)
        for method in ('supervised','source_ssl','official_u','paired','paired_u'):
            if all(any(r['room']==room and r['method']==method and r['variant']==variant for r in reports) for variant in ('seed42','seed2026','ensemble')):continue
            rr=[]
            for seed in (42,2026):
                cp=torch.load(WORK/'fits'/f'{room}_{method}_s{seed}'/'step120.pt',weights_only=True)
                state={k:(cp['initial'][k]+v)*.5 for k,v in cp['state'].items()};student.load_state_dict(state,strict=False)
                zz,raw,_=scores(student,source,b,T);rr.append(raw)
                reports.append(dict(room=room,method=method,variant=f'seed{seed}',step=120,amount=.5,metrics=metrics(zz,source['labels'],base,ix)))
            raw=((rr[0].exp()+rr[1].exp())*.5).clamp_min(1e-8).log();zz=policy(raw,b,T)
            report=dict(room=room,method=method,variant='ensemble',step=120,amount=.5,metrics=metrics(zz,source['labels'],base,ix));reports.append(report)
            print('MATCHED120',room,method,report['metrics']['corrected'],report['metrics']['broken'],flush=True)
        del source,student;torch.cuda.empty_cache()
    save(WORK/'matched120.json',reports)
if __name__=='__main__':main()
