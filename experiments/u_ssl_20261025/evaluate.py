"""Paired comparisons, two-seed probabilities and a fixed0->1 diagnostic guard."""
from ssl_common import *

def rescue(base,z,raw,b,T):
    votes=(raw/T+b+raw.new_tensor([0,.35,0,0,0])).argmax(-1)
    margin=(raw[:,:,1]-raw[:,:,0])/T
    mask=(base.argmax(-1)==0)&(z.argmax(-1)==1)&((votes==1).sum(-1)>=3)&(margin[:,0]>=np.log(4))&(margin[:,1]>=np.log(4))
    return torch.where(mask[:,None],z,base),mask

def main():
    results=[];details=[]
    for room in ('a1','c1','g1','e1','actual'):
        data=torch.load(WORK/'prefixes'/room/'source.pt',weights_only=True);y=data['labels'];rows=data['rows']
        if room=='actual':_,b=build(checkpoint(room));T=1.
        else:
            cal=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline'];b=torch.tensor(cal['bias']);T=cal['temperature']
        for method in ('supervised','source_ssl','official_u','paired','paired_u'):
            complete=[read(WORK/'fits'/f'{room}_{method}_s{s}'/'complete.json') for s in (42,2026)]
            # Identical source sampling across the three experimental methods.
            for seed,c in zip((42,2026),complete):
                reference=read(WORK/'fits'/f'{room}_supervised_s{seed}'/'complete.json');assert reference['source_sampler_sha256']==c['source_sampler_sha256']
            for amount in (.25,.5,1.):
                values=[]
                for seed in (42,2026):
                    with np.load(WORK/'fits'/f'{room}_{method}_s{seed}'/f'predictions_{amount}.npz') as v:
                        values.append({k:torch.from_numpy(v[k].copy()) for k in ('raw','logits','base','labels','test_indices')})
                assert torch.equal(values[0]['base'],values[1]['base']) and torch.equal(values[0]['test_indices'],values[1]['test_indices'])
                base=values[0]['base'];test=values[0]['test_indices'].tolist()
                raw=((values[0]['raw'].exp()+values[1]['raw'].exp())*.5).clamp_min(1e-8).log()
                variants=[('seed42',values[0]['logits'],values[0]['raw']),('seed2026',values[1]['logits'],values[1]['raw']),('ensemble',policy(raw,b,T),raw)]
                for variant,z,rawz in variants:
                    guarded,mask=rescue(base,z,rawz,b,T)
                    for deployment,zz in [('full',z),('rescue',guarded)]:
                        m=metrics(zz,y,base,test);pred=zz.argmax(-1);bp=base.argmax(-1);ix=torch.tensor(test)
                        empty=(y[ix]==0);m['added_empty_false_positives']=int(((pred[ix]!=0)&(bp[ix]==0)&empty).sum())
                        result=dict(room=room,method=method,amount=amount,variant=variant,deployment=deployment,metrics=m)
                        results.append(result)
                        if variant=='ensemble':
                            np.savez_compressed(WORK/f'{room}_{method}_{amount}_{deployment}.npz',logits=zz.numpy(),raw=rawz.numpy(),labels=y.numpy(),base=base.numpy(),test_indices=test)
                    if room=='actual' and variant=='ensemble':
                        correct=(z.argmax(-1)==y)&(base.argmax(-1)!=y);broken=(z.argmax(-1)!=y)&(base.argmax(-1)==y)
                        details.append(dict(method=method,amount=amount,corrected_records=sorted({rows[i]['identifier'] for i in torch.nonzero(correct).flatten().tolist()}),broken_records=sorted({rows[i]['identifier'] for i in torch.nonzero(broken).flatten().tolist()})))
    save(WORK/'evaluations.json',results);save(WORK/'source_record_changes.json',details)
    summary=[]
    for method in ('supervised','source_ssl','official_u','paired','paired_u'):
        for amount in (.25,.5,1.):
            for variant in ('seed42','seed2026','ensemble'):
                for deployment in ('full','rescue'):
                    rr=[r for r in results if r['room']!='actual' and r['method']==method and r['amount']==amount and r['variant']==variant and r['deployment']==deployment]
                    source=next(r for r in results if r['room']=='actual' and r['method']==method and r['amount']==amount and r['variant']==variant and r['deployment']==deployment)
                    net=sum(r['metrics']['corrected']-r['metrics']['broken'] for r in rr);room_net={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in rr}
                    worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in rr)
                    sourceok=all(source['metrics']['confusion_matrix'][k][k]>=old for k,old in enumerate([206,198,66,78,140]))
                    fp=sum(r['metrics']['added_empty_false_positives'] for r in rr)
                    eligible=(net>0 and sum(v>0 for v in room_net.values())>=2 and worst>=-.02 and sourceok and (deployment!='rescue' or fp<=2))
                    s=dict(method=method,amount=amount,variant=variant,deployment=deployment,proxy_net=net,proxy_added_empty_fp=fp,room_net=room_net,worst_room_delta=worst,source=source['metrics'],eligible=eligible)
                    summary.append(s)
    save(WORK/'summary.json',summary)
    print('ELIGIBLE',json.dumps([v for v in summary if v['eligible']]),flush=True)
    print('DEFAULT_U',json.dumps([v for v in summary if v['method']=='official_u' and v['amount']==.5 and v['variant']=='ensemble']),flush=True)

if __name__=='__main__':main()
