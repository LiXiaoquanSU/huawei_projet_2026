"""Ordered-count alternative: aggregate four nested thresholds across six Rx views."""
from n53_common import *
def main():
    save(WORK/'ordinal_plan.json',dict(methods=['rx_ordinal25','rx_ordinal50'],definition='EachRx timepair normalizedclassprobabilities ->P(count>=1..4);average threshold logits across6orders;reconstruct monotoneordinaldistribution;blend25%/50%with054fallback. Exact same054protectedroutes/finalbias. NoGcalbias,no fitting.',reason='Test count ordering rather than treating0..4as unrelated categories. This is a new fixed hypothesis; previously tested opinionordinal pool includedactualGcaltable and wasdependent.',selection='Same held-outroom criteria and source4-person nonregression;no threshold sweep.'))
    results=[]
    for room in ('a1','c1','g1','e1','actual'):
        d=torch.load(WORK/f'{room}.pt',weights_only=True);b,T=old.calibration(room);base=old.policy(d['raw'],b,T);_,_,ti=old.split(room,d['rows'])
        for method in ('rx_ordinal25','rx_ordinal50'):
            z=pool(d['raw'],b,T,method);m=old.metrics(z,d['labels'],base,ti);results.append(dict(room=room,method=method,metrics=m));np.savez_compressed(WORK/f'{room}_{method}.npz',logits=z.numpy(),base=base.numpy(),labels=d['labels'].numpy(),test_indices=ti)
    baseline=next(x for x in read(WORK/'results.json') if x['room']=='actual' and x['method']=='baseline')['metrics'];summary=[]
    for method in ('rx_ordinal25','rx_ordinal50'):
        rr=[r for r in results if r['room']!='actual' and r['method']==method];s=next(r for r in results if r['room']=='actual' and r['method']==method)['metrics'];net={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in rr};worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in rr)
        summary.append(dict(method=method,source=s,proxy_net=sum(net.values()),room_net=net,worst_room_delta=worst,primary_proxy_independent=True,eligible=sum(net.values())>0 and sum(v>0 for v in net.values())>=2 and worst>=-.02 and s['confusion_matrix'][4][4]>=baseline['confusion_matrix'][4][4]))
    save(WORK/'ordinal_results.json',results);save(WORK/'ordinal_summary.json',summary);print('N53_ORDINAL',summary,flush=True)
if __name__=='__main__':main()
