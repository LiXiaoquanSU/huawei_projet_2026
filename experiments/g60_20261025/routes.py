"""Three pre-stated consensus exceptions to054 hard routing; no fitting."""
from g60_common import *
def main():
    save(WORK/'route_plan.json',dict(methods=['consensus_bridge','rescue01','rescue_empty'],thresholds='bridge:5/6votes, margin.2, blend.25; rescue01:5/6votes,P1.6; empty:6/6votes,P0.8',selection='same positive>=2room and no>2pp loss/sourceclass nondecrease criteria; no threshold search',reason='Initial pooling changes failed. Test consensus-limited exceptions to frozen branch errors.'))
    results=[]
    for room in ('a1','c1','g1','e1','actual'):
        d=torch.load(WORK/f'{room}.pt',weights_only=True);b,T=calibration(room);_,_,ti=split(room,d['rows']);base=policy(d['raw'],b,T)
        for method in ('consensus_bridge','rescue01','rescue_empty'):
            z=policy(d['raw'],b,T,method);m=metrics(z,d['labels'],base,ti);results.append(dict(room=room,method=method,metrics=m));np.savez_compressed(WORK/f'{room}_{method}.npz',logits=z.numpy(),base=base.numpy(),labels=d['labels'].numpy(),test_indices=ti)
    old=next(x for x in read(WORK/'summary.json') if x['method']=='baseline')['source'];summary=[]
    for method in ('consensus_bridge','rescue01','rescue_empty'):
        rr=[r for r in results if r['room']!='actual' and r['method']==method];source=next(r for r in results if r['room']=='actual' and r['method']==method)['metrics']
        net={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in rr};worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in rr)
        eligible=sum(net.values())>0 and sum(v>0 for v in net.values())>=2 and worst>=-.02 and all(source['confusion_matrix'][k][k]>=old['confusion_matrix'][k][k] for k in range(5))
        summary.append(dict(method=method,proxy_net=sum(net.values()),room_net=net,source=source,worst_room_delta=worst,eligible=eligible))
    save(WORK/'route_results.json',results);save(WORK/'route_summary.json',summary);print('G60_ROUTES',json.dumps(summary),flush=True)
    print('G60_DIAGNOSTIC',read(WORK/'actual_branch_diagnostic.json')['branches'],flush=True)
if __name__=='__main__':main()
