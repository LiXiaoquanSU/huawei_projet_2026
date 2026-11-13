"""Fixed hierarchical decision; source-only epoch selection, reused room evaluation."""
from train_strength import *
PRIOR=Path('D:/huawei2026-experiments/shared_hierarchy_20261025')

def hard(z):return torch.where(z[:,0].exp()>=.5,0,z[:,1:].argmax(-1)+1)

def main():
    spec=importlib.util.spec_from_file_location('strength_bootstrap',ROOT/'experiments/protection_arbiter_20261025/evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    r=dict(rooms={},records=[],epochs=[])
    for room in ROOMS:
        folder=WORK/f'{room}_strength_s2026';info=read(folder/'result.json');r['epochs'].append(info['selected_epoch']);d=torch.load(folder/'predictions.pt',weights_only=True)
        prev=torch.load(PRIOR/f'{room}_plain_s2026'/'predictions.pt',weights_only=True);ix=d['test_indices'];assert ix==prev['test_indices'];y=d['labels'];pred=hard(d['logits']);oldpred=hard(prev['logits'])
        ref=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room);bp=old.policy(ref['raw'],b,T).argmax(-1)
        cm=torch.bincount(y[ix]*5+pred[ix],minlength=25).reshape(5,5)
        m=dict(correct=int((pred[ix]==y[ix]).sum()),previous_correct=int((oldpred[ix]==y[ix]).sum()),baseline_correct=int((bp[ix]==y[ix]).sum()),total=len(ix),confusion_matrix=cm.tolist(),source_epoch=info['selected_epoch']);r['rooms'][room]=m
        for ident in sorted({d['rows'][i]['identifier'] for i in ix}):
            jj=[i for i in ix if d['rows'][i]['identifier']==ident];c=int((pred[jj]==y[jj]).sum());bc=int((bp[jj]==y[jj]).sum());oc=int((oldpred[jj]==y[jj]).sum())
            r['records'].append(dict(room=room,identifier=ident,windows=len(jj),correct=c,baseline_correct=bc,previous_correct=oc,delta=c-bc))
    r['correct']=sum(x['correct'] for x in r['records']);r['total']=880;r['baseline_correct']=278;r['previous_correct']=269;r['ci95']=ev.bootstrap(r['records'])
    r['ci_vs_previous']=ev.bootstrap([dict(x,delta=x['correct']-x['previous_correct']) for x in r['records']])
    save(WORK/'results.json',r);save(WORK/'selection.json',dict(variant='strength',epochs=int(np.median(r['epochs'])),seed=2026,delivery_seed=2026,scheduler_horizon=20,decision='fixed0.5 presence then count',development=r,platform_score=None))
    print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
