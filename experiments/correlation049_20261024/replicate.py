"""Second fixed seed and whole-record descriptor-head crosschecks."""
import hashlib
from run import *

def main():
    no_clock='--no-clock' in sys.argv;prefix='no_clock_' if no_clock else ''
    torch.set_num_threads(4);d=torch.load(WORK/'features.pt',weights_only=True);x=d['features'][False][:,:-5] if no_clock else d['features'][False];y=d['labels'];rows=d['rows'];reports=[];selected_steps=[]
    save(WORK/f'{prefix}replicate_plan.json',dict(method='normalized covariance + softmax',no_clock=no_clock,seed=2026,mix=.5,guard='base01 exact output preserved',
      selection='Repeat selected seed42 recipe; second seed and equal probability ensemble. Record validation fixed50steps, not chosen on heldout records.',
      caveat='Descriptor head has no learned encoder. Mixed whole-record baseline still inherited backbone label exposure.'))
    for room in ('a1','c1','g1','e1','actual'):
        actual=room=='actual'
        path=WORK/f'{room}_no_clock_predictions.npz' if no_clock else WORK/f'{room}_absFalse_ordFalse.npz'
        with np.load(path) as v:
            first=torch.from_numpy(v['head_logits' if no_clock else 'logits'].copy());base=torch.from_numpy(v['base'].copy());ix=v['test_indices'].tolist()
        if actual:fit_ix=list(range(960));cal_ix=[]
        else:
            manifest=read(OLD/'results'/f's42_{room}'/'manifest.json');ids={k:{r['identifier'] for r in v} for k,v in manifest.items() if k in ('fit','calibration')}
            fit_ix=[i for i,r in enumerate(rows) if r['identifier'] in ids['fit']];cal_ix=[i for i,r in enumerate(rows) if r['identifier'] in ids['calibration']]
        second,step,snapshots=train(x,y,fit_ix,cal_ix,False,f'{prefix}{room}_normalized_seed2026',seed=2026)
        if actual:
            median=float(np.median(selected_steps));step=min(snapshots,key=lambda k:abs(k-median));state=snapshots[step]
            h=CountHead(state['mean'],state['std']).eval();h.load_state_dict(state)
            with torch.no_grad():second=h(x)
            torch.save(dict(state=state,ordinal=False,absolute=False,no_clock=no_clock,chosen_step=step,fit_indices=fit_ix,seed=2026),WORK/f'{prefix}actual_normalized_seed2026_final.pt')
        else:selected_steps.append(step)
        for name,z in [('seed42',first),('seed2026',second),('ensemble',(first.exp()+second.exp()).mul(.5).clamp_min(1e-8).log())]:
            mixed=(.5*base.softmax(-1)+.5*z.exp()).clamp_min(1e-8).log();mixed=torch.where((base.argmax(-1)<=1)[:,None],base,mixed)
            report=dict(room=room,variant=name,step=step,metrics=metrics(mixed,y,base,ix));reports.append(report)
            np.savez_compressed(WORK/f'{prefix}replicate_{room}_{name}.npz',head_logits=z.numpy(),logits=mixed.numpy(),base=base.numpy(),labels=y.numpy(),test_indices=ix)
            print('REPLICATE',report,flush=True)
    save(WORK/f'{prefix}replicate_results.json',reports)
    assignment={};offset=0;unique={r['identifier']:r for r in rows}
    for label,room in sorted({(r['headcount'],r['room']) for r in unique.values()}):
        ids=sorted([k for k,r in unique.items() if r['headcount']==label and r['room']==room],key=lambda k:hashlib.sha256(('correlation42:'+k).encode()).hexdigest())
        for j,key in enumerate(ids):assignment[key]=(offset+j)%3
        offset=(offset+len(ids))%3
    holdout=[]
    with np.load(WORK/'actual_absFalse_ordFalse.npz') as v:base=torch.from_numpy(v['base'].copy())
    # Fit same tiny head, but choose a fixed50-step snapshot for each excluded-record fold.
    allpred=np.empty((960,5),np.float32);allhead=np.empty((960,5),np.float32)
    for fold in range(3):
        fit_ix=[i for i,r in enumerate(rows) if assignment[r['identifier']]!=fold];test=[i for i,r in enumerate(rows) if assignment[r['identifier']]==fold]
        _,_,states=train(x,y,fit_ix,[],False,f'{prefix}record_fold{fold}_seed42')
        state=states[50];h=CountHead(state['mean'],state['std']).eval();h.load_state_dict(state)
        with torch.no_grad():z=h(x)
        mixed=(.5*base.softmax(-1)+.5*z.exp()).clamp_min(1e-8).log();mixed=torch.where((base.argmax(-1)<=1)[:,None],base,mixed)
        allpred[test]=mixed[test].numpy();allhead[test]=z[test].numpy()
        r=dict(fold=fold,head=metrics(z,y,base,test),mixed=metrics(mixed,y,base,test));holdout.append(r);print('CORRELATION_RECORD',r,flush=True)
    save(WORK/f'{prefix}record_results.json',dict(assignment=assignment,folds=holdout,head=metrics(torch.from_numpy(allhead),y,base,list(range(960))),mixed=metrics(torch.from_numpy(allpred),y,base,list(range(960))),
      caveat='Standalone descriptor head trained without its heldout records. Mixed pretrained049 baseline saw all source records; mixed metric not end-to-end independent.'))
if __name__=='__main__':main()
