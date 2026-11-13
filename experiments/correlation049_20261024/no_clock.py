"""Check that covariance branch gain does not require packet-clock metadata."""
from run import *
def main():
    torch.set_num_threads(4);d=torch.load(WORK/'features.pt',weights_only=True);x=d['features'][False][:,:-5];y=d['labels'];rows=d['rows'];steps=[];reports=[]
    save(WORK/'no_clock_plan.json',dict(reason='Remove the5 ATI summary features to diagnose sampling-pattern shortcuts. Retain only114 covariance/ACF/energy-concentration features.',mix=.5,guard='keep01',seed=42))
    for room in ('a1','c1','g1','e1','actual'):
        actual=room=='actual'
        with np.load(WORK/f'{room}_absFalse_ordFalse.npz') as v:base=torch.from_numpy(v['base'].copy());test=v['test_indices'].tolist()
        if actual:fit_ix=list(range(960));cal_ix=[]
        else:
            manifest=read(OLD/'results'/f's42_{room}'/'manifest.json');ids={k:{r['identifier'] for r in v} for k,v in manifest.items() if k in ('fit','calibration')}
            fit_ix=[i for i,r in enumerate(rows) if r['identifier'] in ids['fit']];cal_ix=[i for i,r in enumerate(rows) if r['identifier'] in ids['calibration']]
        z,step,states=train(x,y,fit_ix,cal_ix,False,f'{room}_no_clock')
        if actual:
            step=min(states,key=lambda k:abs(k-float(np.median(steps))));state=states[step];h=CountHead(state['mean'],state['std']).eval();h.load_state_dict(state)
            with torch.no_grad():z=h(x)
            torch.save(dict(state=state,ordinal=False,absolute=False,no_clock=True,chosen_step=step,fit_indices=fit_ix,seed=42),WORK/'actual_no_clock_final.pt')
        else:steps.append(step)
        candidate=(.5*base.softmax(-1)+.5*z.exp()).clamp_min(1e-8).log();candidate=torch.where((base.argmax(-1)<=1)[:,None],base,candidate)
        report=dict(room=room,step=step,metrics=metrics(candidate,y,base,test),head=metrics(z,y,base,test));reports.append(report)
        np.savez_compressed(WORK/f'{room}_no_clock_predictions.npz',head_logits=z.numpy(),logits=candidate.numpy(),base=base.numpy(),labels=y.numpy(),test_indices=test)
        print('NO_CLOCK',report,flush=True)
    save(WORK/'no_clock_results.json',reports)
if __name__=='__main__':main()
