"""Fixed two-seed recipe checked with each whole source recording excluded."""
from run import *
def main():
    torch.set_num_threads(4);d=torch.load(WORK/'features.pt',weights_only=True);x=d['features'][False][:,:-5];y=d['labels'];rows=d['rows'];assignment=read(WORK/'no_clock_record_results.json')['assignment'];results=[]
    with np.load(WORK/'actual_absFalse_ordFalse.npz') as v:base=torch.from_numpy(v['base'].copy())
    mixed_all=torch.empty(960,5);head_all=torch.empty(960,5)
    for fold in range(3):
        fit_ix=[i for i,r in enumerate(rows) if assignment[r['identifier']]!=fold];test=[i for i,r in enumerate(rows) if assignment[r['identifier']]==fold];zz=[]
        for seed in (42,2026):
            _,_,states=train(x,y,fit_ix,[],False,f'record_ensemble_fold{fold}_seed{seed}',seed)
            state=states[50];h=CountHead(state['mean'],state['std']).eval();h.load_state_dict(state)
            with torch.no_grad():zz.append(h(x).exp())
        z=((zz[0]+zz[1])*.5).clamp_min(1e-8).log();mixed=(.5*base.softmax(-1)+.5*z.exp()).clamp_min(1e-8).log();mixed=torch.where((base.argmax(-1)<=1)[:,None],base,mixed)
        mixed_all[test]=mixed[test];head_all[test]=z[test]
        result=dict(fold=fold,head=metrics(z,y,base,test),mixed=metrics(mixed,y,base,test));results.append(result);print('ENSEMBLE_RECORD',result,flush=True)
    save(WORK/'ensemble_record_results.json',dict(folds=results,mixed=metrics(mixed_all,y,base,list(range(960))),head=metrics(head_all,y,base,list(range(960))),
      caveat='Heads/normalization excluded complete heldout records; the inherited049 backbone did not. Only descriptor-head standalone estimate is fully independent of those record labels.'))
if __name__=='__main__':main()
