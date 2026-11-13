"""Constant-output controls and exact feature-space symmetry checks."""
from run import *
def main():
    torch.set_num_threads(4);records=[]
    for room in ('a1','c1','g1','e1','actual'):
        d=torch.load(WORK/'features.pt',weights_only=True);rows=d['rows'];y=d['labels']
        with np.load(WORK/f'no_clock_replicate_{room}_ensemble.npz') as v:
            aux=torch.from_numpy(v['head_logits'].copy());base=torch.from_numpy(v['base'].copy());test=v['test_indices'].tolist()
        if room=='actual':fit_ix=list(range(960))
        else:
            ids={r['identifier'] for r in read(OLD/'results'/f's42_{room}'/'manifest.json')['fit']};fit_ix=[i for i,r in enumerate(rows) if r['identifier'] in ids]
        for method,p in [('uniform',torch.ones(5)/5),('source_prior',torch.bincount(y[fit_ix],minlength=5).float()/len(fit_ix)),('mean_prediction',aux[fit_ix].exp().mean(0)),('actual_features',aux.exp())]:
            z=(.5*base.softmax(-1)+.5*p).clamp_min(1e-8).log();z=torch.where((base.argmax(-1)<=1)[:,None],base,z)
            records.append(dict(room=room,control=method,metrics=metrics(z,y,base,test)))
    save(WORK/'constant_controls.json',records)
    d=torch.load(CACHE,weights_only=True);reverse=torch.load(OLD/'reverse_features.pt',weights_only=True)
    ix=torch.arange(0,960,31);t=d['t'][ix];s=d['s'][ix];descriptor=Descriptor().eval()
    with torch.no_grad():
        ref=descriptor(t,s,False,False);a=descriptor(reverse['t'][ix],reverse['s'][ix],False,False)
        rawreverse=float((a-ref).abs().max());assert rawreverse<.002,rawreverse
        flipped=float((descriptor(t.flip(-1),s,False,False)-ref).abs().max());assert flipped<1e-4,flipped
        perms=[]
        import itertools
        for p in itertools.permutations(range(3)):
            index,sign,si=shared.permutation_indices(p)
            out=descriptor(t.index_select(1,index)*sign,s.index_select(1,si),False,False)
            error=float((out-ref).abs().max());assert error<1e-4,error
            perms.append(dict(permutation=p,error=error))
    save(WORK/'symmetry.json',dict(windows=len(ix),rx_permutations=perms,feature_time_reversal_max_error=flipped,
      raw_reverse_preprocessed_features_max_error=rawreverse,features=114,no_clock_or_absolute_summary=True))
    print('CONTROL_CHECKS',records,rawreverse,perms,flush=True)
if __name__=='__main__':main()
