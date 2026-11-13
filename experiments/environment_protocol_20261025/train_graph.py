"""Source-supervised graph fallback, with/without label-free targetteacher distillation."""
from graph import *
GWORK=WORK/'graph'

def fit(room,seed,fold,d,p,tt,ss,b,T):
    suffix='source' if fold is None else f'fold{fold}'
    path=GWORK/f'{room}_s{seed}_{suffix}.pt'
    if path.exists():return
    torch.manual_seed(seed);np.random.seed(seed);model=ReceiverGraph().cuda()
    initial=invariance(model,tt[:2,0],ss[:2,0]);model.train()
    fit=p['source_fit'];cal=p['source_calibration'];y=d['labels'].cuda()
    teacher=fallback(d['raw'].cuda(),b,T).softmax(-1)
    gen=torch.Generator().manual_seed(seed+173)
    draws=torch.multinomial(weights(d['rows'],d['labels'],fit),16*150,replacement=True,generator=gen).reshape(-1,16)
    indices=torch.tensor(fit)[draws].cuda();target=None
    if fold is not None:
        target=torch.tensor(p['folds'][fold]['adapt_indices']).cuda()
        ug=torch.Generator().manual_seed(seed+743)
        udraws=torch.randint(len(target),(150,16),generator=ug).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.02);snapshots=[];history=[]
    for step in range(1,151):
        model.train();ix=indices[step-1];optimizer.zero_grad(set_to_none=True)
        z=model(tt[ix].flatten(0,1),ss[ix].flatten(0,1)).reshape(-1,2,5)
        ce=F.cross_entropy(z.flatten(0,1),y[ix,None].expand(-1,2).flatten(),label_smoothing=.02)
        distill=-(teacher[ix,None]*z.log_softmax(-1)).sum(-1).mean()
        pv=z.softmax(-1);mean=pv.mean(1,keepdim=True);js=(pv*(pv.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(-1).mean()
        loss=ce+.5*distill+.05*js
        if target is not None:
            ui=target[udraws[step-1]]
            uz=model(tt[ui].flatten(0,1),ss[ui].flatten(0,1)).reshape(-1,2,5)
            loss=loss+.25*(-teacher[ui,None]*uz.log_softmax(-1)).sum(-1).mean()
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);optimizer.step()
        if step in [50,150]:
            scores=student_scores(model,tt[cal],ss[cal]);yy=d['labels'][cal]
            w=1/torch.bincount(yy,minlength=5).clamp_min(1)[yy].float();w/=w.sum()
            choices=[]
            for amount in [0,.25,.5]:
                zcal=hybrid(d['raw'][cal],scores,b.cpu(),T,amount)
                choices.append((float((F.cross_entropy(zcal,yy,reduction='none')*w).sum()),amount))
            nll,amount=min(choices);snapshots.append(dict(step=step,amount=amount,calibration_nll=nll,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
            history.append(dict(step=step,loss=float(loss.detach()),source_calibration_choices=choices))
    selected=min(snapshots,key=lambda r:(r['calibration_nll'],r['amount'],r['step']))
    model.load_state_dict(selected['state']);model.eval();error=invariance(model,tt[:2,0],ss[:2,0])
    outputs=student_scores(model,tt,ss)
    torch.save(dict(state=selected['state'],student_logits=outputs,amount=selected['amount'],step=selected['step'],source_calibration_nll=selected['calibration_nll'],
                    history=history,room=room,seed=seed,fold=fold,initial_invariance_error=initial,final_invariance_error=error,
                    parameters=sum(v.numel() for v in model.parameters()),labelled_draws_sha256=hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest(),
                    target_adaptation_records=p['folds'][fold]['adapt_records'] if fold is not None else [],
                    target_test_records=p['folds'][fold]['test_records'] if fold is not None else [],
                    source_fit_records=sorted({d['rows'][i]['identifier'] for i in fit})),path)
    print('GRAPH_FIT',room,seed,suffix,'step',selected['step'],'blend',selected['amount'],'invariance',error,flush=True)
    del model,optimizer,teacher;gc.collect();torch.cuda.empty_cache()

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;GWORK.mkdir(parents=True,exist_ok=True)
    save(GWORK/'plan.json',dict(stage=3,reason='584parameter conditioning and label-conditional nuisance projection did not yield robust gain. Input-relation structure follows observedRxTTA benefit.',
       architecture='SharedRxnodes, six directedcos/sinphase edges withconjugacy, two graphmessages, time-aware encoder. Strictreceiverrelabelinvariancechecked.',
       teacher='Only calibrated12viewfallback distribution; retain original054v0/v45protection exactly. No distillation of non-invariant original-view guard.',
       protocols=['DGsource','targetUrecordsdisjoint'],seeds=SEEDS,steps=[50,150],amounts=[0,.25,.5],
       selection='Only sourcecalibration selects step andblend, zero blend identity option. Targetlabels evaluation only. Priorfailed full-studentdistill029/031notignored.',
       targetU='Softteacher distillation withoutconfidence gate, no targettrue labels, no uniform-per-room assumption. Teacher can be wrong; not claimed target supervision.'))
    tt=torch.from_numpy(np.array(np.load(PREV/'t.npy',mmap_mode='r'),copy=True)).cuda()
    ss=torch.from_numpy(np.array(np.load(PREV/'s.npy',mmap_mode='r'),copy=True)).cuda();p=read(WORK/'protocols.json')
    for room in ROOMS:
        d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room);b=b.cuda()
        for seed in SEEDS:
            fit(room,seed,None,d,p[room],tt,ss,b,T)
            for fold in [0,1]:fit(room,seed,fold,d,p[room],tt,ss,b,T)
    save(GWORK/'complete.json',dict(fits=24))

if __name__=='__main__':main()
