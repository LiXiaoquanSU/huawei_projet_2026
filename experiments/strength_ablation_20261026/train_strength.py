"""Room-disjoint evaluation, source-only checkpoint selection, then full-data fit."""
import sys,copy,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/protection_arbiter_20261025'))
from arbiter_common import *
from crossfit import sample_weights,actual_split
from strength_network import SharedHierarchy,crossroom_loss,OriginalHierarchy
WORK=Path('D:/huawei2026-experiments/strength_ablation_20261026')
SEEDS=(2026,);EPOCHS=20;VARIANTS=('strength',)

def predict(model,t,s,ix):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(ix),24):
            jj=ix[start:start+24];v=model(t[jj].flatten(0,1),s[jj].flatten(0,1)).reshape(-1,2,5)
            out.append(v.exp().mean(1).clamp_min(1e-12).log().cpu())
    return torch.cat(out)

def invariance(model,t,s):
    model.eval();err=0.
    with torch.inference_mode():
        z=model(t,s).exp()
        for perm in old.PERMS:
            ix,sg,si=[x.to(t.device) for x in old.permutation_indices(perm)]
            err=max(err,float((model(t.index_select(1,ix)*sg,s.index_select(1,si)).exp()-z).abs().max()))
    assert err<1e-5,err
    return err

def fit(room,seed,variant,rows,t,s,y,epochs=EPOCHS,full=False):
    name=f'{room}_{variant}_s{seed}';folder=WORK/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    fitix,cal,test=actual_split(rows) if room=='source_holdout' else old.split(room,rows)
    if full:fitix=list(range(len(rows)));cal=[];test=fitix
    ids=lambda ix:sorted({rows[i]['identifier'] for i in ix})
    if not full:
        assert not set(ids(fitix))&set(ids(cal));assert not set(ids(fitix))&set(ids(test));assert not set(ids(cal))&set(ids(test))
    if room in ROOMS:assert all(rows[i]['room']!=room for i in fitix+cal)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=(OriginalHierarchy() if variant.startswith('control') else SharedHierarchy()).cuda();initial=invariance(model,t[:2,0],s[:2,0])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=.00006)
    gen=torch.Generator().manual_seed(seed);w=sample_weights(rows,y,fitix)
    roomnames=[r['room'] for r in rows];mapping={r:i for i,r in enumerate(sorted(set(roomnames)-{'unknown'}))}
    rr=torch.tensor([mapping.get(r,-1) for r in roomnames],device='cuda');fi=torch.tensor(fitix,device='cuda')
    history=[];best=None
    for epoch in range(1,epochs+1):
        model.train();draw=fi[torch.multinomial(w,len(fitix),replacement=True,generator=gen).cuda()];losses=[]
        for start in range(0,len(draw),24):
            jj=draw[start:start+24];tt=t[jj].flatten(0,1);ss=s[jj].flatten(0,1)
            # Small zero-mean feature noise; timing untouched. No invented room labels.
            tt=torch.cat([tt[:,:960]+.01*torch.randn_like(tt[:,:960]),tt[:,960:]],1)
            tasks=model.tasks(tt,ss);yy=y[jj,None].expand(-1,2).flatten();logp=tasks['logits'];p=logp.exp().reshape(-1,2,5);mean=p.mean(1,keepdim=True)
            ce=F.nll_loss(logp,yy);ordinal=F.binary_cross_entropy_with_logits(tasks['ordinal'],(yy[:,None]>=torch.tensor([2,3,4],device='cuda')).float())
            consistency=(p*(p.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(-1).mean()
            contrast=crossroom_loss(tasks['embedding'].reshape(-1,2,128).mean(1),y[jj],rr[jj]) if variant=='crossroom' else ce*0
            loss=ce+.2*ordinal+.05*consistency+.03*contrast
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);optimizer.step();losses.append(float(loss.detach()))
        scheduler.step();entry=dict(epoch=epoch,loss=float(np.mean(losses)))
        if cal:
            z=predict(model,t,s,cal);yy=y[cal].cpu();cw=1/torch.bincount(yy,minlength=5).clamp_min(1)[yy].float();cw/=cw.sum();nll=float((-z.gather(1,yy[:,None]).squeeze(1)*cw).sum())
            entry.update(calibration_balanced_nll=nll,calibration_correct=int((z.argmax(-1)==yy).sum()))
            if best is None or nll<best['nll']:best=dict(nll=nll,epoch=epoch,state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}))
        history.append(entry)
        if epoch%5==0:print('TRAIN',name,entry,flush=True)
    if not cal:best=dict(epoch=epochs,nll=None,state={k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best['state']);error=invariance(model,t[:2,0],s[:2,0]);z=predict(model,t,s,list(range(len(rows))))
    torch.save(dict(model_state=best['state'],seed=seed,variant=variant,epoch=best['epoch'],history=history,fit_records=ids(fitix),calibration_records=ids(cal),test_records=ids(test)),folder/'final.pt')
    torch.save(dict(logits=z,labels=y.cpu(),rows=rows,test_indices=test,calibration_indices=cal),folder/'predictions.pt')
    result=dict(room=room,seed=seed,variant=variant,selected_epoch=best['epoch'],source_calibration_nll=best['nll'],parameters=sum(v.numel() for v in model.parameters()),initial_invariance_error=initial,final_invariance_error=error,checkpoint_sha256=sha(folder/'final.pt'),history=history,fit_records=ids(fitix),calibration_records=ids(cal),test_records=ids(test))
    evalix=test or cal;yy=y[evalix].cpu();pred=z[evalix].argmax(-1);result['metrics']=dict(correct=int((pred==yy).sum()),total=len(evalix),accuracy=float((pred==yy).float().mean()),confusion_matrix=torch.bincount(yy*5+pred,minlength=25).reshape(5,5).tolist())
    save(folder/'result.json',result);print('DONE',name,best['epoch'],result['metrics'],flush=True)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    WORK.mkdir(parents=True,exist_ok=True);rows=read(PREV/'rows.json');y=torch.tensor([int(r['headcount']) for r in rows],device='cuda')
    t=torch.from_numpy(np.array(np.load(PREV/'t.npy',mmap_mode='r'))).cuda();s=torch.from_numpy(np.array(np.load(PREV/'s.npy',mmap_mode='r'))).cuda()
    save(WORK/'training_config.json',dict(architecture='shared receiver and phase-pair multiscale encoders; invariant set pooling; bounded background modulation; presence times conditional count; auxiliary ordinal supervision',variants=VARIANTS,seeds=SEEDS,epochs=EPOCHS,scheduler_horizon_always=EPOCHS,batch_windows=24,lr=.0006,weight_decay=.02,loss='NLL + .2 ordinalBCE + .05 timeJS + optional .03 crossroom contrast',selection='Each fit selects checkpoint by source calibration balancedNLL only. Outer rooms reused development. Choose family by pooled mean correct across both seeds; final epochs median source-selected epochs. No target-label training or old teacher.',data_sha256=dict(t=sha(PREV/'t.npy'),s=sha(PREV/'s.npy'),rows=sha(PREV/'rows.json'))))
    if '--control' in sys.argv:
        fit('actual',2026,'control',rows,t,s,y,epochs=3,full=True)
    elif '--long-control' in sys.argv:
        fit('actual',2026,'control_long',rows,t,s,y,epochs=20,full=True)
    elif '--matched-control' in sys.argv:
        fit('actual',2026,'control_matched',rows,t,s,y,epochs=read(WORK/'selection.json')['epochs'],full=True)
    elif '--full' in sys.argv:
        selected=read(WORK/'selection.json')
        for seed in SEEDS:fit('actual',seed,selected['variant'],rows,t,s,y,epochs=selected['epochs'],full=True)
    else:
        for variant in VARIANTS:
            for room in ROOMS:
                for seed in SEEDS:fit(room,seed,variant,rows,t,s,y)

if __name__=='__main__':main()
