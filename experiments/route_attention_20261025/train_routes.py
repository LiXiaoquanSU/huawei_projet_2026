"""Predeclared frozen-backbone route residual controls; room-disjoint development."""
import time,copy
from ra_common import *

def run(room,architecture,objective,seed):
    dest=WORK/'fits'/f'{architecture}_{objective}_{seed}_{room}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    d=torch.load(WORK/f'{room}.pt',weights_only=True);fit,cal,test=old.split(room,d['rows'])
    x,base,aux,y,e=[d[k].cuda() for k in ('stats','base','aux','labels','e')]
    fi=torch.tensor(fit,device='cuda');ci=torch.tensor(cal,device='cuda')
    groups=sorted({(int(y[i]),d['rows'][i]['room']) for i in fit})
    gi=torch.tensor([groups.index((int(y[i]),d['rows'][i]['room'])) for i in fit],device='cuda')
    prior=torch.tensor([1/sum(c==c2 for c2,r2 in groups) for c,r in groups],device='cuda');prior=prior/prior.sum()
    ng=torch.bincount(gi,minlength=len(groups));logq=torch.zeros_like(prior)
    region=torch.where(base[fi].argmax(-1)<2,1.,.25)
    m=RouteResidual(architecture).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.05)
    history=[];best=(float('inf'),None);saved={};started=time.perf_counter()
    for step in range(1,401):
        m.train();delta=m(x[fi],base[fi],aux[fi]);z=base[fi]+.5*delta
        ce=F.cross_entropy(z,y[fi],reduction='none',label_smoothing=.03)
        losses=(ce*region).new_zeros(len(groups)).scatter_add_(0,gi,ce*region)/ng
        if objective=='dro':
            with torch.no_grad():logq.add_(.01*losses.detach());logq.sub_(logq.max());logq.clamp_(-8,0)
            gw=(prior*logq.exp());gw=gw/gw.sum()
        else:gw=prior
        # View subsampling regularizes evidence aggregation, not time/label augmentation.
        vi=torch.randperm(12,device='cuda')[:8];aug=m(moments(e[fi][:,vi]),base[fi],aux[fi])
        kl=F.kl_div(z.log_softmax(-1),base[fi].softmax(-1),reduction='batchmean')
        loss=(gw*losses).sum()+.1*kl+.02*delta.square().mean()+.02*(delta-aug).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if step in (50,150,400):
            m.eval()
            with torch.no_grad():all_delta=m(x,base,aux);z=deploy(base,all_delta,.5)
            if cal:
                w=old.balance(d['rows'],d['labels'],cal).cuda();nll=float((F.cross_entropy(z[ci],y[ci],reduction='none')*w).sum())
            else:nll=None
            state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            cp=dict(architecture=architecture,objective=objective,seed=seed,room=room,step=step,state=state,fit_identifiers=sorted({d['rows'][i]['identifier'] for i in fit}),calibration_identifiers=sorted({d['rows'][i]['identifier'] for i in cal}),teacher_sha256=d['checkpoint_sha256'])
            torch.save(cp,dest/f'step{step}.pt');saved[step]=all_delta.cpu();history.append(dict(step=step,calibration_nll=nll,loss=float(loss)))
            if nll is not None and nll<best[0]:best=(nll,step)
    torch.save(saved,dest/'deltas.pt');save(dest/'complete.json',dict(history=history,selected_step=best[1],seconds=time.perf_counter()-started,fit_windows=len(fit),calibration_windows=len(cal),test_windows=len(test)))
    print('RA_FIT',room,architecture,objective,seed,best[1],round(time.perf_counter()-started,1),flush=True)

def evaluate():
    results=[]
    for arch in ('pool','attention'):
      for obj in ('balanced','dro'):
        steps={s:[read(WORK/'fits'/f'{arch}_{obj}_{s}_{r}'/'complete.json')['selected_step'] for r in ('a1','c1','g1','e1')] for s in (42,2026)}
        for seed in (42,2026):
            median=float(np.median(steps[seed]));chosen=min((50,150,400),key=lambda k:abs(k-median))
            p=WORK/'fits'/f'{arch}_{obj}_{seed}_actual'/'complete.json';v=read(p);v['selected_step']=chosen;v['proxy_steps']=steps[seed];save(p,v)
        for seeds in ((42,),(2026,),(42,2026)):
          for amount in (.25,.5,1.):
            item=dict(architecture=arch,objective=obj,seeds=list(seeds),amount=amount,rooms={});net=0;improve=0;worst=1
            for room in ('a1','c1','g1','e1','actual'):
                d=torch.load(WORK/f'{room}.pt',weights_only=True);delta=[]
                for seed in seeds:
                    dest=WORK/'fits'/f'{arch}_{obj}_{seed}_{room}';step=read(dest/'complete.json')['selected_step'];delta.append(torch.load(dest/'deltas.pt',weights_only=True)[step])
                z=deploy(d['base'],torch.stack(delta).mean(0),amount);ix=old.split(room,d['rows'])[2];m=old.metrics(z,d['labels'],d['base'],ix);item['rooms'][room]=m
                if room!='actual':
                    n=m['corrected']-m['broken'];net+=n;improve+=n>0;worst=min(worst,n/m['total'])
            item.update(proxy_net=net,improving_rooms=improve,worst_room_delta=worst,eligible=net>0 and improve>=2 and worst>=-.02);results.append(item)
    save(WORK/'results.json',results)
    for r in sorted(results,key=lambda v:(v['eligible'],v['proxy_net']),reverse=True)[:12]:print('RA_RESULT',r['architecture'],r['objective'],r['seeds'],r['amount'],r['proxy_net'],r['improving_rooms'],round(r['worst_room_delta'],3),r['rooms']['actual']['correct'],r['eligible'],flush=True)

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    for room in ('a1','c1','g1','e1','actual'):
      for arch in ('pool','attention'):
       for obj in ('balanced','dro'):
        for seed in (42,2026):run(room,arch,obj,seed)
    evaluate()
if __name__=='__main__':main()
