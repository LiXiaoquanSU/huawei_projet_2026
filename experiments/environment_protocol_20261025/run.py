"""Record-isolated target-U adaptation vs zero-shot DG, frozen054 backbone."""
import csv,gc,hashlib,importlib.util,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('environment_subnet_shared',ROOT/'experiments/subnet_expectation_20261025/run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
old=shared.old
WORK=Path('D:/huawei2026-experiments/environment_protocol_20261025')
EVID=shared.WORK/'evidence';PREV=shared.PREV
ROOMS=('a1','c1','g1','e1');SEEDS=(42,2026);STEPS=(50,150)
save,read,sha=old.save,old.read,old.sha

def key(record):return hashlib.sha256(('env-protocol-v1:'+record).encode()).hexdigest()

def protocols(rows):
    out={}
    for room in ROOMS:
        ids=sorted({r['identifier'] for r in rows if r['room']==room},key=key)
        a,b=ids[::2],ids[1::2];assert a and b and set(a).isdisjoint(b)
        fit,cal,test=old.split(room,rows)
        assert set(test)=={i for i,r in enumerate(rows) if r['room']==room}
        out[room]=dict(source_fit=fit,source_calibration=cal,target_records=ids,
                       folds=[dict(adapt_records=a,test_records=b),dict(adapt_records=b,test_records=a)])
        for fold in out[room]['folds']:
            fold['adapt_indices']=[i for i,r in enumerate(rows) if r['identifier'] in fold['adapt_records']]
            fold['test_indices']=[i for i,r in enumerate(rows) if r['identifier'] in fold['test_records']]
            assert set(fold['adapt_indices']).isdisjoint(fold['test_indices'])
            assert all(rows[i]['room']!=room for i in fit+cal)
    return out

def weights(rows,y,ix):
    groups={}
    for i in ix:
        r=rows[i];g=r['identifier'] if r['room']=='unknown' else r['room']
        groups[int(y[i]),g]=groups.get((int(y[i]),g),0)+1
    domains={c:sum(label==c for label,g in groups) for c in range(5)}
    return torch.tensor([1/(groups[int(y[i]),rows[i]['identifier'] if rows[i]['room']=='unknown' else rows[i]['room']]*domains[int(y[i])]) for i in ix],dtype=torch.float32)

class Normalizer(nn.Module):
    """A few hundred trainable parameters; no class-bias parameter or room-ID input."""
    def __init__(self,e,s,fit,conditioned=True):
        super().__init__();self.conditioned=conditioned
        descriptor=e.mean((1,2))[:,0];x=descriptor[fit]
        mu=x.mean(0);_,_,v=torch.linalg.svd(x-mu,full_matrices=False);basis=v[:16].T
        pc=(x-mu)@basis;pcscale=pc.std(0).clamp_min(.05)
        sm=s.mean((1,2));summary=torch.cat([sm[:,:6],sm[:,12:17]],-1)
        q=summary[fit];self.register_buffer('static_mean',mu);self.register_buffer('basis',basis)
        self.register_buffer('pcscale',pcscale);self.register_buffer('summary_mean',q.mean(0))
        self.register_buffer('summary_std',q.std(0).clamp_min(.05))
        self.register_buffer('e_std',e[fit].flatten(0,2).std(0).clamp_min(.05))
        if conditioned:
            self.net=nn.Sequential(nn.Linear(27,16),nn.SiLU(),nn.Linear(16,8))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        else:self.coefficients=nn.Parameter(torch.zeros(8))
    def context(self,e,s):
        descriptor=e.mean((1,2))[:,0];sm=s.mean((1,2));q=torch.cat([sm[:,:6],sm[:,12:17]],-1)
        return torch.cat([((descriptor-self.static_mean)@self.basis)/self.pcscale,
                          (q-self.summary_mean)/self.summary_std],-1).clamp(-5,5)
    def forward(self,e,s):
        c=self.net(self.context(e,s)) if self.conditioned else self.coefficients[None].expand(len(e),-1)
        scale,shift=c.tanh().chunk(2,-1)
        return e*(1+.1*scale[:,None,None,:,None])+.1*shift[:,None,None,:,None]*self.e_std[None,None,None]

def raw_scores(model,normalizer,e,s):
    changed=normalizer(e,s)
    return shared.decode(model,changed.flatten(0,2),s.flatten(0,2),(1.,1.,1.,1.)).reshape(-1,6,2,5)

def kl(p,z):return (p*(p.clamp_min(1e-8).log()-z.log_softmax(-1))).sum(-1)

def supervised_loss(raw,y,b,T):
    final=old.policy(raw,b,T)
    loss=F.cross_entropy(final,y)
    view=raw/T+b+raw.new_tensor([0,.35,0,0,0])
    return .75*loss+.25*F.cross_entropy(view.flatten(0,2),y[:,None,None].expand(-1,6,2).flatten())

def score(model,norm,e,s,b,T,batch=64):
    out=[]
    with torch.no_grad():
        for start in range(0,len(e),batch):out.append(old.policy(raw_scores(model,norm,e[start:start+batch],s[start:start+batch]),b,T).cpu())
    return torch.cat(out)

def train(room,seed,method,fold_id,d,p,model,b,T):
    suffix='source' if fold_id is None else f'fold{fold_id}'
    path=WORK/'fits'/f'{room}_s{seed}_{method}_{suffix}.pt'
    if path.exists():return
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    fit=p['source_fit'];cal=p['source_calibration'];e=d['e'].cuda();s=d['s'].cuda();y=d['labels'].cuda()
    n=Normalizer(d['e'],d['s'],fit,method!='global').cuda();optimizer=torch.optim.AdamW(n.parameters(),lr=.003,weight_decay=.02)
    # Separate generators ensure identical labelled draws for every method/fold.
    gen=torch.Generator().manual_seed(seed+173)
    source_draws=torch.multinomial(weights(d['rows'],d['labels'],fit),16*max(STEPS),replacement=True,generator=gen).reshape(-1,16)
    source_indices=torch.tensor(fit)[source_draws].cuda();source_hash=hashlib.sha256(source_indices.cpu().numpy().tobytes()).hexdigest()
    target=None
    if fold_id is not None:
        ids=p['folds'][fold_id]['adapt_indices']
        # No target labels or row metadata enter this adaptation object.
        target={'e':e[ids], 's':s[ids], 'teacher':old.policy(d['raw'][ids].cuda(),b,T).softmax(-1)}
        teacher_views=(d['raw'][ids].cuda()/T+b).softmax(-1)
        winner=target['teacher'].argmax(-1)
        target['reliable']=(target['teacher'].amax(-1)>=.8)&((teacher_views.argmax(-1)==winner[:,None,None]).sum((1,2))>=10)
        assert set(target)=={'e','s','teacher','reliable'}
        target_gen=torch.Generator().manual_seed(seed+743)
        target_draws=torch.randint(len(ids),(max(STEPS),16),generator=target_gen).cuda()
    teacher=old.policy(d['raw'].cuda(),b,T).softmax(-1)
    checkpoints=[];history=[]
    for step in range(1,max(STEPS)+1):
        ix=source_indices[step-1];optimizer.zero_grad(set_to_none=True)
        raw=raw_scores(model,n,e[ix],s[ix]);z=old.policy(raw,b,T)
        loss=supervised_loss(raw,y[ix],b,T)+.5*kl(teacher[ix],z).mean()
        penalty=sum(v.square().mean() for v in n.parameters());loss=loss+.001*penalty
        unsup=raw.new_zeros(())
        if target is not None:
            j=target_draws[step-1];uraw=raw_scores(model,n,target['e'][j],target['s'][j]);uz=old.policy(uraw,b,T)
            pv=(uraw/T+b).softmax(-1);mean=pv.mean((1,2),keepdim=True)
            js=(pv*(pv.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(-1).mean()
            unsup=.05*js+.2*kl(target['teacher'][j],uz).mean()
            if method=='conditional_u':
                trust=target['reliable'][j].float();perview=-(target['teacher'][j,None,None]*F.log_softmax(uraw/T+b,-1)).sum(-1).mean((1,2))
                unsup=unsup+.1*(perview*trust).sum()/trust.sum().clamp_min(1)
            loss=loss+unsup
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),1);optimizer.step()
        if step in STEPS:
            zcal=score(model,n,e[cal],s[cal],b,T);yy=d['labels'][cal]
            w=1/torch.bincount(yy,minlength=5).clamp_min(1)[yy].float();w/=w.sum()
            nll=float((F.cross_entropy(zcal,yy,reduction='none')*w).sum())
            checkpoints.append(dict(step=step,calibration_balanced_nll=nll,state={k:v.detach().cpu().clone() for k,v in n.state_dict().items()}))
            history.append(dict(step=step,loss=float(loss.detach()),unsupervised_loss=float(unsup.detach()),source_calibration_nll=nll))
    best=min(checkpoints,key=lambda q:q['calibration_balanced_nll']);n.load_state_dict(best['state']);n.eval()
    logits=score(model,n,e,s,b,T)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=best['state'],selected_step=best['step'],history=history,method=method,seed=seed,room=room,fold=fold_id,
                    conditioned=method!='global',source_draws_sha256=source_hash,
                    source_fit_records=sorted({d['rows'][i]['identifier'] for i in fit}),
                    source_calibration_records=sorted({d['rows'][i]['identifier'] for i in cal}),
                    target_adaptation_records=p['folds'][fold_id]['adapt_records'] if fold_id is not None else [],
                    target_test_records=p['folds'][fold_id]['test_records'] if fold_id is not None else [],
                    reliable_target_windows=int(target['reliable'].sum()) if target is not None else 0,
                    logits=logits,trainable_parameters=sum(v.numel() for v in n.parameters())),path)
    print('ENV_FIT',room,seed,method,suffix,'step',best['step'],'cal',round(best['calibration_balanced_nll'],4),flush=True)
    del n,optimizer,e,s,y,target,teacher;gc.collect();torch.cuda.empty_cache()

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    WORK.mkdir(parents=True,exist_ok=True);rows=read(PREV/'rows.json');p=protocols(rows)
    save(WORK/'protocols.json',p)
    save(WORK/'plan.json',dict(goal_platform=.60,baseline_platform=.54,source_checkpoint_sha256=old.SOURCE_SHA,
      protocols=['Zero-shot targetroom absent from labels andU','Target-U record-disjoint adaptation then swapped role fold'],
      rooms=ROOMS,seeds=SEEDS,methods=['global','conditioned','view_u','conditional_u'],steps=STEPS,
      training='Frozen054encoder/readout; source-only PCA/contextnormalization; max10percent per-route affine; no classbias or test-roomID input.',
      selection='Source calibration NLL only chooses50/150steps. No targetlabels for fitting, pseudo-label selection or earlystopping. Paired target test records same for all methods.',
      caveats=['These known rooms are reuseddevelopment, not a fresh blind test.','b1 has one record, cannot form targetU/test split; unknown not pooled as one room.','Target-U half-records may miss occupancy classes, do not enforce uniform labels.'],
      final_candidate_gate='Require positive>=2rooms inDG andUDA vs original, no roomloss>2pp, mean>=2pp, whole-recordinterval positive. If no pass do not export.'))
    with (ROOT/'docs/ENVIRONMENT_PROTOCOL_SUPPORT_20261025.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['room','fold','role','records','windows','count0','count1','count2','count3','count4']);w.writeheader()
        for room,pp in p.items():
            for fold,ff in enumerate(pp['folds']):
                for role in ['adapt','test']:
                    ix=ff[role+'_indices'];labels=[int(rows[i]['headcount']) for i in ix]
                    w.writerow(dict(room=room,fold=fold,role=role,records=len(ff[role+'_records']),windows=len(ix),**{f'count{k}':labels.count(k) for k in range(5)}))
    # Reuse explicitly-labelled earlier raw audit, not incompatible FLOAT64 feature caches.
    previous=read(Path('D:/huawei2026-experiments/u_ssl_20261025/u_raw_manifest.json'))
    save(WORK/'official_u_audit_reused.json',dict(source='Previous raw-file audit, not a fresh scan; not reusing itsFLOAT64features.',records=len(previous),
      rooms={room:sum(r['room']==room for r in previous) for room in sorted({r['room'] for r in previous})},
      shapes=sorted({str(r['shape']) for r in previous}),active_fraction_range=[min(r['active_power_gt1e_8_fraction'] for r in previous),max(r['active_power_gt1e_8_fraction'] for r in previous)],
      fields_without_groundtruth=True))
    for room in ROOMS:
        started=time.perf_counter();d=torch.load(EVID/f'{room}.pt',weights_only=True)
        assert d['zero_error']<3e-5
        model,_=old.build(old.checkpoint(room));model.requires_grad_(False).cuda().eval();b,T=old.calibration(room);b=b.cuda()
        for seed in SEEDS:
            for method in ['global','conditioned']:train(room,seed,method,None,d,p[room],model,b,T)
            for fold in [0,1]:
                for method in ['view_u','conditional_u']:train(room,seed,method,fold,d,p[room],model,b,T)
        print('ENV_ROOM_DONE',room,round(time.perf_counter()-started,1),flush=True)
        del model,d;gc.collect();torch.cuda.empty_cache()
    save(WORK/'training_complete.json',dict(fits=48,source='Fixed methods/seeds/budgets; targetlabels not used to select checkpoints.'))

if __name__=='__main__':main()
