"""Identity-initialized temporal adapter; record/room-separated proxy controls."""
import copy,json,sys,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tta_policy_20261024'))
from common import build,permutation_indices,sha
from model import ordered_probabilities
WORK=Path('D:/huawei2026-experiments/adapter049_20261024')
OLD=Path('D:/huawei2026-experiments/protected_context_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
REV=OLD/'reverse_features.pt'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

class AdapterHead(nn.Module):
    def __init__(self,model):
        super().__init__()
        for name in ('context','gate','experts','shared','strength','classifier','ordinal','presence'):
            module=copy.deepcopy(getattr(model,name)).eval()
            module.requires_grad_(False);setattr(self,name,module)
        self.down=nn.Linear(256,32);self.up=nn.Linear(32,256)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,e,s,scale=1.):
        delta=self.up(F.silu(self.down(e[:,3])))*scale
        evidence=[e[:,0],e[:,1],e[:,2],e[:,3]+delta]
        content=torch.cat(evidence,1)
        context=self.context(torch.cat([s[:,:6],s[:,12:17]],1))
        gate=.5*self.gate(torch.cat([content,context],1)).softmax(-1)+.125
        experts=torch.stack([h(z) for h,z in zip(self.experts,evidence)],1)
        mixture=(experts.softmax(-1)*gate[:,:,None]).sum(1)
        strength=torch.cat([s[:,6:12],s[:,12:17]],1)
        z=self.shared(content)+self.strength(strength)
        direct=.5*self.classifier(z).softmax(-1)+.5*mixture
        ordinal=self.ordinal(z)
        presence=self.presence(torch.cat([*evidence[1:],strength],1))
        ordered,_=ordered_probabilities(torch.cat([presence,ordinal[:,1:]],1))
        return ((direct+ordered)*.5).clamp_min(1e-8).log(),delta

def policy(raw,b,T=1.):
    original=raw[:,0]/T+b
    q=raw.exp().reshape(-1,2,2,5).mean(2)
    identity=q[:,0].clamp_min(1e-12).log()/T+b
    average=q.mean(1).clamp_min(1e-12).log()/T+b
    label=identity.argmax(-1)
    legacy=torch.where(((label==0)|(label==2))[:,None],identity,average)
    chosen=torch.where((original.argmax(-1)==1)[:,None],original,legacy)
    return chosen+chosen.new_tensor([0,.35,0,0,0])

def prepare():
    if all((WORK/f'{r}_latents.pt').exists() for r in ('a1','c1','g1','e1')):return
    d=torch.load(CACHE,weights_only=True);rev=torch.load(REV,weights_only=True)
    assert d['rows']==rev['rows']
    idx,sign,si=[x.cuda() for x in permutation_indices((2,1,0))]
    for room in ('a1','c1','g1','e1'):
        path=WORK/f'{room}_latents.pt'
        if path.exists():continue
        checkpoint=OLD/'fits'/f'baseline_s42_{room}'/'final.pt'
        manifest=read(OLD/'results'/f's42_{room}'/'manifest.json')
        cp=torch.load(checkpoint,weights_only=True)
        assert set(cp['fit_identifiers'])=={r['identifier'] for r in manifest['fit']}
        assert set(cp['fit_identifiers']).isdisjoint({r['identifier'] for r in manifest['test']+manifest['calibration']})
        m,b=build(checkpoint);m=m.cuda().eval();captures={};hooks=[]
        def hook(name):
            def fn(module,args,out):captures[name]=out.detach()
            return fn
        routes=('static','dynamic','phase','temporal')
        for name in routes:hooks.append(getattr(m,name).register_forward_hook(hook(name)))
        ee=[];ss=[];zz=[]
        with torch.inference_mode():
            for start in range(0,960,8):
                t,s=d['t'][start:start+8].cuda(),d['s'][start:start+8].cuda()
                rt,rs=rev['t'][start:start+8].cuda(),rev['s'][start:start+8].cuda()
                views=[(t,s),(rt,rs),(t.index_select(1,idx)*sign,s.index_select(1,si)),(rt.index_select(1,idx)*sign,rs.index_select(1,si))]
                ev=[];sv=[];zv=[]
                for tt,st in views:
                    z=m.forward_tasks(tt,st)['logits'];ev.append(torch.stack([captures[n] for n in routes],1).cpu());sv.append(st.cpu());zv.append(z.cpu())
                ee.append(torch.stack(ev,1));ss.append(torch.stack(sv,1));zz.append(torch.stack(zv,1))
        for h in hooks:h.remove()
        torch.save(dict(latents=torch.cat(ee),summaries=torch.cat(ss),logits=torch.cat(zz),labels=d['y'],rows=d['rows'],checkpoint_sha256=sha(checkpoint)),path)
        print('PREPARED',room,flush=True)
        del m,cp;captures.clear();gc.collect();torch.cuda.empty_cache()

def weights(rows,y,indices):
    groups={}
    for i in indices:
        key=int(y[i]),rows[i]['room'];groups[key]=groups.get(key,0)+1
    domains={k:len({r for c,r in groups if c==k}) for k in range(5)}
    w=torch.tensor([1/(groups[int(y[i]),rows[i]['room']]*domains[int(y[i])]) for i in indices],device='cuda')
    return w/w.sum()

def metrics(z,y,base,indices):
    ix=torch.tensor(indices,device=z.device);p=z[ix].argmax(-1);truth=y[ix];bp=base[ix].argmax(-1)
    cm=torch.bincount(truth*5+p,minlength=25).reshape(5,5)
    return dict(correct=int((p==truth).sum()),total=len(ix),corrected=int(((p==truth)&(bp!=truth)).sum()),broken=int(((p!=truth)&(bp==truth)).sum()),confusion_matrix=cm.cpu().tolist())

def fit(cache,checkpoint,fit_ids,cal_ids,method,room,b,T,steps=150):
    torch.manual_seed(2026)
    model,_=build(checkpoint);head=AdapterHead(model).cuda().eval();del model
    e=cache['latents'].cuda();s=cache['summaries'].cuda();raw=cache['logits'].cuda();y=cache['labels'].cuda();rows=cache['rows']
    b=b.cuda();base=policy(raw,b,T)
    with torch.no_grad():
        check,_=head(e.flatten(0,1),s.flatten(0,1));err=float((check.reshape(-1,4,5).exp()-raw.exp()).abs().max());assert err<2e-6,err
    fit_ix=[i for i,r in enumerate(rows) if r['identifier'] in fit_ids]
    cal_ix=[i for i,r in enumerate(rows) if r['identifier'] in cal_ids]
    ix=torch.tensor(fit_ix,device='cuda');w=weights(rows,y,fit_ix)
    ce=nn.CrossEntropyLoss(reduction='none',label_smoothing=.02)
    optim=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    results=[];snapshots={}
    for step in range(1,steps+1):
        z,delta=head(e[ix].flatten(0,1),s[ix].flatten(0,1));z=z.reshape(-1,4,5)
        target=y[ix,None].expand(-1,4)
        supervised=(ce(z.flatten(0,1),target.flatten()).reshape(-1,4).mean(1)*w).sum()
        # Frozen teacher retention, identity-seeking residual norm, optional view consistency.
        teacher=raw[ix].exp();kl=(teacher*(raw[ix]-z)).sum(-1).mean(1)
        if method.startswith('calibrated'):
            final=policy(z,b,T)
            calibrated_views=(z/T+b+z.new_tensor([0,.35,0,0,0]))
            perview=ce(calibrated_views.flatten(0,1),target.flatten()).reshape(-1,4).mean(1)
            supervised=(.75*ce(final,y[ix])+.25*perview).mul(w).sum()
            if method=='calibrated_dro':
                sampleloss=.75*ce(final,y[ix])+.25*perview
                groupmeans=[]
                for label,domain in sorted({(int(cache['labels'][i]),rows[i]['room']) for i in fit_ix}):
                    mask=torch.tensor([int(cache['labels'][i])==label and rows[i]['room']==domain for i in fit_ix],device='cuda')
                    groupmeans.append(sampleloss[mask].mean())
                groupmeans=torch.stack(groupmeans)
                robust=torch.logsumexp(groupmeans,0)-np.log(len(groupmeans))
                supervised=.5*supervised+.5*robust
        loss=supervised+.2*(kl*w).sum()+.01*delta.square().mean()
        if method in ('consistent','aligned','calibrated_consistent','calibrated_aligned','calibrated_dro'):
            prob=z.exp();mean=prob.mean(1,keepdim=True)
            js=(prob*(z-mean.clamp_min(1e-8).log())).sum(-1).mean(1)
            loss=loss+.2*(js*w).sum()
        if method in ('aligned','calibrated_aligned'):
            # Same-class only: never force differently occupied rooms together.
            adapted=e[ix,:,3]+delta.reshape(-1,4,256);terms=[]
            for label in (0,1):
                centers=[]
                for domain in sorted({rows[i]['room'] for i in fit_ix}):
                    mask=torch.tensor([int(y[i])==label and rows[i]['room']==domain for i in fit_ix],device='cuda')
                    if mask.any():centers.append(F.normalize(adapted[mask].mean((0,1)),dim=0))
                if len(centers)>1:terms.append(torch.stack(centers).var(0,unbiased=False).sum())
            if terms:loss=loss+(.2 if method=='calibrated_aligned' else .05)*torch.stack(terms).mean()
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad],1.);optim.step()
        if step in (25,75,150):
            with torch.no_grad():
                zz,_=head(e.flatten(0,1),s.flatten(0,1));zz=zz.reshape(-1,4,5)
                score=policy(zz,b,T)
                ci=torch.tensor(cal_ix,device='cuda') if cal_ix else ix
                cw=weights(rows,y,cal_ix or fit_ix)
                cal_loss=float((F.cross_entropy(score[ci],y[ci],reduction='none')*cw).sum())
                results.append(dict(step=step,calibration_nll=cal_loss,fit=metrics(score,y,base,fit_ix),calibration=metrics(score,y,base,cal_ix or fit_ix)))
                snapshots[step]={k:v.detach().cpu().clone() for k,v in head.state_dict().items() if k.startswith(('down.','up.'))}
    chosen=min(results,key=lambda r:r['calibration_nll'])['step']
    return head,snapshots,results,chosen,base

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(methods=['supervised','consistent','aligned'],steps=[25,75,150],seed=2026,lr=.001,adapter_parameters=16672,
      frozen='All original encoder and head weights in eval; only zero-initialized temporal adapter trained',
      split='Proxy adapter fits use checkpoint fit records only. Epoch selected by held-out whole calibration records. Outer rooms used for reporting, never adapter fitting.',
      caveat='Repeated historical development on these proxies; not untouched final validation. Actual encoder has seen every source label.',
      fixed_bias='Original .44 prior calibration and .49 final +.35 unchanged. No bias search.'))
    prepare();reports=[]
    for room in ('a1','c1','g1','e1'):
        cache=torch.load(WORK/f'{room}_latents.pt',weights_only=True)
        manifest=read(OLD/'results'/f's42_{room}'/'manifest.json')
        ids={k:{r['identifier'] for r in manifest[k]} for k in ('fit','calibration','test')}
        cal=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline']
        for method in ('supervised','consistent','aligned'):
            checkpoint=OLD/'fits'/f'baseline_s42_{room}'/'final.pt'
            head,states,history,chosen,base=fit(cache,checkpoint,ids['fit'],ids['calibration'],method,room,torch.tensor(cal['bias']),cal['temperature'])
            head.load_state_dict(states[chosen],strict=False)
            with torch.no_grad():
                z,_=head(cache['latents'].cuda().flatten(0,1),cache['summaries'].cuda().flatten(0,1))
                z=policy(z.reshape(-1,4,5),torch.tensor(cal['bias'],device='cuda'),cal['temperature'])
                test_ix=[i for i,r in enumerate(cache['rows']) if r['identifier'] in ids['test']]
                report=dict(room=room,method=method,chosen_step=chosen,history=history,test=metrics(z,cache['labels'].cuda(),base,test_ix))
            save(WORK/'folds'/f'{room}_{method}.json',report);reports.append(report)
            print('FOLD',room,method,chosen,report['test'],flush=True)
            del head;gc.collect();torch.cuda.empty_cache()
    save(WORK/'proxy_results.json',reports)
    cache=torch.load(Path('D:/huawei2026-experiments/evidence049_research_20261024/frozen_latents.pt'),weights_only=True)
    checkpoint=ROOT/'models/best_037/final.pt';_,b=build(checkpoint)
    full=[]
    for method in ('supervised','consistent','aligned'):
        ids={r['identifier'] for r in cache['rows']}
        head,states,history,_,base=fit(cache,checkpoint,ids,set(),method,'actual',b,1.)
        # Epoch fixed from proxy calibration choices, no actual-source peak chasing.
        step=int(np.median([r['chosen_step'] for r in reports if r['method']==method]));step=min(states,key=lambda k:abs(k-step))
        head.load_state_dict(states[step],strict=False)
        for scale in (.25,.5,1.):
            with torch.no_grad():
                raw,_=head(cache['latents'].cuda().flatten(0,1),cache['summaries'].cuda().flatten(0,1),scale)
                raw=raw.reshape(-1,4,5);z=policy(raw,b.cuda())
                report=dict(method=method,step=step,scale=scale,source_diagnostic=metrics(z,cache['labels'].cuda(),base,list(range(960))))
            full.append(report);print('ACTUAL',report,flush=True)
            torch.save(dict(adapter=states[step],method=method,step=step,scale=scale,fit_identifiers=sorted(ids),history=history),WORK/f'{method}_{scale}.pt')
            np.savez_compressed(WORK/f'{method}_{scale}_predictions.npz',logits=z.cpu().numpy(),raw=raw.cpu().numpy(),labels=cache['labels'].numpy())
        del head;gc.collect();torch.cuda.empty_cache()
    save(WORK/'actual_results.json',full)

if __name__=='__main__':main()
