"""Label-conditional nuisance directions, class-centroid protection, label-free references."""
from run import *
evspec=importlib.util.spec_from_file_location('environment_eval_shared',ROOT/'experiments/environment_protocol_20261025/evaluate.py')
ev=importlib.util.module_from_spec(evspec);evspec.loader.exec_module(ev)
bootstrap=ev.bootstrap

class Nuisance:
    def __init__(self,d,fit,rank):
        e=d['e'];x=e.mean((1,2));self.mu=x[fit].mean(0);self.std=x[fit].std(0).clamp_min(.05)
        w=(x-self.mu)/self.std;rows=d['rows'];y=d['labels'];bases=[];checks=[]
        for route in range(4):
            centroids=[];residuals=[]
            for label in range(5):
                groups={}
                for i in fit:
                    if int(y[i])!=label:continue
                    room=rows[i]['room'] if rows[i]['room']!='unknown' else rows[i]['identifier']
                    groups.setdefault(room,[]).append(i)
                means=[w[ix,route].mean(0) for ix in groups.values()]
                if not means:continue
                center=torch.stack(means).mean(0);centroids.append(center)
                if len(means)>1:residuals.extend(m-center for m in means)
            centers=torch.stack(centroids);centered=centers-centers.mean(0)
            _,sv,v=torch.linalg.svd(centered,full_matrices=False);labelbasis=v[sv>1e-5]
            residual=torch.stack(residuals);residual=residual-(residual@labelbasis.T)@labelbasis
            _,singular,v=torch.linalg.svd(residual,full_matrices=False)
            basis=v[:min(rank,int((singular>1e-5).sum()))]
            if len(basis)<rank:basis=torch.cat([basis,torch.zeros(rank-len(basis),256)])
            checks.append(float((basis@labelbasis.T).abs().max()) if labelbasis.numel() else 0.)
            bases.append(basis)
        self.basis=torch.stack(bases);self.max_class_centroid_overlap=max(checks)
        assert self.max_class_centroid_overlap<1e-4
        # Localized reference gating uses source within-record variation, no target labels.
        coords=self.coords(e)
        dev=[]
        for record in sorted({rows[i]['identifier'] for i in fit}):
            ix=[i for i in fit if rows[i]['identifier']==record];q=coords[ix];dev.append((q-q.mean(0)).square().mean((1,2)))
        self.radius2=float(torch.cat(dev).median().clamp_min(.25))*4
    def coords(self,e):
        w=(e.mean((1,2))-self.mu)/self.std
        return torch.einsum('nrd,rkd->nrk',w,self.basis)
    def apply(self,e,alpha,reference=None):
        w=(e-self.mu[None,None,None])/self.std[None,None,None]
        if reference is None:
            component=torch.einsum('nvtrd,rkd->nvtrk',w,self.basis)
            correction=torch.einsum('nvtrk,rkd->nvtrd',component,self.basis)
        else:
            q=self.coords(e);r=reference
            dist=(q-r[None]).square().mean((1,2));strength=torch.exp(-dist/self.radius2)
            v=torch.einsum('rk,rkd->rd',r,self.basis)
            correction=strength[:,None,None,None,None]*v[None,None,None]
        return e-alpha*correction*self.std[None,None,None]

def logits(model,e,s,b,T):
    output=[]
    with torch.no_grad():
        for start in range(0,len(e),64):
            raw=shared.decode(model,e[start:start+64].cuda().flatten(0,2),s[start:start+64].cuda().flatten(0,2),(1.,1.,1.,1.)).cpu().reshape(-1,6,2,5)
            output.append(old.policy(raw,b,T))
    return torch.cat(output)

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(WORK/'subspace_plan.json',dict(stage=2,reason='Confidence-based targetselftraining supplied almost no reliable labels; use samecount/differentroom source residuals.',
       ranks=[2,4],strengths=[0,.25,.5],selection='Source calibration balancedNLL selects strength; target labels evaluation only. No percentile/threshold sweep.',
       controls=['Projection with no targetU','Soft-localized reference from targetU records, no roomID passed to predictor'],
       constraints='Nuisance basis orthogonal to sourceclasscentroid span, identity alpha0 always available. Learned source directions not proof of pure room background.',
       reused_development=True))
    pro=read(WORK/'protocols.json');results=[];chosen=[]
    for rank in [2,4]:
        allrecords={kind:[] for kind in ['projection','reference']};rooms={kind:{} for kind in allrecords}
        for room in ROOMS:
            d=torch.load(EVID/f'{room}.pt',weights_only=True);fit=pro[room]['source_fit'];cal=pro[room]['source_calibration']
            model,_=old.build(old.checkpoint(room));model.requires_grad_(False).cuda().eval();b,T=old.calibration(room)
            base=old.policy(d['raw'],b,T);adapter=Nuisance(d,fit,rank)
            loss=[]
            yy=d['labels'][cal];ww=1/torch.bincount(yy,minlength=5).clamp_min(1)[yy].float();ww/=ww.sum()
            for alpha in [0,.25,.5]:
                z=logits(model,adapter.apply(d['e'][cal],alpha),d['s'][cal],b,T)
                loss.append((float((F.cross_entropy(z,yy,reduction='none')*ww).sum()),alpha))
            _,alpha=min(loss)
            chosen.append(dict(room=room,rank=rank,strength=alpha,calibration=loss,max_centroid_overlap=adapter.max_class_centroid_overlap))
            source=logits(model,adapter.apply(d['e'],alpha),d['s'],b,T);reference=base.clone()
            for ff in pro[room]['folds']:
                # Adaptation labels never accessed; only four-route input evidence.
                ref=adapter.coords(d['e'][ff['adapt_indices']]).mean(0)
                ix=ff['test_indices'];reference[ix]=logits(model,adapter.apply(d['e'][ix],alpha,ref),d['s'][ix],b,T)
            for kind,z in [('projection',source),('reference',reference)]:
                ix=old.split(room,d['rows'])[2];m=old.metrics(z,d['labels'],base,ix);rooms[kind][room]=dict(metrics=m,baseline=old.metrics(base,d['labels'],base,ix))
                for record in pro[room]['target_records']:
                    jj=[i for i in ix if d['rows'][i]['identifier']==record];y=d['labels'][jj]
                    cc=int((z[jj].argmax(-1)==y).sum());bc=int((base[jj].argmax(-1)==y).sum());sc=int((source[jj].argmax(-1)==y).sum())
                    allrecords[kind].append(dict(room=room,identifier=record,windows=len(jj),delta=cc-bc,matched_delta=cc-sc,correct=cc,baseline_correct=bc))
                torch.save(dict(logits=z,baseline=base,labels=d['labels'],indices=ix),WORK/f'subspace_{kind}_k{rank}_{room}.pt')
            torch.save(dict(mean=adapter.mu,std=adapter.std,basis=adapter.basis,radius2=adapter.radius2,alpha=alpha),WORK/f'subspace_{room}_k{rank}.pt')
            print('SUBSPACE',room,rank,'selectedstrength',alpha,'cal',loss,flush=True)
            del model,d;gc.collect();torch.cuda.empty_cache()
        for kind in allrecords:
            records=allrecords[kind];net=sum(r['delta'] for r in records)
            results.append(dict(kind=kind,rank=rank,rooms=rooms[kind],net=net,records=records,ci95=bootstrap(records),
               matched_source_net=sum(r['matched_delta'] for r in records),matched_source_ci95=bootstrap(records,'matched_delta')))
    save(WORK/'subspace_choices.json',chosen);save(WORK/'subspace_results.json',results)
    print(json.dumps([dict(kind=r['kind'],rank=r['rank'],net=r['net'],ci=r['ci95'],matched=r['matched_source_net']) for r in results],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
