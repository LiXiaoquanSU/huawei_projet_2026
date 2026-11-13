"""Cache12 frozen views, test six fixed pools, train two bounded reliability heads."""
import gc,itertools,time
from g60_common import *

def prepare(room):
    path=WORK/f'{room}.pt'
    if path.exists():
        d=torch.load(path,weights_only=True);assert d['checkpoint_sha256']==sha(checkpoint(room));return d
    d=torch.load(CACHE,weights_only=True);r=torch.load(OLD/'reverse_features.pt',weights_only=True);assert d['rows']==r['rows']
    model,b=build(checkpoint(room),full=room=='actual');model.cuda().eval();raw=[];start=time.perf_counter()
    if room!='actual':
        cp=torch.load(checkpoint(room),weights_only=True);fi,_,_=split(room,d['rows']);assert set(cp['fit_identifiers'])=={d['rows'][i]['identifier'] for i in fi};del cp
    with torch.inference_mode():
        for permutation in PERMS:
            idx,sign,si=[x.cuda() for x in permutation_indices(permutation)];pred=[]
            for offset in range(0,960,16):
                t=torch.stack([d['t'][offset:offset+16],r['t'][offset:offset+16]],1).flatten(0,1).cuda();s=torch.stack([d['s'][offset:offset+16],r['s'][offset:offset+16]],1).flatten(0,1).cuda()
                z=model.forward_features(t.index_select(1,idx)*sign,s.index_select(1,si));pred.append(z.cpu().reshape(-1,2,5))
            raw.append(torch.cat(pred));print('G60_VIEWS',room,permutation,round(time.perf_counter()-start,1),flush=True)
    out=dict(raw=torch.stack(raw,1),labels=d['y'],rows=d['rows'],checkpoint_sha256=sha(checkpoint(room)))
    torch.save(out,path);del model,d,r;gc.collect();torch.cuda.empty_cache();return out

def train_gate(room,d,b,T,seed):
    path=WORK/'fits'/f'{room}_s{seed}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    fi,ci,ti=split(room,d['rows']);torch.manual_seed(seed);g=Reliability();raw=d['raw'];y=d['labels'];base=policy(raw,b,T)
    assert torch.max((policy(raw,b,T,'learned',[g])-base).abs())<1e-5
    opt=torch.optim.AdamW(g.parameters(),lr=.003,weight_decay=.01);wi=balance(d['rows'],y,fi);ix=torch.tensor(fi);history=[];states={}
    for step in range(1,401):
        z=policy(raw,b,T,'learned',[g]);ce=(F.cross_entropy(z[ix],y[ix],reduction='none',label_smoothing=.02)*wi).sum()
        teacher=base[ix].softmax(-1);kl=(teacher*(teacher.clamp_min(1e-8).log()-z[ix].log_softmax(-1))).sum(-1).mean()
        w=g(features(raw,b,T));loss=ce+.1*kl+.02*(w*6-1).square().mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(g.parameters(),1.);opt.step()
        if step in (50,150,400):
            states[step]={k:v.detach().clone() for k,v in g.state_dict().items()};cix=torch.tensor(ci or fi);cw=balance(d['rows'],y,ci or fi)
            with torch.no_grad():z=policy(raw,b,T,'learned',[g]);nll=float((F.cross_entropy(z[cix],y[cix],reduction='none')*cw).sum())
            history.append(dict(step=step,calibration_nll=nll))
    if room=='actual':
        steps=[torch.load(WORK/'fits'/f'{r}_s{seed}.pt',weights_only=True)['chosen_step'] for r in ('a1','c1','g1','e1')];median=float(np.median(steps));chosen=min(states,key=lambda k:abs(k-median))
    else:chosen=min(history,key=lambda x:x['calibration_nll'])['step']
    out=dict(state=states[chosen],chosen_step=chosen,history=history,seed=seed,parameters=sum(p.numel() for p in g.parameters()),fit_identifiers=sorted({d['rows'][i]['identifier'] for i in fi}),checkpoint_sha256=d['checkpoint_sha256'])
    path.parent.mkdir(exist_ok=True);torch.save(out,path);return out

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(target=.60,baseline_sha256=SOURCE_SHA,
      methods=['baseline','normalized','power075','trimmed25','timegeom25','time_js','learned'],
      controls='4whole-room teachers exclude their test/calibration records; unchanged own calibration bias/temperature. Source960 diagnosed separately; prior room reuse disclosed.',
      learned='two seeds42/2026,177parameter shared class-agnostic9->16->1 reliability, bounded weights maxratioexp1; no class/order/room identity, no output bias. Frozen backbone; fit labels only;calibration selects50/150/400.',
      selection='Need >=2 improving proxy rooms, no room worse than2pp, positive aggregate and no source class deterioration. Prefer cross-seed consistency, no guarantee to060. If none eligible record failure rather than claim improvement.',
      papers=['https://arxiv.org/abs/2011.11156','https://arxiv.org/abs/2110.09506']))
    results=[]
    for room in ('a1','c1','g1','e1','actual'):
        d=prepare(room);b,T=calibration(room);raw=d['raw'];y=d['labels'];fi,ci,ti=split(room,d['rows']);base=policy(raw,b,T)
        v0=raw[:,0,0]/T+b;v45=raw[:,0].exp().mean(1).log()/T+b
        branch=torch.where(v0.argmax(-1)==1,0,torch.where((v45.argmax(-1)==0)|(v45.argmax(-1)==2),1,2))
        anyview=(((raw/T+b+raw.new_tensor([0,.35,0,0,0])).argmax(-1))==y[:,None,None]).flatten(1).any(-1)
        diagnostic=[]
        for label,name in enumerate(('original_one','identity_time_02','fallback')):
            ix=torch.tensor(ti);ix=ix[branch[ix]==label]
            diagnostic.append(dict(branch=name,total=len(ix),correct=int((base[ix].argmax(-1)==y[ix]).sum()),wrong_with_at_least_one_correct_view=int(((base[ix].argmax(-1)!=y[ix])&anyview[ix]).sum())))
        save(WORK/f'{room}_branch_diagnostic.json',dict(branches=diagnostic,oracle_warning='Any-correct-view uses ground truth and is only a non-deployable diagnostic ceiling, not an achievable accuracy.',errors=[dict(identifier=d['rows'][i]['identifier'],start=d['rows'][i]['start'],truth=int(y[i]),prediction=int(base[i].argmax()),branch=int(branch[i]),any_correct_view=bool(anyview[i])) for i in ti if base[i].argmax()!=y[i]]))
        gates=[]
        for seed in (42,2026):
            fit=train_gate(room,d,b,T,seed);g=Reliability();g.load_state_dict(fit['state']);gates.append(g)
        variants=[(m,None) for m in ('baseline','normalized','power075','trimmed25','timegeom25','time_js')]+[('learned42',[gates[0]]),('learned2026',[gates[1]]),('learned',gates)]
        for name,gg in variants:
            with torch.no_grad():z=policy(raw,b,T,'learned' if gg else name,gg)
            result=dict(room=room,method=name,metrics=metrics(z,y,base,ti));results.append(result);np.savez_compressed(WORK/f'{room}_{name}.npz',logits=z.numpy(),base=base.numpy(),labels=y.numpy(),test_indices=ti)
            print('G60_RESULT',room,name,result['metrics']['corrected'],result['metrics']['broken'],flush=True)
        save(WORK/'results.json',results)
    summary=[]
    for method in [x[0] for x in variants]:
        rr=[r for r in results if r['room']!='actual' and r['method']==method];source=next(r for r in results if r['room']=='actual' and r['method']==method)['metrics'];baseline=next(r for r in results if r['room']=='actual' and r['method']=='baseline')['metrics']
        roomnet={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in rr};net=sum(roomnet.values());worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in rr)
        eligible=net>0 and sum(v>0 for v in roomnet.values())>=2 and worst>=-.02 and all(source['confusion_matrix'][k][k]>=baseline['confusion_matrix'][k][k] for k in range(5))
        summary.append(dict(method=method,room_net=roomnet,proxy_net=net,source=source,worst_room_delta=worst,eligible=eligible))
    save(WORK/'summary.json',summary);print('G60_SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()
