"""Controlled cross-seed evidence: isolated-room proxy fits before conditional fusion."""
import argparse,csv,gc,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
from model import StructuredModel,signal_targets
from utils import task_loss

WORK=Path('D:/huawei2026-experiments/seed_experts_20261025')
PREV=Path('D:/huawei2026-experiments/next053_20261025')
PACKAGE=ROOT/'2026-10-25_037配方与种子复现'
ROOMS=('a1','c1','g1','e1')
SEEDS=(42,310)
METHODS=('mean25','weak_consensus','low_consensus','fallback50','unanimous_rescue')
save,read,sha=old.save,old.read,old.sha

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def round_storage(model):
    keys=old.precision_spec()[0]
    with torch.no_grad():
        for k,v in model.state_dict().items():
            if k in keys:v.copy_(v.half().float())
    return model

def weights(rows,y,ix):
    groups={}
    for i in ix:
        r=rows[i];g=r['identifier'] if r['room']=='unknown' else r['room']
        groups[int(y[i]),g]=groups.get((int(y[i]),g),0)+1
    classes={c:len({g for label,g in groups if label==c}) for c in range(5)}
    return torch.tensor([1/(groups[int(y[i]),rows[i]['identifier'] if rows[i]['room']=='unknown' else rows[i]['room']]*classes[int(y[i])]) for i in ix],dtype=torch.double)

def fit(room,seed,rows,tt,ss,y):
    folder=WORK/f'fit_{room}_s{seed}';path=folder/'final.pt'
    if path.exists():return path
    fit_ix,cal_ix,test_ix=old.split(room,rows)
    assert not set(fit_ix)&set(cal_ix) and not set(fit_ix)&set(test_ix)
    assert all(rows[i]['room']!=room for i in fit_ix+cal_ix)
    seed_all(seed);model=StructuredModel('spectral','physical').cuda().train()
    t=torch.from_numpy(np.array(tt[fit_ix,0],copy=True));s=torch.from_numpy(np.array(ss[fit_ix,0],copy=True))
    yy=y[fit_ix];targets=signal_targets(t)
    draws=len(fit_ix)
    sampler=WeightedRandomSampler(weights(rows,y,fit_ix),draws,replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(TensorDataset(t,s,yy,targets),batch_size=32,sampler=sampler)
    opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01);history=[]
    for epoch in range(1,6):
        losses=[];correct=total=0
        for a,b,label,aux in loader:
            a,b,label,aux=a.cuda(),b.cuda(),label.cuda(),aux.cuda()
            opt.zero_grad(set_to_none=True);tasks=model.forward_tasks(a,b);loss,_=task_loss(tasks,label,aux,'physical')
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
            losses.append(float(loss.detach()));correct+=int((tasks['logits'].argmax(-1)==label).sum());total+=len(label)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),sampled_fit_accuracy=correct/total))
        print('FIT',room,seed,history[-1],flush=True)
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state=model.cpu().state_dict(),seed=seed,config=dict(epochs=5,size='spectral',mode='physical'),
                    fit_identifiers=sorted({rows[i]['identifier'] for i in fit_ix}),
                    excluded_room=room,history=history),path)
    save(folder/'manifest.json',dict(fit=fit_ix,calibration=cal_ix,test=test_ix,draws_per_epoch=draws,
          note='Reproduced structure/recipe with actual054 features; not bit-identical historical retraining. No held-room labels used to fit or calibrate.'))
    del model,opt,loader,t,s;gc.collect();torch.cuda.empty_cache()
    return path

def infer(model,tt,ss):
    output=[];model.cuda().eval()
    with torch.inference_mode():
        for perm in old.PERMS:
            idx,sign,si=[x.cuda() for x in old.permutation_indices(perm)];parts=[]
            for start in range(0,len(tt),16):
                t=torch.from_numpy(np.array(tt[start:start+16],copy=True)).cuda().flatten(0,1)
                s=torch.from_numpy(np.array(ss[start:start+16],copy=True)).cuda().flatten(0,1)
                parts.append(model.forward_features(t.index_select(1,idx)*sign,s.index_select(1,si)).cpu().reshape(-1,2,5))
            output.append(torch.cat(parts))
    return torch.stack(output,1)

def calibrate(raw,y,ix):
    # Fixed balanced-NLL calibration, labels from calibration records only.
    q=raw[ix].exp().mean((1,2)).clamp_min(1e-8).log();yy=y[ix]
    counts=torch.bincount(yy,minlength=5).clamp_min(1)
    w=1/counts[yy].float();w=w/w.sum()
    bias=torch.zeros(5,requires_grad=True);logt=torch.zeros((),requires_grad=True)
    opt=torch.optim.Adam([bias,logt],lr=.025)
    for _ in range(150):
        opt.zero_grad();b=bias-bias.mean();T=logt.exp().clamp(.5,2)
        loss=(torch.nn.functional.cross_entropy(q/T+b,yy,reduction='none')*w).sum()+.02*b.square().mean()+.02*logt.square()
        loss.backward();opt.step()
    return (bias-bias.mean()).detach(),float(logt.exp().clamp(.5,2).detach())

def combine(base,raw,b,T,method,base_raw=None,base_b=None,base_T=1.):
    p=base.softmax(-1)
    # Expert uses symmetric12view evidence, without inheriting the original model's protection rules.
    z=raw.exp().mean((1,2)).clamp_min(1e-12).log()/T+b
    q=z.softmax(-1)
    votes=((raw.exp().mean(2).clamp_min(1e-12).log()/T+b).argmax(-1)==q.argmax(-1)[:,None]).sum(-1)
    pp=p.topk(2,-1).values;qq=q.topk(2,-1).values
    pm=pp[:,0]-pp[:,1];qm=qq[:,0]-qq[:,1]
    if method in ('adjacent_up','adjacent_up_consensus','one_to_two'):
        proposal=(.75*p+.25*q).clamp_min(1e-12).log()
        before=base.argmax(-1);after=proposal.argmax(-1)
        mask=after==before+1
        if method=='adjacent_up_consensus':mask=mask&(votes>=4)&(qm>.1)
        if method=='one_to_two':mask=mask&(before==1)
        return torch.where(mask[:,None],proposal,base)
    if method=='mean25':mask=torch.ones_like(pm,dtype=torch.bool);amount=.25
    elif method=='weak_consensus':mask=(pm<.2)&(votes>=5)&(qm>.2);amount=.5
    elif method=='low_consensus':mask=(base.argmax(-1)<2)&(pm<.3)&(votes>=5)&(qm>.2);amount=.5
    elif method=='unanimous_rescue':mask=(pm<.3)&(votes==6)&(qm>.4);amount=.75
    elif method=='fallback50':
        v0=(base_raw[:,0,0]/base_T+base_b).argmax(-1)
        v45=(base_raw[:,0].exp().mean(1).clamp_min(1e-12).log()/base_T+base_b).argmax(-1)
        mask=(v0!=1)&(v45!=0)&(v45!=2);amount=.5
    else:raise ValueError(method)
    mixed=((1-amount)*p+amount*q).clamp_min(1e-12).log()
    return torch.where(mask[:,None],mixed,base)

def metric(z,base,y,ix,rows):
    out=old.metrics(z,y,base,ix);out['changed']=int((z[ix].argmax(-1)!=base[ix].argmax(-1)).sum())
    records=sorted({rows[i]['identifier'] for i in ix});acc=[]
    for record in records:
        jj=[i for i in ix if rows[i]['identifier']==record]
        acc.append(float((z[jj].argmax(-1)==y[jj]).float().mean()))
    out['mean_record_accuracy']=float(np.mean(acc));return out

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(goal_platform=.60,baseline_platform=.54,seeds=SEEDS,rooms=ROOMS,methods=METHODS,
       primary='Held-room proxy improvement; require >=2 rooms positive, pooled gain>=.02, no room worse>.03; whole-record bootstrap; source accuracy diagnostic only.',
       no_hidden_labels=True,training='Each new proxy excludes fit/calibration/test room entirely, fixed5epochs official labels only; no seed-selection using training accuracy.',
       calibration='150steps balancedNLL+regularization, calibration records only. Actual final uses all-source calibration explicitly non-independent.',
       reused_development_rooms=True,notes='Known-room proxies have been used before; not a fresh blind test. Pairing different historical original proxies with current-feature new fits also conflates training implementation.'))
    rows=read(PREV/'rows.json');tt=np.load(PREV/'t.npy',mmap_mode='r');ss=np.load(PREV/'s.npy',mmap_mode='r')
    y=torch.tensor([int(r['headcount']) for r in rows]);results=[]
    for room in (*ROOMS,'actual'):
        data=torch.load(PREV/f'{room}.pt',weights_only=True);base_raw=data['raw'];bb,bt=old.calibration(room)
        base=old.policy(base_raw,bb,bt);fit_ix,cal_ix,test_ix=old.split(room,rows)
        for seed in SEEDS:
            cp=PACKAGE/f'weights/checkpoints/base037_full_s{seed}.pt' if room=='actual' else fit(room,seed,rows,tt,ss,y)
            cache=WORK/f'raw_{room}_s{seed}.pt'
            if cache.exists():raw=torch.load(cache,weights_only=True)['raw']
            else:
                model=StructuredModel('spectral','physical');model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model_state'],strict=True)
                round_storage(model);started=time.perf_counter();raw=infer(model,tt,ss)
                torch.save(dict(raw=raw,checkpoint_sha256=sha(cp),preprocessing_sha256=sha(PREV/'preprocessing.onnx')),cache)
                print('INFER',room,seed,round(time.perf_counter()-started,1),flush=True)
                del model;gc.collect();torch.cuda.empty_cache()
            b,T=calibrate(raw,y,cal_ix if room!='actual' else fit_ix)
            save(WORK/f'cal_{room}_s{seed}.json',dict(bias=b.tolist(),temperature=T,checkpoint_sha256=sha(cp),
                 calibration_identifiers=sorted({rows[i]['identifier'] for i in (cal_ix if room!='actual' else fit_ix)})))
            expert=(raw.exp().mean((1,2)).clamp_min(1e-12).log()/T+b)
            save(WORK/f'expert_{room}_s{seed}.json',metric(expert,base,y,test_ix,rows))
            for method in METHODS:
                z=combine(base,raw,b,T,method,base_raw,bb,bt)
                results.append(dict(room=room,seed=seed,method=method,metrics=metric(z,base,y,test_ix,rows),
                                    baseline=metric(base,base,y,test_ix,rows)))
                torch.save(dict(logits=z,baseline=base,labels=y,indices=test_ix),WORK/f'pred_{room}_s{seed}_{method}.pt')
        save(WORK/'results.json',results)
        print('ROOM_COMPLETE',room,flush=True)
    summaries=[]
    for seed in SEEDS:
        for method in METHODS:
            rr=[r for r in results if r['seed']==seed and r['method']==method and r['room']!='actual']
            gains=[r['metrics']['accuracy']-r['baseline']['accuracy'] for r in rr]
            total=sum(r['metrics']['total'] for r in rr)
            gain=sum(r['metrics']['correct']-r['baseline']['correct'] for r in rr)/total
            summaries.append(dict(seed=seed,method=method,pooled_gain=gain,room_gains=dict(zip(ROOMS,gains)),
               positive_rooms=sum(g>0 for g in gains),passed=gain>=.02 and min(gains)>=-.03 and sum(g>0 for g in gains)>=2))
    summaries.sort(key=lambda r:(r['passed'],r['pooled_gain']),reverse=True)
    save(WORK/'summary.json',summaries)
    passing=[r for r in summaries if r['passed']]
    save(WORK/'selection.json',dict(selected=passing[0] if passing else None,reason='Predeclared held-room gate; no selection from actual-source accuracy.',
          status='candidate_pending_bootstrap_export' if passing else 'no_candidate_passed',platform_accuracy=None))
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
