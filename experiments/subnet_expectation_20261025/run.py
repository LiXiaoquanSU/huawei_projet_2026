"""Exact expected inference under the original route-dropout training distribution."""
import gc,itertools,json,sys,time,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/clock_origin_20261025'))
spec=importlib.util.spec_from_file_location('clock_origin_shared',ROOT/'experiments/clock_origin_20261025/run.py')
clock=importlib.util.module_from_spec(spec);spec.loader.exec_module(clock)
old=clock.old
WORK=Path('D:/huawei2026-experiments/subnet_expectation_20261025')
PREV=clock.PREV;LATENTS=clock.LATENTS;ROOMS=clock.ROOMS
save,read,sha=old.save,old.read,old.sha
from model import ordered_probabilities

def mask_distribution():
    masks=[];weights=[]
    for bits in itertools.product((0.,1.),repeat=4):
        kept=sum(bits)
        if not kept:continue
        p=.75**kept*.25**(4-kept)
        if kept==4:p+=.25**4 # original code restores all routes if all sampled absent
        masks.append(bits);weights.append(p)
    assert abs(sum(weights)-1)<1e-12
    return masks,weights

def decode(model,e,s,mask):
    # Match original TRAINING route-dropout semantics exactly. Context/gates and
    # presence use unmasked evidence, shared classifier uses masked content.
    evidence=[e[:,k] for k in range(4)];content=e.flatten(1)
    context=model.context(torch.cat([s[:,:6],s[:,12:17]],1))
    gate=.5*model.gate(torch.cat([content,context],1)).softmax(-1)+.125
    keep=e.new_tensor(mask).reshape(1,4)
    gate=gate*keep;gate=gate/gate.sum(-1,keepdim=True)
    experts=torch.stack([head(v) for head,v in zip(model.experts,evidence)],1)
    mixture=(experts.softmax(-1)*gate[:,:,None]).sum(1)
    content=(e*keep[:,:,None]).flatten(1)
    strength=torch.cat([s[:,6:12],s[:,12:17]],1)
    z=model.shared(content)+model.strength(strength)
    direct=.5*model.classifier(z).softmax(-1)+.5*mixture
    ordinal=model.ordinal(z);presence=model.presence(torch.cat([*evidence[1:],strength],1))
    ordered,_=ordered_probabilities(torch.cat([presence,ordinal[:,1:]],1))
    return ((direct+ordered)*.5).clamp_min(1e-8).log()

def evidence(room,tt,ss):
    folder=WORK/'evidence';folder.mkdir(exist_ok=True);path=folder/f'{room}.pt'
    if path.exists():return
    model,_=old.build(old.checkpoint(room),full=room=='actual');model.cuda().eval()
    lat=torch.load(LATENTS/f'{room}.pt',weights_only=True);prior=torch.load(PREV/f'{room}.pt',weights_only=True)
    es=[];summaries=[];zero=0.;started=time.perf_counter()
    with torch.inference_mode():
        for k,perm in enumerate(old.PERMS):
            idx,sign,si=[v.cuda() for v in old.permutation_indices(perm)];ee=[];sss=[]
            for start in range(0,960,16):
                t=torch.from_numpy(np.array(tt[start:start+16],copy=True)).cuda().flatten(0,1).index_select(1,idx)*sign
                s=torch.from_numpy(np.array(ss[start:start+16],copy=True)).cuda().flatten(0,1).index_select(1,si)
                a=t[:,:384].reshape(-1,3,128,150)
                e=torch.cat([model.static(a)[:,None],lat['e'][start:start+16,2*k:2*k+2].cuda().flatten(0,1)],1)
                zero=max(zero,float((decode(model,e,s,(1.,1.,1.,1.))-prior['raw'][start:start+16,k].cuda().flatten(0,1)).abs().max()))
                ee.append(e.cpu().reshape(-1,2,4,256));sss.append(s.cpu().reshape(-1,2,17))
            es.append(torch.cat(ee));summaries.append(torch.cat(sss))
    assert zero<3e-5,zero
    torch.save(dict(e=torch.stack(es,1),s=torch.stack(summaries,1),raw=prior['raw'],labels=prior['labels'],rows=prior['rows'],
                    checkpoint_sha256=sha(old.checkpoint(room)),zero_error=zero),path)
    print('SUBNET_EVIDENCE',room,round(time.perf_counter()-started,1),zero,flush=True)
    del model,lat,prior,es,summaries;gc.collect();torch.cuda.empty_cache()

def prepare():
    WORK.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    masks,weights=mask_distribution()
    save(WORK/'plan.json',dict(goal_platform=.60,original_platform=.54,last_seed_expert_platform=.53,
       hypothesis='Original model trained with25% independent route dropout, but inference uses all routes. Marginalize the actually trained15 nonempty masks to reduce reliance on room shortcuts.',
       masks=masks,weights=weights,training=False,new_bias=False,
       constraints='No new seed/external data, preserve original presence semantics. Full mask numerical control required. Source not independent.',
       selection='Fixed configs, positive net in>=2 held rooms, no roomloss>2pp, aggregate>=2pp, whole-record uncertainty. Reused development, not a blind test.'))
    tt=np.load(PREV/'t.npy',mmap_mode='r');ss=np.load(PREV/'s.npy',mmap_mode='r')
    for room in ROOMS:
        evidence(room,tt,ss);path=WORK/f'{room}.pt'
        if path.exists():continue
        d=torch.load(WORK/'evidence'/f'{room}.pt',weights_only=True);model,_=old.build(old.checkpoint(room),full=room=='actual');model.cuda().eval()
        outputs=[]
        with torch.inference_mode():
            for mask in masks:
                parts=[]
                for start in range(0,960,32):
                    e=d['e'][start:start+32].cuda().flatten(0,2);s=d['s'][start:start+32].cuda().flatten(0,2)
                    parts.append(decode(model,e,s,mask).cpu().reshape(-1,6,2,5))
                outputs.append(torch.cat(parts))
        out=torch.stack(outputs,1)
        assert float((out[:,-1]-d['raw']).abs().max())<3e-5
        torch.save(dict(raw=d['raw'],subnets=out,masks=masks,weights=weights,labels=d['labels'],rows=d['rows']),path)
        print('SUBNET_DONE',room,flush=True);del model,d,out,outputs;gc.collect();torch.cuda.empty_cache()

def alternate(raw,subnets,kind):
    masks,weights=mask_distribution()
    if kind=='expected':return (subnets.exp()*subnets.new_tensor(weights)[None,:,None,None,None]).sum(1).clamp_min(1e-12).log()
    target={'no_static':(0.,1.,1.,1.),'no_time':(1.,1.,1.,0.)}[kind]
    return subnets[:,masks.index(target)]

def policy(raw,alt,b,T,scope,amount):
    base=old.policy(raw,b,T)
    if scope=='fallback':
        v=alt.exp().mean(2).clamp_min(1e-12).log()/T+b
        other=old.finish(raw,b,T,v.exp().mean(1).clamp_min(1e-12).log())
    else:other=old.policy(alt,b,T)
    mixed=((1-amount)*base.softmax(-1)+amount*other.softmax(-1)).clamp_min(1e-12).log()
    if scope=='weak':
        top=base.softmax(-1).topk(2,-1).values;mask=top[:,0]-top[:,1]<.2
        return torch.where(mask[:,None],mixed,base)
    if scope=='low':return torch.where((base.argmax(-1)<2)[:,None],mixed,base)
    return mixed

def evaluate():
    configs=[dict(kind=kind,scope=scope,amount=a) for kind in ('expected','no_static','no_time') for scope,a in [('all',.25),('all',.5),('weak',.5),('fallback',.5)]]
    # Stage2: pre-existing selective054 high-count protection, motivated by the
    # observed source3->4 breakage. Explicitly post-stage1 development, not blind.
    configs += [dict(kind='no_time',scope='low',amount=a) for a in (.25,.5)]
    results=[]
    for c in configs:
        key=f'{c["kind"]}_{c["scope"]}_{c["amount"]}';items=[]
        for room in ROOMS:
            d=torch.load(WORK/f'{room}.pt',weights_only=True);b,T=old.calibration(room);base=old.policy(d['raw'],b,T)
            alt=alternate(d['raw'],d['subnets'],c['kind']);z=policy(d['raw'],alt,b,T,c['scope'],c['amount'])
            ix=old.split(room,d['rows'])[2];m=old.metrics(z,d['labels'],base,ix)
            items.append(dict(room=room,metrics=m,baseline=old.metrics(base,d['labels'],base,ix)))
            torch.save(dict(logits=z,baseline=base,labels=d['labels'],indices=ix),WORK/f'pred_{room}_{key}.pt')
        proxy=items[:-1];net=sum(r['metrics']['correct']-r['baseline']['correct'] for r in proxy);n=sum(r['metrics']['total'] for r in proxy)
        gains={r['room']:r['metrics']['accuracy']-r['baseline']['accuracy'] for r in proxy}
        results.append(dict(config=c,key=key,rooms=items,proxy_net=net,proxy_total=n,pooled_gain=net/n,room_gains=gains,
          eligible=net/n>=.02 and min(gains.values())>=-.02 and sum(g>0 for g in gains.values())>=2))
    results.sort(key=lambda r:(r['eligible'],r['pooled_gain']),reverse=True);save(WORK/'results.json',results)
    save(WORK/'selection.json',dict(selected=next((r for r in results if r['eligible']),None),platform_score=None))
    print(json.dumps([dict(key=r['key'],net=r['proxy_net'],rooms=r['room_gains'],eligible=r['eligible']) for r in results],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':prepare();evaluate()
