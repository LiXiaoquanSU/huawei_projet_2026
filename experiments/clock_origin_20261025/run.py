"""Marginalize arbitrary Fourier clock origin, reuse frozen evidence, no hidden labels."""
import gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
from model import ordered_probabilities
WORK=Path('D:/huawei2026-experiments/clock_origin_20261025')
PREV=Path('D:/huawei2026-experiments/next053_20261025')
LATENTS=Path('D:/huawei2026-experiments/route_attention_20261025')
ROOMS=('a1','c1','g1','e1','actual')
OFFSETS=(.125,-.125,.25,-.25,.5,-.5)
save,read,sha=old.save,old.read,old.sha

def decode(model,e,s):
    evidence=[e[:,k] for k in range(4)];content=torch.cat(evidence,1)
    context=model.context(torch.cat([s[:,:6],s[:,12:17]],1))
    gate=.5*model.gate(torch.cat([content,context],1)).softmax(-1)+.125
    experts=torch.stack([h(v) for h,v in zip(model.experts,evidence)],1)
    mixture=(experts.softmax(-1)*gate[:,:,None]).sum(1)
    strength=torch.cat([s[:,6:12],s[:,12:17]],1);z=model.shared(content)+model.strength(strength)
    direct=.5*model.classifier(z).softmax(-1)+.5*mixture
    ordinal=model.ordinal(z);presence=model.presence(torch.cat([*evidence[1:],strength],1))
    ordered,_=ordered_probabilities(torch.cat([presence,ordinal[:,1:]],1))
    return ((direct+ordered)*.5).clamp_min(1e-8).log()

def temporal_origin(model,conv,timing,offset):
    clock=timing.reshape(-1,3,75,2).mean(-1).transpose(1,2)
    angle=(clock[:,:,:1]+offset)*model.temporal.frequencies[None,None,:]*6.283185307179586
    x=conv+model.temporal.time(torch.cat([angle.sin(),angle.cos(),clock[:,:,1:]],-1))
    x=model.temporal.blocks(x);mean=x.mean(1);std=(x.sub(mean[:,None]).square().mean(1)+1e-5).sqrt()
    return model.temporal.project(torch.cat([mean,std],1))

def prepare():
    WORK.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(WORK/'plan.json',dict(goal_platform=.60,baseline_platform=.54,last_crossseed_platform=.53,
      hypothesis='Absolute Fourier position depends on arbitrary window time origin; marginalize origin, keep all physical durations/content and all weights unchanged.',
      offsets_seconds=OFFSETS,training=False,new_bias=False,room_specific_rules=False,
      controls='Zero offset decoder and temporal output must match actual054 caches; unchanged amplitude/phase evidence.',
      selection='>=2 improving rooms, aggregate>=2pp, no room worse>2pp, whole-recordbootstrap. Actual source diagnostic only.',
      reused_development=True,scope='Offset origin inside Fourier embedding only, not change ATI, CSI, temporal step, or gaps; not actual motion-state labels.'))
    tt=np.load(PREV/'t.npy',mmap_mode='r');ss=np.load(PREV/'s.npy',mmap_mode='r')
    for room in ROOMS:
        path=WORK/f'{room}.pt'
        if path.exists():continue
        prior=torch.load(PREV/f'{room}.pt',weights_only=True);lat=torch.load(LATENTS/f'{room}.pt',weights_only=True)
        model,_=old.build(old.checkpoint(room),full=room=='actual');model.cuda().eval()
        assert lat['checkpoint_sha256']==sha(old.checkpoint(room))
        output=[];maxcheck=0.;started=time.perf_counter()
        with torch.inference_mode():
            for k,permutation in enumerate(old.PERMS):
                idx,sign,si=[v.cuda() for v in old.permutation_indices(permutation)]
                accum=[[] for _ in OFFSETS]
                for start in range(0,960,16):
                    t=torch.from_numpy(np.array(tt[start:start+16],copy=True)).cuda().flatten(0,1).index_select(1,idx)*sign
                    s=torch.from_numpy(np.array(ss[start:start+16],copy=True)).cuda().flatten(0,1).index_select(1,si)
                    a=t[:,:768].reshape(-1,6,128,150);p=t[:,768:960].reshape(-1,6,32,150)
                    static=model.static(a[:,:3]);rest=lat['e'][start:start+16,2*k:2*k+2].cuda().flatten(0,1)
                    e=torch.cat([static[:,None],rest],1)
                    dynamic=torch.cat([a[:,3:].reshape(-1,3,32,4,150).mean(3),p],1).reshape(-1,288,150)
                    conv=model.temporal.input(dynamic).transpose(1,2)
                    zero=temporal_origin(model,conv,t[:,960:],0.)
                    maxcheck=max(maxcheck,float((zero-e[:,3]).abs().max()),
                        float((decode(model,e,s)-prior['raw'][start:start+16,k].cuda().flatten(0,1)).abs().max()))
                    for j,offset in enumerate(OFFSETS):
                        newtime=temporal_origin(model,conv,t[:,960:],offset)
                        shifted=torch.cat([e[:,:3],newtime[:,None]],1)
                        accum[j].append(decode(model,shifted,s).cpu().reshape(-1,2,5))
                output.append(torch.stack([torch.cat(v) for v in accum],1))
                print('CLOCK_PREP',room,k,round(time.perf_counter()-started,1),flush=True)
        assert maxcheck<3e-5,maxcheck
        shifted=torch.stack(output,2) # N,offset,rx,time,class
        torch.save(dict(raw=prior['raw'],shifted=shifted,offsets=OFFSETS,labels=prior['labels'],rows=prior['rows'],
                        zero_offset_maxerror=maxcheck,checkpoint_sha256=sha(old.checkpoint(room))),path)
        print('CLOCK_ZERO_CONTROL',room,maxcheck,flush=True)
        del model,prior,lat,output,shifted;gc.collect();torch.cuda.empty_cache()

def result(raw,shifted,b,T,scope,amount):
    base=old.policy(raw,b,T)
    if scope=='fallback':
        q=(raw.exp()+shifted.exp().sum(1))/(1+shifted.shape[1])
        l=q.mean(2).clamp_min(1e-12).log()/T+b
        alternative=old.finish(raw,b,T,l.exp().mean(1).clamp_min(1e-12).log())
    else:
        marginal=((raw.exp()+shifted.exp().sum(1))/(1+shifted.shape[1])).clamp_min(1e-12).log()
        alternative=old.policy(marginal,b,T)
    mixed=((1-amount)*base.softmax(-1)+amount*alternative.softmax(-1)).clamp_min(1e-12).log()
    if scope=='weak':
        top=base.softmax(-1).topk(2,-1).values;mask=top[:,0]-top[:,1]<.2
        return torch.where(mask[:,None],mixed,base)
    return mixed

def evaluate():
    configs=[dict(pair=pair,scope=scope,amount=amount) for pair in (0,1,2) for scope,amount in [('all',.25),('all',.5),('weak',.5),('fallback',1.)]]
    rows=read(PREV/'rows.json');results=[];summary=[]
    for c in configs:
        key=f'p{c["pair"]}_{c["scope"]}_{c["amount"]}';items=[]
        for room in ROOMS:
            d=torch.load(WORK/f'{room}.pt',weights_only=True);b,T=old.calibration(room)
            base=old.policy(d['raw'],b,T);part=d['shifted'][:,2*c['pair']:2*c['pair']+2]
            z=result(d['raw'],part,b,T,c['scope'],c['amount']);ix=old.split(room,rows)[2]
            m=old.metrics(z,d['labels'],base,ix);items.append(dict(room=room,metrics=m,baseline=old.metrics(base,d['labels'],base,ix)))
            torch.save(dict(logits=z,baseline=base,labels=d['labels'],indices=ix),WORK/f'pred_{room}_{key}.pt')
        proxy=items[:-1];gains={r['room']:r['metrics']['accuracy']-r['baseline']['accuracy'] for r in proxy}
        n=sum(r['metrics']['total'] for r in proxy);net=sum(r['metrics']['correct']-r['baseline']['correct'] for r in proxy)
        item=dict(config=c,key=key,rooms=items,proxy_net=net,proxy_total=n,pooled_gain=net/n,room_gains=gains,
          eligible=net/n>=.02 and min(gains.values())>=-.02 and sum(g>0 for g in gains.values())>=2)
        results.append(item)
    results.sort(key=lambda r:(r['eligible'],r['pooled_gain']),reverse=True)
    save(WORK/'results.json',results);save(WORK/'selection.json',dict(selected=next((r for r in results if r['eligible']),None),platform_score=None))
    print(json.dumps([dict(key=r['key'],gain=r['pooled_gain'],net=r['proxy_net'],rooms=r['room_gains'],eligible=r['eligible']) for r in results],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':prepare();evaluate()
