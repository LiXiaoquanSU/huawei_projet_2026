"""Separate learned categorical/ordinal heads, preserve existing routing decisions."""
import gc,time
from ra_common import *
from export_batch import FeatureHead
from model import ordered_probabilities
DEC=Path('D:/huawei2026-experiments/decoder_consensus_20261025')

def parts(head,e,s):
    evidence=[e[:,k] for k in range(4)];content=torch.cat(evidence,1)
    context=head.context(torch.cat([s[:,:6],s[:,12:17]],1));gate=.5*head.gate(torch.cat([content,context],1)).softmax(-1)+.125
    experts=torch.stack([h(z) for h,z in zip(head.experts,evidence)],1);mix=(experts.softmax(-1)*gate[:,:,None]).sum(1)
    strength=torch.cat([s[:,6:12],s[:,12:17]],1);z=head.shared(content)+head.strength(strength)
    direct=.5*head.classifier(z).softmax(-1)+.5*mix;ordinal=head.ordinal(z);presence=head.presence(torch.cat([*evidence[1:],strength],1))
    ordered,_=ordered_probabilities(torch.cat([presence,ordinal[:,1:]],1));return direct,ordered

def fuse(d,o,method):
    p=(d+o)*.5
    if method=='direct':q=d
    elif method=='ordinal':q=o
    elif method=='entropy':
        hd=-(d*d.clamp_min(1e-8).log()).sum(-1);ho=-(o*o.clamp_min(1e-8).log()).sum(-1)
        w=((np.log(5)-hd).clamp_min(.01)/((2*np.log(5)-hd-ho).clamp_min(.02))).clamp(.25,.75);q=w[...,None]*d+(1-w[...,None])*o
    elif method=='margin':
        a=d.topk(2,-1).values;b=o.topk(2,-1).values;ad=(a[...,0]-a[...,1]).clamp_min(.01);bo=(b[...,0]-b[...,1]).clamp_min(.01);w=(ad/(ad+bo)).clamp(.25,.75);q=w[...,None]*d+(1-w[...,None])*o
    elif method=='geometric':q=(d.clamp_min(1e-8)*o.clamp_min(1e-8)).sqrt();q=q/q.sum(-1,keepdim=True)
    else:raise ValueError(method)
    return (.75*p+.25*q).clamp_min(1e-8).log()

def fixed_policy(new,reference,b,T):
    v0=reference[:,0,0]/T+b;v45=reference[:,0].exp().mean(1).clamp_min(1e-12).log()/T+b
    n0=new[:,0,0]/T+b;n45=new[:,0].exp().mean(1).clamp_min(1e-12).log()/T+b
    full=(new.exp().mean(2).clamp_min(1e-12).log()/T+b).exp().mean(1).clamp_min(1e-12).log()
    inside=torch.where(((v45.argmax(-1)==0)|(v45.argmax(-1)==2))[:,None],n45,full)
    return torch.where((v0.argmax(-1)==1)[:,None],n0,inside)+new.new_tensor([0,.35,0,0,0])

def result(d,method,scope,b,T):
    raw,student=d['raw'],d['student'];base0=old.policy(raw,b,T);base=prev.transfer(base0,old.policy(student,b,T),'low_guard')
    new0=fixed_policy(fuse(d['direct'],d['ordinal'],method),raw,b,T)
    new1=fixed_policy(fuse(d['student_direct'],d['student_ordinal'],method),student,b,T)
    new=torch.where((base0.argmax(-1)<2)[:,None],new1,new0)
    if scope=='low':new=torch.where((base.argmax(-1)<2)[:,None],new,base)
    return new,base

def prepare():
    DEC.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(DEC/'plan.json',dict(base_sha256=SOURCE_SHA,methods=['direct','ordinal','entropy','margin','geometric'],scopes=['all','low'],amount=.25,weights_frozen=True,routing='All original054/student policies and final054selective branch decisions fixed to their existing scores; only selected score pooling changes.',selection='10fixedconfigurations,no strength/bias/threshold scan. Proxy totalgain>0,>=2roomsimprove,no roomloss>2pp,source>=679,source4personloss<=1. Reuseddevelopmentrooms,not blind.',support_round='6fixedlabel-free support gates failed; allretained,noexport'))
    tt=np.load(PREV/'t.npy',mmap_mode='r');ss=np.load(PREV/'s.npy',mmap_mode='r');rows=read(PREV/'rows.json')
    from quantized import updated_state
    from ssl_common import Student
    for room in ('a1','c1','g1','e1','actual'):
        path=DEC/f'{room}.pt'
        if path.exists():continue
        m,b=old.build(old.checkpoint(room),full=room=='actual');m=m.cuda().eval();student=Student(m).cuda().eval();state,_=updated_state(room)
        student.load_state_dict({k.replace('phase.project.','phase_project.').replace('temporal.blocks.2.','time_block.').replace('temporal.project.','time_project.'):v.cuda() for k,v in state.items()},strict=False)
        captured={};handles=[]
        def pre(name):
            def hook(module,args):captured[name]=args[0]
            return hook
        def post(name):
            def hook(module,args,out):captured[name]=out
            return hook
        handles+=[m.phase.project.register_forward_pre_hook(pre('p')),m.temporal.blocks[-1].register_forward_pre_hook(pre('t'))]
        for n in ('static','dynamic','phase','temporal'):handles.append(getattr(m,n).register_forward_hook(post(n)))
        pools={k:[] for k in ('direct','ordinal','student_direct','student_ordinal')};started=time.perf_counter()
        with torch.inference_mode():
          for permutation in old.PERMS:
            idx,sign,si=[v.cuda() for v in old.permutation_indices(permutation)];acc={k:[] for k in pools}
            for start in range(0,960,16):
                t=torch.from_numpy(np.array(tt[start:start+16],copy=True)).cuda().flatten(0,1);s=torch.from_numpy(np.array(ss[start:start+16],copy=True)).cuda().flatten(0,1);s=s.index_select(1,si)
                tasks=m.forward_tasks(t.index_select(1,idx)*sign,s);e=torch.stack([captured[n] for n in ('static','dynamic','phase','temporal')],1);p,timevalue=student.encode(captured['p'],captured['t']);se=torch.stack([e[:,0],e[:,1],p,timevalue],1);sd,so=parts(student.decoder,se,s)
                vals=[tasks['direct'],tasks['ordinal'],sd,so]
                for k,v in zip(pools,vals):acc[k].append(v.cpu().reshape(-1,2,5))
            for k in pools:pools[k].append(torch.cat(acc[k]))
            print('DEC_PREP',room,permutation,round(time.perf_counter()-started,1),flush=True)
        prior=torch.load(PREV/f'{room}.pt',weights_only=True);d={k:torch.stack(v,1) for k,v in pools.items()}
        for prefix,ref in (('',prior['raw']),('student_',prior['student'])):
            err=float((((d[prefix+'direct']+d[prefix+'ordinal'])*.5).clamp_min(1e-8).log()-ref).abs().max());assert err<1e-5,err
        d.update(raw=prior['raw'],student=prior['student'],labels=prior['labels'],rows=rows);torch.save(d,path)
        for h in handles:h.remove()
        del m,student,prior,d,pools;gc.collect();torch.cuda.empty_cache()

def evaluate():
    results=[]
    for method in ('direct','ordinal','entropy','margin','geometric'):
     for scope in ('all','low'):
      item=dict(method=method,scope=scope,amount=.25,rooms={});net=0;improve=0;worst=1
      for room in ('a1','c1','g1','e1','actual'):
        d=torch.load(DEC/f'{room}.pt',weights_only=True);b,T=old.calibration(room);z,base=result(d,method,scope,b,T);ix=old.split(room,d['rows'])[2];mt=old.metrics(z,d['labels'],base,ix);item['rooms'][room]=mt
        torch.save(dict(logits=z,base=base,labels=d['labels'],test=ix),DEC/f'{method}_{scope}_{room}.pt')
        if room!='actual':n=mt['corrected']-mt['broken'];net+=n;improve+=n>0;worst=min(worst,n/mt['total'])
      source=item['rooms']['actual'];item.update(proxy_net=net,improving_rooms=improve,worst_room_delta=worst,eligible=net>0 and improve>=2 and worst>=-.02 and source['correct']>=679 and source['confusion_matrix'][4][4]>=135);results.append(item)
    save(DEC/'results.json',results)
    for r in results:print('DEC_RESULT',r['method'],r['scope'],r['proxy_net'],r['improving_rooms'],r['worst_room_delta'],r['rooms']['actual']['correct'],r['rooms']['actual']['confusion_matrix'][4][4],r['eligible'],{k:v['corrected']-v['broken'] for k,v in r['rooms'].items()},flush=True)
    eligible=[r for r in results if r['eligible']]
    if eligible:save(DEC/'selection.json',max(eligible,key=lambda r:(r['proxy_net'],r['worst_room_delta'])))
if __name__=='__main__':prepare();evaluate()
