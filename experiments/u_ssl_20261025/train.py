"""Matched supervised/source-SSL/official-U SSL updates, honest outer-room controls."""
import argparse,gc,time
from ssl_common import *

def frozen_hash(student):
    h=hashlib.sha256()
    for k,v in sorted(student.decoder.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def indices_and_cal(room,rows):
    if room=='actual':return list(range(960)),[],list(range(960)),None
    manifest=read(OLD/'results'/f's42_{room}'/'manifest.json')
    ids={k:{r['identifier'] for r in manifest[k]} for k in ('fit','calibration','test')}
    assert all(ids[a].isdisjoint(ids[b]) for a,b in (('fit','calibration'),('fit','test'),('calibration','test')))
    assert all(r['room']!=room for r in manifest['fit'])
    cal=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline']
    return *[[i for i,r in enumerate(rows) if r['identifier'] in ids[k]] for k in ('fit','calibration','test')],cal

def u_balance(rows):
    groups={};counts={}
    for r in rows:
        groups.setdefault(r['room'],set()).add(r['identifier']);counts[r['identifier']]=counts.get(r['identifier'],0)+1
    w=torch.tensor([1/(len(groups)*len(groups[r['room']])*counts[r['identifier']]) for r in rows]);return w/w.sum()

def interpolated(student,initial,state,amount):
    params={k:(initial[k]*(1-amount)+state[k]*amount) for k in initial}
    student.load_state_dict(params,strict=False)

def fit(room,method,seed,steps=240):
    folder=WORK/'fits'/f'{room}_{method}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return read(folder/'complete.json')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    cp=checkpoint(room);source=torch.load(WORK/'prefixes'/room/'source.pt',weights_only=True)
    fit_ix,cal_ix,test_ix,cal=indices_and_cal(room,source['rows']);labels=source['labels'];rows=source['rows']
    model,b=build(cp,full=room=='actual');student=Student(model).cuda().eval();del model
    assert all(not module.training for module in student.modules())
    frozen=frozen_hash(student);initial=student.selected_state()
    trainable=sum(p.numel() for name,p in student.named_parameters() if p.requires_grad and not name.startswith('projector.'))
    # The original source split excludes complete calibration and test recordings.
    if room!='actual':
        prefit=torch.load(cp,weights_only=True);assert set(prefit['fit_identifiers'])=={rows[i]['identifier'] for i in fit_ix};del prefit
    T=1. if cal is None else cal['temperature'];bias=b if cal is None else torch.tensor(cal['bias'])
    base=policy(source['logits'],bias,T)
    before,raw,_=scores(student,source,bias,T)
    parity=float((raw.exp()-source['logits'].exp()).abs().max());assert parity<1e-5,parity
    assert torch.equal(before.argmax(-1),base.argmax(-1))
    gpu={k:source[k].cuda() for k in ('phase','time','latents','summaries','logits')};yy=labels.cuda();bb=bias.cuda()
    sourceweights=balance(rows,labels,fit_ix);sourceids=torch.tensor(fit_ix)
    gsource=torch.Generator().manual_seed(seed);gssl=torch.Generator().manual_seed(seed+10000)
    gpair=torch.Generator().manual_seed(seed+20000);pair_rooms=[];pair_indices={}
    if method in ('paired','paired_u'):
        for domain in sorted({rows[i]['room'] for i in fit_ix if rows[i]['room']!='unknown'}):
            indices={label:[i for i in fit_ix if rows[i]['room']==domain and int(labels[i])==label] for label in (0,1)}
            if all(indices.values()):pair_rooms.append(domain);pair_indices[domain]=indices
        assert pair_rooms
    if method in ('official_u','paired_u'):
        urows=read(WORK/'prefixes'/room/'u_rows.json');uw=u_balance(urows)
        up=np.load(WORK/'prefixes'/room/'u_phase.npy',mmap_mode='r');ut=np.load(WORK/'prefixes'/room/'u_time.npy',mmap_mode='r')
    groups=[dict(params=[p for name,p in student.named_parameters() if p.requires_grad and not name.startswith('projector.')],lr=5e-5),
            dict(params=student.projector.parameters(),lr=5e-4)]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    history=[];snapshots={};sampler_hash=hashlib.sha256();start_time=time.perf_counter()
    for step in range(1,steps+1):
        selection=sourceids[torch.multinomial(sourceweights,16,replacement=True,generator=gsource)]
        sampler_hash.update(selection.numpy().tobytes());ix=selection.cuda()
        z,rep=student(*[gpu[k][ix].flatten(0,1) for k in ('phase','time','latents','summaries')]);z=z.reshape(-1,4,5)
        score=policy(z,bb,T);target=yy[ix]
        view_scores=z/T+bb+z.new_tensor([0,.35,0,0,0])
        ce=.75*F.cross_entropy(score,target,label_smoothing=.02)+.25*F.cross_entropy(view_scores.flatten(0,1),target[:,None].expand(-1,4).flatten(),label_smoothing=.02)
        teacher=gpu['logits'][ix];kl=(teacher.exp()*(teacher-z)).sum(-1).mean()
        oldrep=gpu['latents'][ix,:,2:].flatten(0,1).flatten(1)
        drift=F.mse_loss(rep,oldrep)
        loss=ce+.1*kl+.02*drift;ssl_terms={};ssl_value=0.
        if method in ('source_ssl','official_u','paired_u'):
            if method=='source_ssl':
                ui=sourceids[torch.multinomial(sourceweights,32,replacement=True,generator=gssl)]
                vi=torch.randint(1,4,(32,),generator=gssl)
                pa=gpu['phase'][ui.cuda(),0];ta=gpu['time'][ui.cuda(),0]
                pb=gpu['phase'][ui.cuda(),vi.cuda()];tb=gpu['time'][ui.cuda(),vi.cuda()]
            else:
                ui=torch.multinomial(uw,32,replacement=True,generator=gssl).numpy();vi=torch.randint(1,4,(32,),generator=gssl).numpy()
                pa=torch.from_numpy(np.array(up[ui,0],copy=True)).cuda();ta=torch.from_numpy(np.array(ut[ui,0],copy=True)).cuda()
                pb=torch.from_numpy(np.array(up[ui,vi],copy=True)).cuda();tb=torch.from_numpy(np.array(ut[ui,vi],copy=True)).cuda()
            a=student.encode(pa,ta);b=student.encode(pb,tb)
            qa=student.projector(torch.cat(a,-1));qb=student.projector(torch.cat(b,-1))
            ssl,ssl_terms=vicreg(qa,qb);loss=loss+.02*ssl;ssl_value=float(ssl.detach())
        pair_info={}
        if method in ('paired','paired_u'):
            pairs=[]
            for pair in range(8):
                domain=pair_rooms[int(torch.randint(len(pair_rooms),(1,),generator=gpair))]
                for label in (0,1):
                    options=pair_indices[domain][label];pairs.append(options[int(torch.randint(len(options),(1,),generator=gpair))])
            pi=torch.tensor(pairs,device='cuda')
            paired_z,_=student(*[gpu[k][pi].flatten(0,1) for k in ('phase','time','latents','summaries')]);paired_z=paired_z.reshape(16,4,5)
            odds=(paired_z[:,:,1]-paired_z[:,:,0])/T+(bb[1]-bb[0])+.35
            binary=F.binary_cross_entropy_with_logits(odds,yy[pi,None].float().expand(-1,4))
            pair_odds=odds.mean(1).reshape(8,2);ranking=F.softplus(1-(pair_odds[:,1]-pair_odds[:,0])).mean()
            loss=loss+.3*binary+.1*ranking
            pair_info=dict(binary=float(binary.detach()),ranking=float(ranking.detach()))
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm);optimizer.step()
        if step in (40,120,240):
            current=student.selected_state();snapshots[step]=current
            zz,raw,reps=scores(student,source,bias,T)
            ci=torch.tensor(cal_ix or fit_ix);w=balance(rows,labels,cal_ix or fit_ix)
            nll=float((F.cross_entropy(zz[ci],labels[ci],reduction='none')*w).sum())
            record=dict(step=step,loss=float(loss.detach()),ce=float(ce.detach()),teacher_kl=float(kl.detach()),representation_drift=float(drift.detach()),
              ssl=ssl_value,ssl_terms=ssl_terms,pair_loss=pair_info,calibration_nll=nll,fit=metrics(zz,labels,base,fit_ix),calibration=metrics(zz,labels,base,cal_ix or fit_ix),
              represented_std_median=float(reps.flatten(0,1).std(0).median()),elapsed_s=time.perf_counter()-start_time)
            history.append(record);torch.save(dict(state=current,initial=initial,room=room,method=method,seed=seed,step=step,checkpoint_sha256=sha(cp),fit_identifiers=sorted({rows[i]['identifier'] for i in fit_ix})),folder/f'step{step}.pt')
            print('SSL_SNAPSHOT',room,method,seed,step,'cal_nll',round(nll,5),'fit_correct',record['fit']['correct'],'seconds',round(record['elapsed_s'],1),flush=True)
    assert frozen_hash(student)==frozen
    if room=='actual':
        # Only internal calibration choices from proxy fits set the final step.
        proxy_steps=[read(WORK/'fits'/f'{r}_{method}_s{seed}'/'complete.json')['chosen_step'] for r in ROUTES if r!='actual']
        median=float(np.median(proxy_steps));chosen=min(snapshots,key=lambda k:abs(k-median))
    else:chosen=min(history,key=lambda v:v['calibration_nll'])['step']
    final=snapshots[chosen];variants=[]
    for amount in (.25,.5,1.):
        interpolated(student,initial,final,amount);zz,raw,_=scores(student,source,bias,T)
        report=dict(amount=amount,metrics=metrics(zz,labels,base,test_ix));variants.append(report)
        np.savez_compressed(folder/f'predictions_{amount}.npz',logits=zz.numpy(),raw=raw.numpy(),labels=labels.numpy(),base=base.numpy(),test_indices=test_ix)
    save(folder/'history.json',history)
    complete=dict(room=room,method=method,seed=seed,chosen_step=chosen,updated_encoder_parameters=trainable,
      frozen_decoder_sha256=frozen,initial_probability_parity_error=parity,source_sampler_sha256=sampler_hash.hexdigest(),
      fit_identifiers=sorted({rows[i]['identifier'] for i in fit_ix}),calibration_identifiers=sorted({rows[i]['identifier'] for i in cal_ix}),test_identifiers=sorted({rows[i]['identifier'] for i in test_ix}),paired_room_support=pair_rooms,
      variants=variants,elapsed_s=time.perf_counter()-start_time,platform_score=None)
    save(folder/'complete.json',complete);print('SSL_FIT_DONE',room,method,seed,'chosen',chosen,'net',[v['metrics']['corrected']-v['metrics']['broken'] for v in variants],flush=True)
    del source,gpu,student,optimizer;gc.collect();torch.cuda.empty_cache()
    return complete

def main():
    p=argparse.ArgumentParser();p.add_argument('--room',default='all');p.add_argument('--methods',default='supervised,source_ssl,official_u');p.add_argument('--seeds',default='42,2026');p.add_argument('--wait-cache',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(methods=['supervised','source_ssl','official_u','paired','paired_u'],seeds=[42,2026],steps=[40,120,240],source_batch=16,ssl_batch=32,
      encoder_lr=5e-5,projector_lr=5e-4,ssl_weight=.02,teacher_kl_weight=.1,source_latent_drift_weight=.02,updated=['phase.project','temporal.blocks.2','temporal.project'],
      fixed='All other weights eval/frozen. Existing calibration temperature/bias and final+0.35 unchanged. No pseudo labels/external data/U class prior fitting.',
      sample='Source balanced class/room, same sequence acrossB/C/D. U filename group then record then window balancing; unknown bucket not asserted one physical room.',
      selection='Default parameter interpolation0.5, compare0.25/1 as fixed ablations. Prefer U over matched source-only if both seeds aggregate netpositive, >=2 improving rooms and no room>2pp loss. Inspect class/source degradation, never claim source accuracy is hidden accuracy.',
      fallback='If full rerouting unstable, evaluate separately limited old0->new1 with fixed two-time/receiver view agreement. Do not change original1 outputs or refit bias.',
      added_pair_stage='After initial30 fits showed1 recall pressure and no rescue triggers, compare same-room0/1 pair gradients on updated1.512M encoder suffixes with/withoutU. Extra binaryCEweight0.3, within-room softplus margin1 weight0.1,8pairs/step, no fake missing-class pairs.',
      caveat='Historical outer rooms reused for development. U does not include outer room labels or CSI; target same-room assumptions not made.'))
    rooms=('a1','c1','g1','e1','actual') if a.room=='all' else (a.room,)
    for room in rooms:
        while a.wait_cache and not (WORK/'prefixes'/room/'complete.json').exists():time.sleep(5)
        assert (WORK/'prefixes'/room/'complete.json').exists(),room
        for seed in map(int,a.seeds.split(',')):
            for method in a.methods.split(','):fit(room,method,seed)

if __name__=='__main__':main()
