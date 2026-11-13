"""Factorial control: previously trained encoder suffixes with12-view054 routing."""
import gc,importlib.util,time
from g60_common import *
spec=importlib.util.spec_from_file_location('g60_previous_ssl',ROOT/'experiments/u_ssl_20261025/ssl_common.py')
ssl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssl)

def predict(room):
    dest=WORK/f'{room}_paired12.pt'
    if dest.exists():return torch.load(dest,weights_only=True)
    data=torch.load(CACHE,weights_only=True);reverse=torch.load(OLD/'reverse_features.pt',weights_only=True);assert data['rows']==reverse['rows']
    model,b=build(checkpoint(room),full=room=='actual');model.cuda().eval();students=[];provenance=[]
    for seed in (42,2026):
        folder=ssl.WORK/'fits'/f'{room}_paired_s{seed}';complete=read(folder/'complete.json');path=folder/f"step{complete['chosen_step']}.pt";cp=torch.load(path,weights_only=True)
        assert cp['checkpoint_sha256']==sha(checkpoint(room));fi,_,_=split(room,data['rows']);assert set(cp['fit_identifiers'])=={data['rows'][i]['identifier'] for i in fi}
        student=ssl.Student(model).cuda().eval();student.load_state_dict({k:(cp['initial'][k]+v)*.5 for k,v in cp['state'].items()},strict=False);students.append(student)
        provenance.append(dict(seed=seed,trained_checkpoint=str(path),sha256=sha(path),step=complete['chosen_step'],amount=.5))
    captured={};handles=[]
    def pre(name):
        def hook(m,args):captured[name]=args[0]
        return hook
    def post(name):
        def hook(m,args,out):captured[name]=out
        return hook
    handles+=[model.phase.project.register_forward_pre_hook(pre('phase_input')),model.temporal.blocks[-1].register_forward_pre_hook(pre('time_input'))]
    for name in ('static','dynamic','phase','temporal'):handles.append(getattr(model,name).register_forward_hook(post(name)))
    raw=[[],[]];base=[];started=time.perf_counter()
    with torch.inference_mode():
        for permutation in PERMS:
            idx,sign,si=[x.cuda() for x in permutation_indices(permutation)];rr=[[],[]];bb=[]
            for offset in range(0,960,16):
                t=torch.stack([data['t'][offset:offset+16],reverse['t'][offset:offset+16]],1).flatten(0,1).cuda();s=torch.stack([data['s'][offset:offset+16],reverse['s'][offset:offset+16]],1).flatten(0,1).cuda()
                t=t.index_select(1,idx)*sign;s=s.index_select(1,si);z=model.forward_features(t,s);bb.append(z.cpu().reshape(-1,2,5))
                evidence=torch.stack([captured[k] for k in ('static','dynamic','phase','temporal')],1)
                for k,student in enumerate(students):
                    zz,_=student(captured['phase_input'],captured['time_input'],evidence,s);rr[k].append(zz.cpu().reshape(-1,2,5))
            base.append(torch.cat(bb))
            for k in range(2):raw[k].append(torch.cat(rr[k]))
            print('G60_COMBINE',room,permutation,round(time.perf_counter()-started,1),flush=True)
    original=torch.load(WORK/f'{room}.pt',weights_only=True);error=float((torch.stack(base,1)-original['raw']).abs().max());assert error<1e-5,error
    out=dict(raw42=torch.stack(raw[0],1),raw2026=torch.stack(raw[1],1),labels=data['y'],rows=data['rows'],provenance=provenance,original_raw_max_error=error)
    torch.save(out,dest)
    for h in handles:h.remove()
    del model,students,data,reverse;gc.collect();torch.cuda.empty_cache();return out

def main():
    torch.set_num_threads(4);save(WORK/'combine_plan.json',dict(change='Reuse previous paired encoder training, apply its fixed0.5 weight interpolation to all12 views and054routing. No new label fitting or claim of newly trained12view encoder.',seeds=[42,2026],source_training='Previously50fits, no external/U in selectedpaired method',comparison='two seeds and probability ensemble; compare exact054 cache in same split; same eligibility as first stage'))
    results=[]
    for room in ('a1','c1','g1','e1','actual'):
        d=predict(room);control=torch.load(WORK/f'{room}.pt',weights_only=True);b,T=calibration(room);base=policy(control['raw'],b,T);_,_,ti=split(room,d['rows'])
        for name,raw in [('paired12_42',d['raw42']),('paired12_2026',d['raw2026']),('paired12',((d['raw42'].exp()+d['raw2026'].exp())*.5).clamp_min(1e-8).log())]:
            z=policy(raw,b,T);m=metrics(z,d['labels'],base,ti);results.append(dict(room=room,method=name,metrics=m));np.savez_compressed(WORK/f'{room}_{name}.npz',logits=z.numpy(),base=base.numpy(),labels=d['labels'].numpy(),test_indices=ti)
            print('G60_COMBINED_RESULT',room,name,m,flush=True)
        save(WORK/'combine_results.json',results)
    summary=[]
    for method in ('paired12_42','paired12_2026','paired12'):
        rr=[r for r in results if r['room']!='actual' and r['method']==method];source=next(r for r in results if r['room']=='actual' and r['method']==method)['metrics'];control=read(WORK/'summary.json');old=next(x for x in control if x['method']=='baseline')['source']
        net={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in rr};worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in rr)
        eligible=sum(net.values())>0 and sum(v>0 for v in net.values())>=2 and worst>=-.02 and all(source['confusion_matrix'][k][k]>=old['confusion_matrix'][k][k] for k in range(5))
        summary.append(dict(method=method,proxy_net=sum(net.values()),room_net=net,source=source,worst_room_delta=worst,eligible=eligible))
    save(WORK/'combine_summary.json',summary);print('G60_COMBINE_SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
