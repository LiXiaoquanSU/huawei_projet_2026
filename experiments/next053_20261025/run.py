"""Exact preprocessing controls: selective transfer, scored opinions, frequency mirror."""
import gc,time
from n53_common import *
sys.path.insert(0,str(ROOT/'experiments/u_ssl_20261025'))
from ssl_common import Student
from quantized import updated_state

def predict(room):
    path=WORK/f'{room}.pt'
    if path.exists():return torch.load(path,weights_only=True)
    model,b=old.build(old.checkpoint(room),full=room=='actual');model.cuda().eval();student=Student(model).cuda().eval();state,provenance=updated_state(room)
    mapping={k.replace('phase.project.','phase_project.').replace('temporal.blocks.2.','time_block.').replace('temporal.project.','time_project.'):v for k,v in state.items()};student.load_state_dict(mapping,strict=False)
    rows=read(WORK/'rows.json');fi,_,_=old.split(room,rows)
    if room!='actual':
        cp=torch.load(old.checkpoint(room),weights_only=True);assert set(cp['fit_identifiers'])=={rows[i]['identifier'] for i in fi};del cp
    tt=np.load(WORK/'t.npy',mmap_mode='r');ss=np.load(WORK/'s.npy',mmap_mode='r');captured={};handles=[]
    def pre(name):
        def hook(m,args):captured[name]=args[0]
        return hook
    def post(name):
        def hook(m,args,out):captured[name]=out
        return hook
    handles.extend([model.phase.project.register_forward_pre_hook(pre('p')),model.temporal.blocks[-1].register_forward_pre_hook(pre('t'))])
    for name in ('static','dynamic','phase','temporal'):handles.append(getattr(model,name).register_forward_hook(post(name)))
    raw=[];modified=[];mirrored=[];started=time.perf_counter()
    with torch.inference_mode():
        for permutation in PERMS:
            idx,sign,si=[x.cuda() for x in old.permutation_indices(permutation)];z0=[];z1=[];zm=[]
            for offset in range(0,960,16):
                t=torch.from_numpy(np.array(tt[offset:offset+16],copy=True)).cuda().flatten(0,1);s=torch.from_numpy(np.array(ss[offset:offset+16],copy=True)).cuda().flatten(0,1)
                t=t.index_select(1,idx)*sign;s=s.index_select(1,si);base=model.forward_features(t,s)
                e=torch.stack([captured[k] for k in ('static','dynamic','phase','temporal')],1);adapted,_=student(captured['p'],captured['t'],e,s)
                mirror=model.forward_features(mirror_features(t),s)
                z0.append(base.cpu().reshape(-1,2,5));z1.append(adapted.cpu().reshape(-1,2,5));zm.append(mirror.cpu().reshape(-1,2,5))
            raw.append(torch.cat(z0));modified.append(torch.cat(z1));mirrored.append(torch.cat(zm));print('N53_INFER',room,permutation,round(time.perf_counter()-started,1),flush=True)
    out=dict(raw=torch.stack(raw,1),student=torch.stack(modified,1),mirror=torch.stack(mirrored,1),labels=torch.tensor([r['headcount'] for r in rows]),rows=rows,checkpoint_sha256=sha(old.checkpoint(room)),student_provenance=provenance,preprocessing_sha256=read(WORK/'features_complete.json')['preprocessor_sha256'])
    torch.save(out,path)
    for h in handles:h.remove()
    del model,student;gc.collect();torch.cuda.empty_cache();return out

def main():
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    save(WORK/'comparison_plan.json',dict(methods=['baseline','gnp_mean','opinion_mean','opinion_median','ordinal_logit','transfer_all_mean','transfer_low_guard','transfer_binary01','transfer_presence','mirror25','mirror50'],
      primary_controls='exact054preprocessing;4teachersfitlabels exclude outer room andcalibration;fixed methods,no biasfitting;source labels aretrained diagnostics',
      gcal_caveat='Opinion policies containing actual teammateGcal Ubiasedtable are transfer stress tests, not fully independent room validation: table depends on actual full-source teacher. Do not use their proxy gain as independent evidence.',
      transfer='reuse prior quarterseed42encoder weights; no new training; update lowcount or presence only so teacher2+outputs can be fixed',
      mirror='feature frequency-axis reversal, keep timing/summary; not asserted physical rawCSI invariance; additionalviews onlyfallback; amount.25/.5 fixed',
      selection='Prefer >=2 improving rooms, positiveaggregate,no>2pp roomloss,source4-person no regression. If none record failures; do not claim060. Opinioncal controls flagged dependent.'))
    results=[]
    for room in ('a1','c1','g1','e1','actual'):
        d=predict(room);b,T=old.calibration(room);raw=d['raw'];base=old.policy(raw,b,T);modified=old.policy(d['student'],b,T);_,_,ti=old.split(room,d['rows'])
        zz={m:pool(raw,b,T,m) for m in ('baseline','gnp_mean','opinion_mean','opinion_median','ordinal_logit')}
        for name in ('all_mean','low_guard','binary01','presence'):zz['transfer_'+name]=transfer(base,modified,name)
        zz.update(mirror25=spectral(raw,d['mirror'],b,T,.25),mirror50=spectral(raw,d['mirror'],b,T,.5))
        for method,z in zz.items():
            metric=old.metrics(z,d['labels'],base,ti);results.append(dict(room=room,method=method,metrics=metric));np.savez_compressed(WORK/f'{room}_{method}.npz',logits=z.numpy(),base=base.numpy(),labels=d['labels'].numpy(),test_indices=ti)
            print('N53_RESULT',room,method,metric['corrected'],metric['broken'],flush=True)
        save(WORK/'results.json',results)
    baseline=next(x for x in results if x['room']=='actual' and x['method']=='baseline')['metrics'];summary=[]
    for method in zz:
        rr=[x for x in results if x['room']!='actual' and x['method']==method];source=next(x for x in results if x['room']=='actual' and x['method']==method)['metrics'];net={x['room']:x['metrics']['corrected']-x['metrics']['broken'] for x in rr};worst=min((x['metrics']['corrected']-x['metrics']['broken'])/x['metrics']['total'] for x in rr)
        independent=method not in ('opinion_mean','opinion_median','ordinal_logit');eligible=independent and sum(net.values())>0 and sum(v>0 for v in net.values())>=2 and worst>=-.02 and source['confusion_matrix'][4][4]>=baseline['confusion_matrix'][4][4]
        summary.append(dict(method=method,source=source,room_net=net,proxy_net=sum(net.values()),worst_room_delta=worst,primary_proxy_independent=independent,eligible=eligible))
    save(WORK/'summary.json',summary);print('N53_SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
