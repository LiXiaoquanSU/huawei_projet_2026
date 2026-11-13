"""Verify actual original-storage rounding of selected paired encoder weights."""
import gc,time
from g60_common import *
def updated_state(room,seed=42):
    folder=Path('D:/huawei2026-experiments/u_ssl_20261025/fits')/f'{room}_paired_s{seed}';fit=read(folder/'complete.json');path=folder/f"step{fit['chosen_step']}.pt";cp=torch.load(path,weights_only=True)
    half=set(precision_spec()[0]);state={};mapping={}
    for k,v in cp['state'].items():
        name=k.replace('phase_project.','phase.project.').replace('time_block.','temporal.blocks.2.').replace('time_project.','temporal.project.')
        a=cp['initial'][k]*.75+v*.25;state[name]=a.half().float() if name in half else a;mapping[name]=k
    return state,dict(checkpoint=str(path),sha256=sha(path),seed=seed,step=fit['chosen_step'],interpolation=.25,rounded_to_original_fp16_storage_keys=sorted(set(state)&half))

def main():
    torch.set_num_threads(4);results=[]
    save(WORK/'quantized_plan.json',dict(method='paired12_42_storage',purpose='Single seed42 direct replacement in054 weight storage, avoid suffix duplication/runtime growth. Back off from0.5 to fixed0.25 interpolation because0.5 lost6correct4person source windows despite improving all4proxy rooms. Exact original fp16/fp32 storage types. Do not relax source-class eligibility; record tradeoff if still fails. No new training/calibration.',source54_sha256=SOURCE_SHA))
    for room in ('a1','c1','g1','e1','actual'):
        path=WORK/f'{room}_paired12_42_storage.npz';d=torch.load(WORK/f'{room}.pt',weights_only=True);b,T=calibration(room);base=policy(d['raw'],b,T);_,_,ti=split(room,d['rows'])
        if path.exists():
            with np.load(path) as f:raw=torch.from_numpy(f['raw'].copy());z=torch.from_numpy(f['logits'].copy())
        else:
            data=torch.load(CACHE,weights_only=True);reverse=torch.load(OLD/'reverse_features.pt',weights_only=True);model,_=build(checkpoint(room),full=room=='actual');state,provenance=updated_state(room);assert sha(checkpoint(room))==d['checkpoint_sha256'];model.load_state_dict(state,strict=False);model.cuda().eval();raw=[];started=time.perf_counter()
            with torch.inference_mode():
                for permutation in PERMS:
                    idx,sign,si=[x.cuda() for x in permutation_indices(permutation)];pred=[]
                    for offset in range(0,960,16):
                        t=torch.stack([data['t'][offset:offset+16],reverse['t'][offset:offset+16]],1).flatten(0,1).cuda();s=torch.stack([data['s'][offset:offset+16],reverse['s'][offset:offset+16]],1).flatten(0,1).cuda()
                        pred.append(model.forward_features(t.index_select(1,idx)*sign,s.index_select(1,si)).cpu().reshape(-1,2,5))
                    raw.append(torch.cat(pred));print('G60_STORAGE',room,permutation,round(time.perf_counter()-started,1),flush=True)
            raw=torch.stack(raw,1);z=policy(raw,b,T);np.savez_compressed(path,raw=raw.numpy(),logits=z.numpy(),base=base.numpy(),labels=d['labels'].numpy(),test_indices=ti);save(WORK/f'{room}_storage_provenance.json',provenance)
            del model,data,reverse;gc.collect();torch.cuda.empty_cache()
        m=metrics(z,d['labels'],base,ti);results.append(dict(room=room,method='paired12_42_storage',metrics=m));save(WORK/'quantized_results.json',results);print('G60_STORAGE_RESULT',room,m,flush=True)
    source=results[-1]['metrics'];old=next(r for r in read(WORK/'summary.json') if r['method']=='baseline')['source'];rr=results[:-1]
    net={r['room']:r['metrics']['corrected']-r['metrics']['broken'] for r in rr};worst=min((r['metrics']['corrected']-r['metrics']['broken'])/r['metrics']['total'] for r in rr)
    summary=dict(method='paired12_42_storage',room_net=net,proxy_net=sum(net.values()),source=source,worst_room_delta=worst,eligible=sum(net.values())>0 and sum(v>0 for v in net.values())>=2 and worst>=-.02 and all(source['confusion_matrix'][k][k]>=old['confusion_matrix'][k][k] for k in range(5)))
    save(WORK/'quantized_summary.json',summary);print('G60_STORAGE_SUMMARY',summary,flush=True)
if __name__=='__main__':main()
