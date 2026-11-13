"""Frozen temporal multiscale/noise marginalization; no fitting or new biases."""
import gc,json,sys,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('adapter049_shared',ROOT/'experiments/adapter049_20261024/run.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
AdapterHead,policy,build,permutation_indices,sha=adapter.AdapterHead,adapter.policy,adapter.build,adapter.permutation_indices,adapter.sha
save,read=adapter.save,adapter.read
WORK=Path('D:/huawei2026-experiments/multiscale049_20261024')
OLD=Path('D:/huawei2026-experiments/protected_context_20261024')
LATENTS=Path('D:/huawei2026-experiments/adapter049_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
METHODS=('time_smooth','frequency_smooth','local100','local120')

def temporal_views(t,method):
    a=t[:,384:768].reshape(-1,3,32,4,150).mean(3)
    p=t[:,768:960].reshape(-1,6,32,150)
    clock=t[:,960:]
    if method in ('time_smooth','frequency_smooth'):
        def smooth(x):
            axis=-1 if method=='time_smooth' else -2
            if axis==-2:x=x.transpose(-1,-2)
            shape=x.shape;x=x.reshape(-1,1,shape[-1]);x=F.pad(x,(1,1),mode='replicate')
            x=F.conv1d(x,x.new_tensor([.25,.5,.25]).reshape(1,1,3)).reshape(shape)
            if axis==-2:x=x.transpose(-1,-2)
            return x
        return [(smooth(a),smooth(p),clock)]
    width=100 if method=='local100' else 120;result=[]
    for start in (0,150-width):
        def resize(x):
            shape=x.shape
            return F.interpolate(x[...,start:start+width].reshape(-1,1,width),size=150,mode='linear',align_corners=True).reshape(*shape[:-1],150)
        # Absolute elapsed time is rebased; actual time span retained. Step/gap
        # interpolate with selected observations rather than inventing regular5ms.
        c=resize(clock);c=torch.cat([c[:,:1]-c[:,:1,:1],c[:,1:]],1)
        result.append((resize(a),resize(p),c))
    return result

def predict(model,head,t,s,latent,method,strength):
    alternatives=torch.stack([model.temporal(a,p,c) for a,p,c in temporal_views(t,method)],1).mean(1)
    e=latent.clone();e[:,3]=(1-strength)*latent[:,3]+strength*alternatives
    return head(e,s,0.)[0]

def metric(z,y,base,indices):
    ix=torch.tensor(indices);pred=z[ix].argmax(-1);bp=base[ix].argmax(-1);yy=y[ix]
    cm=torch.bincount(yy*5+pred,minlength=25).reshape(5,5)
    return dict(correct=int((pred==yy).sum()),total=len(ix),corrected=int(((pred==yy)&(bp!=yy)).sum()),broken=int(((pred!=yy)&(bp==yy)).sum()),confusion_matrix=cm.tolist())

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;WORK.mkdir(parents=True,exist_ok=True)
    save(WORK/'plan.json',dict(methods=list(METHODS),strengths=[.25,.5,1.],training=False,
      rationale='Replace only temporal evidence with fixed denoised or two local-time encodings, blend latent evidence. No new fit, class bias, external data or pseudo labels.',
      control='Exact scored049 inherited weights. Each room-excluded proxy uses own frozen teacher calibration.',
      selection='Require positive aggregate independent-room net and no source-class degradation for submission consideration; avoid source-only peak selection.',
      caveat='Local crops are feature-space temporal views, not newly normalized raw CSI windows; inherit original window statistics. Reused development rooms, not untouched tests.'))
    d=torch.load(CACHE,weights_only=True);rev=torch.load(OLD/'reverse_features.pt',weights_only=True);assert d['rows']==rev['rows']
    index,sign,summary_index=[v.cuda() for v in permutation_indices((2,1,0))]
    reports=[]
    for room in ('actual','a1','c1','g1','e1'):
        actual=room=='actual'
        checkpoint=ROOT/'models/best_037/final.pt' if actual else OLD/'fits'/f'baseline_s42_{room}'/'final.pt'
        latentpath=Path('D:/huawei2026-experiments/evidence049_research_20261024/frozen_latents.pt') if actual else LATENTS/f'{room}_latents.pt'
        latent=torch.load(latentpath,weights_only=True)
        if not actual:assert latent['checkpoint_sha256']==sha(checkpoint)
        model,bias=build(checkpoint,full=actual);model=model.cuda().eval();head=AdapterHead(model).cuda().eval()
        if actual:T=1.;test=list(range(960))
        else:
            calibration=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline'];bias=torch.tensor(calibration['bias']);T=calibration['temperature']
            manifest=read(OLD/'results'/f's42_{room}'/'manifest.json');ids={r['identifier'] for r in manifest['test']};test=[i for i,r in enumerate(d['rows']) if r['identifier'] in ids]
        base=policy(latent['logits'],bias,T);yy=d['y']
        for method in METHODS:
            out={s:[] for s in (.25,.5,1.)}
            with torch.no_grad():
                for start in range(0,960,12):
                    end=start+12;t,s=d['t'][start:end].cuda(),d['s'][start:end].cuda();rt,rs=rev['t'][start:end].cuda(),rev['s'][start:end].cuda()
                    views=[(t,s),(rt,rs),(t.index_select(1,index)*sign,s.index_select(1,summary_index)),(rt.index_select(1,index)*sign,rs.index_select(1,summary_index))]
                    chunks={s:[] for s in out}
                    for k,(tt,ss) in enumerate(views):
                        e=latent['latents'][start:end,k].cuda()
                        alternative=torch.stack([model.temporal(a,p,c) for a,p,c in temporal_views(tt,method)],1).mean(1)
                        for strength in out:
                            ev=e.clone();ev[:,3]=(1-strength)*e[:,3]+strength*alternative
                            chunks[strength].append(head(ev,ss,0.)[0].cpu())
                    for strength in out:out[strength].append(torch.stack(chunks[strength],1))
            for strength in out:
                raw=torch.cat(out[strength]);z=policy(raw,bias,T)
                result=dict(room=room,method=method,strength=strength,metrics=metric(z,yy,base,test));reports.append(result)
                np.savez_compressed(WORK/f'{room}_{method}_{strength}.npz',raw=raw.numpy(),logits=z.numpy(),labels=yy.numpy(),test_indices=test)
                print('MULTISCALE',result,flush=True)
            save(WORK/'progress.json',reports)
        del model,head;gc.collect();torch.cuda.empty_cache()
    save(WORK/'results.json',reports)
    summary=[]
    for method in METHODS:
        for strength in (.25,.5,1.):
            r=[r for r in reports if r['room']!='actual' and r['method']==method and r['strength']==strength]
            source=next(r for r in reports if r['room']=='actual' and r['method']==method and r['strength']==strength)
            summary.append(dict(method=method,strength=strength,proxy_net=sum(x['metrics']['corrected']-x['metrics']['broken'] for x in r),
              per_room_net={x['room']:x['metrics']['corrected']-x['metrics']['broken'] for x in r},source=source['metrics']))
    save(WORK/'summary.json',summary);print('FINAL_SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()
