"""Second ablation: supervise the actual calibrated four-view decision pipeline."""
import gc
import numpy as np
import torch
from run import *

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    extended='--extended' in sys.argv
    methods=('calibrated_aligned','calibrated_dro') if extended else ('calibrated','calibrated_consistent');reports=[]
    tag='extended' if extended else 'calibrated'
    save(WORK/f'{tag}_plan.json',dict(methods=list(methods),reason='Uncalibrated-view training decreased actual class1 recall. Align loss with fixed deployed calibration/policy.',
      loss='0.75 final deployed scores CE +0.25 perview calibrated CE, smoothing.02; fixed.2 raw teacher KL,.01 residual penalty; optional.2 JS',
      selection='Epoch by training-side calibration NLL only. Fixed residualscales0.25,0.5,1 and output mixes0.5,1 reported as development ablation, not independent final test.',
      no_bias_training=True))
    for room in ('a1','c1','g1','e1'):
        cache=torch.load(WORK/f'{room}_latents.pt',weights_only=True)
        manifest=read(OLD/'results'/f's42_{room}'/'manifest.json');ids={k:{r['identifier'] for r in manifest[k]} for k in ('fit','calibration','test')}
        cal=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline'];b=torch.tensor(cal['bias']);T=cal['temperature']
        for method in methods:
            checkpoint=OLD/'fits'/f'baseline_s42_{room}'/'final.pt'
            head,states,history,chosen,base=fit(cache,checkpoint,ids['fit'],ids['calibration'],method,room,b,T)
            torch.save(dict(adapter=states[chosen],method=method,step=chosen,scale=1.,fit_identifiers=sorted(ids['fit'])),WORK/f'proxy_{room}_{method}.pt')
            head.load_state_dict(states[chosen],strict=False)
            test_ix=[i for i,r in enumerate(cache['rows']) if r['identifier'] in ids['test']]
            variants=[]
            with torch.no_grad():
                for scale in (.25,.5,1.):
                    raw,_=head(cache['latents'].cuda().flatten(0,1),cache['summaries'].cuda().flatten(0,1),scale)
                    z=policy(raw.reshape(-1,4,5),b.cuda(),T)
                    for mix in (.5,1.):
                        zz=((1-mix)*base.softmax(-1)+mix*z.softmax(-1)).clamp_min(1e-8).log()
                        variants.append(dict(scale=scale,mix=mix,test=metrics(zz,cache['labels'].cuda(),base,test_ix)))
            result=dict(room=room,method=method,chosen_step=chosen,history=history,variants=variants);reports.append(result)
            save(WORK/'calibrated_folds'/f'{room}_{method}.json',result)
            print('CALIBRATED_PROXY',room,method,chosen,[(r['scale'],r['mix'],r['test']['corrected'],r['test']['broken']) for r in variants],flush=True)
            del head;gc.collect();torch.cuda.empty_cache()
    save(WORK/f'{tag}_proxy_results.json',reports)
    cache=torch.load(Path('D:/huawei2026-experiments/evidence049_research_20261024/frozen_latents.pt'),weights_only=True)
    checkpoint=ROOT/'models/best_037/final.pt';model,b=build(checkpoint);del model
    ids={r['identifier'] for r in cache['rows']};full=[]
    for method in methods:
        head,states,history,_,base=fit(cache,checkpoint,ids,set(),method,'actual',b,1.)
        median=float(np.median([r['chosen_step'] for r in reports if r['method']==method]));step=min(states,key=lambda k:abs(k-median))
        head.load_state_dict(states[step],strict=False)
        for scale in (.25,.5,1.):
            cp=dict(adapter=states[step],method=method,step=step,scale=scale,fit_identifiers=sorted(ids),history=history)
            torch.save(cp,WORK/f'{method}_{scale}.pt')
            with torch.no_grad():
                raw,_=head(cache['latents'].cuda().flatten(0,1),cache['summaries'].cuda().flatten(0,1),scale)
                raw=raw.reshape(-1,4,5);z=policy(raw,b.cuda())
                for mix in (.5,1.):
                    zz=((1-mix)*base.softmax(-1)+mix*z.softmax(-1)).clamp_min(1e-8).log()
                    result=dict(method=method,step=step,scale=scale,mix=mix,source_diagnostic=metrics(zz,cache['labels'].cuda(),base,list(range(960))))
                    full.append(result);print('CALIBRATED_ACTUAL',result,flush=True)
                    np.savez_compressed(WORK/f'{method}_{scale}_mix{mix}_predictions.npz',logits=zz.cpu().numpy(),raw=raw.cpu().numpy(),labels=cache['labels'].numpy())
        del head;gc.collect();torch.cuda.empty_cache()
    save(WORK/f'{tag}_actual_results.json',full)

if __name__=='__main__':main()
