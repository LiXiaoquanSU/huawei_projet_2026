"""Frozen latent evidence diagnostics/preparation; no parameter optimization."""
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('D:/huawei2026-experiments/evidence049_research_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
REVERSE=Path('D:/huawei2026-experiments/protected_context_20261024/reverse_features.pt')
CHECKPOINT=ROOT/'models/best_037/final.pt'
sys.path.insert(0,str(ROOT/'experiments/tta_policy_20261024'))
from common import build,permutation_indices


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    WORK.mkdir(parents=True,exist_ok=True)
    metadata=dict(feature_cache_sha256=sha(CACHE),checkpoint_sha256=sha(CHECKPOINT),
        source_model_sha256=sha(ROOT/'src/model.py'),source_features_sha256=sha(ROOT/'src/features.py'),
        scored049_sha256=sha(ROOT/'models/final_bias_20261024/01_final_bias_one035.onnx'),
        method='Frozen exact inference weights; four cached preprocessed input views; no training',
        views=['original','time_reverse','rx210','rx210_time_reverse'],routes=['static','dynamic','phase','temporal'])
    path=WORK/'frozen_latents.pt'
    if path.exists():
        cache=torch.load(path,map_location='cpu',weights_only=True);assert cache['metadata']==metadata
    else:
        d=torch.load(CACHE,map_location='cpu',weights_only=True);d.pop('absolute',None)
        rev=torch.load(REVERSE,map_location='cpu',weights_only=True)
        assert rev['cache_sha256']==metadata['feature_cache_sha256'] and rev['rows']==d['rows']
        model,bias=build(CHECKPOINT,full=True);model=model.cuda().eval()
        assert all(not module.training for module in model.modules())
        captures={};handles=[]
        def capture(name):
            def hook(module,args,output):captures[name]=output.detach()
            return hook
        for name in metadata['routes']:handles.append(getattr(model,name).register_forward_hook(capture(name)))
        index,signs,summary_index=[v.cuda() for v in permutation_indices((2,1,0))]
        latents=[];summaries=[];logits=[]
        with torch.inference_mode():
            for offset in range(0,len(d['y']),8):
                t,s=d['t'][offset:offset+8].cuda(),d['s'][offset:offset+8].cuda()
                rt,rs=rev['t'][offset:offset+8].cuda(),rev['s'][offset:offset+8].cuda()
                views=[(t,s),(rt,rs),(t.index_select(1,index)*signs,s.index_select(1,summary_index)),
                    (rt.index_select(1,index)*signs,rs.index_select(1,summary_index))]
                vv=[];ss=[];zz=[]
                for tt,summary in views:
                    tasks=model.forward_tasks(tt,summary)
                    vv.append(torch.stack([captures[name] for name in metadata['routes']],1).cpu())
                    ss.append(summary.cpu());zz.append(tasks['logits'].cpu())
                latents.append(torch.stack(vv,1));summaries.append(torch.stack(ss,1));logits.append(torch.stack(zz,1))
        for handle in handles:handle.remove()
        cache=dict(latents=torch.cat(latents),summaries=torch.cat(summaries),logits=torch.cat(logits),
            labels=d['y'],rows=d['rows'],metadata=metadata)
        old=torch.load(Path('D:/huawei2026-experiments/presence_rescue_20261024/actual_evidence.pt'),weights_only=True)
        probability_error=float((cache['logits'].exp()-old['logits'].exp()).abs().max())
        assert probability_error<.001
        save(WORK/'cache_parity.json',dict(previous_reconstruction_probability_max_error=probability_error,
            note='Cached PyTorch reconstruction comparison, not new ONNX validation'))
        torch.save(cache,path)
        del model,d,rev,old;captures.clear();gc.collect();torch.cuda.empty_cache()
        print('FROZEN_CACHE_READY',list(cache['latents'].shape),flush=True)
    rows=cache['rows'];labels=cache['labels'].numpy()
    ids=sorted({r['identifier'] for r in rows if r['room']!='unknown'})
    record_rows=[next(r for r in rows if r['identifier']==identifier) for identifier in ids]
    means=np.stack([cache['latents'][[i for i,r in enumerate(rows) if r['identifier']==identifier],0].numpy().mean(0) for identifier in ids])
    # Record-level descriptors, never treat 20 correlated windows as20 independent rooms.
    yy=np.array([r['headcount'] for r in record_rows]);rooms=np.array([r['room'] for r in record_rows])
    route_results={}
    for k,route in enumerate(metadata['routes']):
        features=means[:,k];features=features-features.mean(0,keepdims=True)
        features=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
        distance=1-features@features.T
        upper=np.triu(np.ones(distance.shape,bool),1)
        masks=dict(same_count_different_room=upper&(yy[:,None]==yy[None,:])&(rooms[:,None]!=rooms[None,:]),
            different_count_same_room=upper&(yy[:,None]!=yy[None,:])&(rooms[:,None]==rooms[None,:]))
        binary=yy<=1
        cross_pred=[];cross_truth=[]
        for i in np.flatnonzero(binary):
            allowed=binary&(rooms!=rooms[i]);ix=np.flatnonzero(allowed);assert len(ix)>0
            winner=ix[np.argmin(distance[i,ix])];cross_pred.append(int(yy[winner]));cross_truth.append(int(yy[i]))
        nearest=distance.copy();np.fill_diagonal(nearest,np.inf);nearest=nearest.argmin(1)
        route_results[route]=dict(
            pair_distances={name:dict(record_pairs=int(mask.sum()),mean=float(distance[mask].mean()),median=float(np.median(distance[mask]))) for name,mask in masks.items()},
            nearest_record_same_room_fraction=float((rooms[nearest]==rooms).mean()),
            nearest_record_same_count_fraction=float((yy[nearest]==yy).mean()),
            binary_crossroom_nearest_count_correct=int((np.array(cross_pred)==np.array(cross_truth)).sum()),
            binary_crossroom_records=len(cross_pred),
            caveat='Descriptors already trained on all source labels; nearest-neighbour statistics are geometry diagnostics, not validation accuracy.')
    # Inspect the previously hypothesized static-to-presence bridge using cached heads.
    with np.load(Path('D:/huawei2026-experiments/final_bias_20261024/cached_exact048.npz')) as values:
        truth=values['labels'];pred=(values['logits']+np.array([0,.35,0,0,0],np.float32)).argmax(1)
    old=torch.load(Path('D:/huawei2026-experiments/presence_rescue_20261024/actual_evidence.pt'),weights_only=True)
    f=old['features'].numpy()[:,:172].reshape(-1,4,43);expert=f[:,0,18:38].reshape(-1,4,5).argmax(-1);presence=f[:,0,10]
    groups={}
    for name,mask in dict(missed_one_as_empty=(truth==1)&(pred==0),correct_empty=(truth==0)&(pred==0),correct_one=(truth==1)&(pred==1)).items():
        groups[name]=dict(windows=int(mask.sum()),presence_quartiles=np.quantile(presence[mask],[.25,.5,.75]).tolist(),
            static_expert_one=int((mask&(expert[:,0]==1)).sum()),static_one_and_presence_low=int((mask&(expert[:,0]==1)&(presence<.5)).sum()),
            expert_one={name:int((mask&(expert[:,k]==1)).sum()) for k,name in enumerate(metadata['routes'])})
    report=dict(metadata=metadata,cache_bytes=path.stat().st_size,records=len(ids),routes=route_results,
        static_presence_hypothesis=groups,
        notes=['No optimizer, training, new candidate export or modification of current best.',
            'Pair distributions confounded by room/class coverage. Do not infer causal room nuisance from geometry alone.',
            'Static expert often predicts1 in true empty rooms; adding its vote to presence is not justified.',
            'Cache can support future frozen-encoder adapter ablations; no adapter has been trained.'])
    save(WORK/'summary.json',report);save(ROOT/'checks/frozen_evidence049_research.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
