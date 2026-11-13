"""Fixed small policy comparison on historical room-disjoint frozen checkpoints."""
import gc
import json
import time
import shutil
from pathlib import Path

import blosc2
import numpy as np
import torch
from common import ROOT,WORK,OLD,PERMS,POLICIES,ReceiverViews,Submission,build,precision_spec,scores,decide,rows,read,save,sha


def validate_feature_rule(model):
    views=ReceiverViews().cuda();errors=[]
    with np.load(ROOT/'checks/smoke_input.npz') as data:
        x=torch.from_numpy(data['frames'].copy()).cuda()[None].double();dt=torch.from_numpy(data['intervals'].copy()).cuda()[None].double()
    with torch.no_grad():
        t,s=model.preprocess(x,dt);tt,ss=views(t.float(),s.float())
        for k,p in enumerate(PERMS):
            rt,rs=model.preprocess(x[:,:,:,list(p)].contiguous(),dt)
            fe=float((rt.float()[0]-tt[k]).abs().max());se=float((rs.float()[0]-ss[k]).abs().max())
            assert fe<.0001 and se<.0001,(p,fe,se)
            errors.append(dict(permutation=p,feature_error=fe,summary_error=se))
    save(WORK/'permutation_feature_audit.json',dict(status='pass',cases=errors))


def predict_fold(fold,seeds):
    paths=[WORK/'predictions'/f'{fold}_s{s}.npz' for s in seeds]
    if all(p.exists() for p in paths):return
    folders=[OLD/'structure_events_20261021/fits'/f'baseline_s{s}_{fold}' for s in seeds]
    rr=rows(folders[0]/'validation_manifest.csv');tr=rows(folders[0]/'train_manifest.csv')
    assert {r['identifier'] for r in tr}.isdisjoint({r['identifier'] for r in rr})
    assert all(r['room'] not in ('unknown',fold) for r in tr)
    for folder in folders:
        assert rows(folder/'validation_manifest.csv')==rr and rows(folder/'train_manifest.csv')==tr
    models=[build(p/'final.pt')[0].cuda() for p in folders]
    if not (WORK/'permutation_feature_audit.json').exists():validate_feature_rule(models[0])
    views=ReceiverViews().cuda();byid={}
    for index,row in enumerate(rr):byid.setdefault(row['identifier'],[]).append(index)
    out=[np.empty((len(rr),6,2,5),np.float32) for _ in seeds]
    raw=Path('D:/huawei2026-data/labeled');reverse=torch.cat([torch.arange(598,-1,-1),torch.tensor([599])]).cuda()
    start_time=time.perf_counter()
    for identifier,ix in byid.items():
        csi=blosc2.load_array(str(raw/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(raw/f'ATI_{identifier}.b2'))
        for start in range(0,len(ix),2):
            chosen=ix[start:start+2];n=len(chosen)
            x=np.stack([csi[int(rr[j]['start']):int(rr[j]['stop'])].view(np.float32).reshape(600,1,3,1024,2).copy() for j in chosen])
            dt=np.stack([ati[int(rr[j]['start']):int(rr[j]['stop'])].copy() for j in chosen]);dt[:,-1]=0
            x=torch.from_numpy(x).cuda();dt=torch.from_numpy(dt).cuda()
            x=torch.cat([x,x.flip(1)],0).double();dt=torch.cat([dt,dt.index_select(1,reverse)],0).double()
            with torch.no_grad():
                t,s=models[0].preprocess(x,dt);t,s=views(t.float(),s.float())
                for k,model in enumerate(models):
                    p=model.forward_features(t,s).reshape(6,2*n,5)
                    p=torch.stack([p[:,:n],p[:,n:]],dim=2).permute(1,0,2,3).cpu().numpy()
                    out[k][chosen]=p
        del csi,ati
        print('PREDICTED',fold,identifier,'seconds',round(time.perf_counter()-start_time,1),flush=True)
    (WORK/'predictions').mkdir(exist_ok=True)
    for path,p in zip(paths,out):np.savez_compressed(path,log_probability=p,labels=np.array([int(r['headcount']) for r in rr]))
    save(WORK/'predictions'/f'{fold}_manifest.json',dict(validation=rr,train_identifiers=sorted({r['identifier'] for r in tr}),record_disjoint=True,seeds=seeds,checkpoint_hashes=[sha(p/'final.pt') for p in folders]))
    del models,out;gc.collect();torch.cuda.empty_cache()


def evaluate(folds,seeds):
    _,bias,_,_=precision_spec();result=[]
    for seed in seeds:
        for policy in POLICIES:
            rooms=[]
            for fold in folds:
                with np.load(WORK/'predictions'/f'{fold}_s{seed}.npz') as data:raw=torch.from_numpy(data['log_probability'].copy());y=torch.from_numpy(data['labels'].copy())
                q=scores(raw,bias);pred=decide(q,policy).argmax(-1);baseline=decide(q,'light').argmax(-1)
                cm=torch.zeros(5,5,dtype=torch.long)
                for a,b in zip(y,pred):cm[a,b]+=1
                rooms.append(dict(room=fold,accuracy=float((pred==y).float().mean()),confusion_matrix=cm.tolist(),
                                  corrected=int(((baseline!=y)&(pred==y)).sum()),broken=int(((baseline==y)&(pred!=y)).sum())))
            cm=np.sum([r['confusion_matrix'] for r in rooms],axis=0)
            result.append(dict(seed=seed,policy=policy,rooms=rooms,room_mean=float(np.mean([r['accuracy'] for r in rooms])),
                               macro_recall=float(np.mean(np.diag(cm)/cm.sum(1))),worst_room=min(r['accuracy'] for r in rooms)))
    summary=[]
    for policy in POLICIES:
        rr=[x for x in result if x['policy']==policy]
        summary.append(dict(policy=policy,room_mean=float(np.mean([x['room_mean'] for x in rr])),
                            macro_recall=float(np.mean([x['macro_recall'] for x in rr])),worst_room=min(x['worst_room'] for x in rr)))
    return result,summary


def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;assert torch.cuda.is_available()
    WORK.mkdir(exist_ok=True,parents=True)
    plan=dict(policies=list(POLICIES),consensus_rule='at least5/6 views agree with aggregate winner, probability margin >=.20',
              changes='frozen precision-matched baseline networks; no training or bias fitting',bias='fixed actual teammate044 vector, common across every policy and fold; not estimated from heldout labels',
              development='a1,c1,g1; seeds42/2026; historical physical records reused',stress='e1 seed42 after selection',
              selection='maximize min(two-seed mean room accuracy,macro recall), then worst-room; baselines included',
              eligibility='a challenger must beat best light/full proxy accuracy and not reduce macro recall; otherwise export faithful efficient light only',
              limitations='historical retrained baseline checkpoints are proxies, not independently heldout predictions of actual .47 weights')
    save(WORK/'plan.json',plan)
    for fold in ('a1','c1','g1'):predict_fold(fold,[42,2026])
    result,summary=evaluate(('a1','c1','g1'),(42,2026))
    baselines=[x for x in summary if x['policy'] in ('light','full')];baseline=max(baselines,key=lambda x:x['room_mean'])
    eligible=[x for x in summary if x['policy'] not in ('light','full') and x['room_mean']>baseline['room_mean'] and x['macro_recall']>=baseline['macro_recall']]
    selected=max(eligible,key=lambda x:(min(x['room_mean'],x['macro_recall']),x['worst_room'])) if eligible else next(x for x in summary if x['policy']=='light')
    selection=dict(policy=selected['policy'],eligible_challengers=len(eligible),results=result,summary=summary,baseline=baseline,plan=plan)
    save(WORK/'selection.json',selection);print('SELECTION',selected['policy'],json.dumps(summary),flush=True)
    predict_fold('e1',[42])
    # Save all stress results without using them to change the selected policy.
    with np.load(WORK/'predictions/e1_s42.npz') as data:raw=torch.from_numpy(data['log_probability'].copy());y=torch.from_numpy(data['labels'].copy())
    bias=precision_spec()[1];q=scores(raw,bias);stress={p:float((decide(q,p).argmax(-1)==y).float().mean()) for p in POLICIES}
    save(WORK/'stress.json',dict(results=stress,note='observed after selection, no policy retuning'))
    snapshot=WORK/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    save(WORK/'complete.json',dict(status='pass',new_training_fits=0,prediction_checkpoints=7,selection_policy=selected['policy']))
    print('COMPLETE',selected['policy'],flush=True)


if __name__=='__main__':main()
