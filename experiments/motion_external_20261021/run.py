"""Reproducible room-disjoint motion-adapter screen; existing results never overwritten."""
import argparse
import gc
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from net import ROOT, MotionModel, Submission, VARIANTS, fresh
from model import signal_targets
from utils import task_loss,write_csv

WORK=Path('D:/huawei2026-experiments/motion_external_20261021')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
FOLDS=['a1','c1','g1']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def seed_training():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def load_data():
    d=torch.load(CACHE,map_location='cpu',weights_only=True);d.pop('absolute',None)
    for name,value in d['source_hashes'].items():assert sha(ROOT/'src'/name)==value
    return d

def evaluate(model,d,indices):
    model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(indices),32):
            ix=indices[start:start+32]
            values.append(model.forward_features(d['t'][ix].cuda(),d['s'][ix].cuda()).cpu())
    z=torch.cat(values);y=d['y'][indices];cm=torch.zeros(5,5,dtype=torch.long)
    for a,b in zip(y.tolist(),z.argmax(1).tolist()):cm[a,b]+=1
    return dict(accuracy=float((z.argmax(1)==y).float().mean()),
                nll=float(torch.nn.functional.nll_loss(z,y)),confusion_matrix=cm.tolist()),z

def fit(d,variant,seed,fold,transfer=None):
    key=f'{variant}_s{seed}_{fold}';folder=WORK/'fits'/key
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    folder.mkdir(parents=True,exist_ok=True)
    rows=d['rows'];ti=list(range(960)) if fold=='full' else [i for i,r in enumerate(rows) if r['room'] not in ['unknown',fold]]
    vi=[] if fold=='full' else [i for i,r in enumerate(rows) if r['room']==fold]
    ids={rows[i]['identifier'] for i in ti};assert ids.isdisjoint({rows[i]['identifier'] for i in vi})
    model=fresh('baseline' if transfer else variant,seed)
    if transfer:
        state=torch.load(transfer,map_location='cpu',weights_only=True)['encoder']
        target=model.static.state_dict()
        for name,v in state.items():
            if v.shape!=target[name].shape:
                assert name in ['layers.0.body.0.weight','layers.0.skip.weight']
                v=v.repeat(1,3,1,1)/3
            assert v.shape==target[name].shape
            target[name]=v
        model.static.load_state_dict(target)
    config=dict(variant=variant,seed=seed,fold=fold,epochs=5,lr=.0004,weight_decay=.01,batch_size=32,
                train_records=len(ids),train_windows=len(ti),validation_records=len({rows[i]['identifier'] for i in vi}),
                initialization='fresh random' if not transfer else 'external single-Rx static encoder; other weights fresh',
                transfer_checkpoint=str(transfer) if transfer else None,parameter_count=sum(p.numel() for p in model.parameters()),
                official_U=False,training_seed=42,sampling_seed=42,platform_score=None)
    save(folder/'config.json',config);write_csv(folder/'train_manifest.csv',[rows[i] for i in ti])
    if vi:write_csv(folder/'validation_manifest.csv',[rows[i] for i in vi])
    model=model.cuda();seed_training()
    groups={};domain=lambda r:r['identifier'] if r['room']=='unknown' else r['room']
    for i in ti:
        k=(int(d['y'][i]),domain(rows[i]));groups[k]=groups.get(k,0)+1
    counts={c:len({g for label,g in groups if label==c}) for c in range(5)}
    weights=[1/(groups[(int(d['y'][i]),domain(rows[i]))]*counts[int(d['y'][i])]) for i in ti]
    sampler=WeightedRandomSampler(weights,len(ti),generator=torch.Generator().manual_seed(42))
    target=signal_targets(d['t'][ti]);loader=DataLoader(TensorDataset(d['t'][ti],d['s'][ti],d['y'][ti],target,torch.arange(len(ti))),batch_size=32,sampler=sampler)
    opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01);history=[];seen=set()
    print('START',key,flush=True)
    for epoch in range(1,6):
        start=time.perf_counter();model.train();losses=[];correct=total=0
        for t,s,y,target,local in loader:
            seen.update(local.tolist());t,s,y,target=t.cuda(),s.cuda(),y.cuda(),target.cuda()
            opt.zero_grad(set_to_none=True);tasks=model.forward_tasks(t,s);loss,_=task_loss(tasks,y,target,'physical')
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3)
            assert torch.isfinite(norm);opt.step();losses.append(loss.item())
            correct+=int((tasks['logits'].argmax(1)==y).sum());total+=len(y)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),sampled_train_accuracy=correct/total,
                            distinct_windows=len(seen),seconds=time.perf_counter()-start))
        write_csv(folder/'history.csv',history)
    val,z=evaluate(model,d,vi) if vi else (None,None)
    if vi:write_csv(folder/'predictions.csv',[dict(**rows[i],prediction=int(z[j].argmax()),**{f'log_probability_{k}':float(v) for k,v in enumerate(z[j])}) for j,i in enumerate(vi)])
    result=dict(config=config,validation=val,distinct_train_windows=len(seen),
                sampled_train_accuracy=history[-1]['sampled_train_accuracy'],platform_score=None)
    torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),folder/'final.pt')
    save(folder/'metrics.json',result);print('DONE',key,val,flush=True)
    del model,opt,loader;gc.collect();torch.cuda.empty_cache()
    return result

def summary(variant,seed):
    rr=[read(WORK/'fits'/f'{variant}_s{seed}_{f}'/'metrics.json') for f in FOLDS]
    cm=np.asarray([r['validation']['confusion_matrix'] for r in rr]).sum(0)
    assert (cm.sum(1)>0).all()
    mean=float(np.mean([r['validation']['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    return dict(variant=variant,seed=seed,room_mean_accuracy=mean,macro_recall=macro,selection_score=min(mean,macro),
                worst_room_accuracy=min(r['validation']['accuracy'] for r in rr),
                mean_nll=float(np.mean([r['validation']['nll'] for r in rr])),confusion_matrix=cm.tolist(),
                rooms={f:r['validation']['accuracy'] for f,r in zip(FOLDS,rr)})
def rank(v):return (-v['selection_score'],-v['worst_room_accuracy'],v['mean_nll'],v['variant'])

def main(args):
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available()
    WORK.mkdir(parents=True,exist_ok=True)
    snapshot=WORK/'source_snapshot'
    for p in [*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'PLAN.md']:
        target=snapshot/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(p),f'Source changed: {p}'
        else:shutil.copy2(p,target)
    d=load_data()
    if args.stage=='official':
        for variant in VARIANTS:
            for fold in FOLDS:fit(d,variant,42,fold)
        screened=[summary(v,42) for v in VARIANTS];save(WORK/'screen.json',screened)
        selected=sorted([r for r in screened if r['variant'] not in ['baseline','capacity_control']],key=rank)[:2]
        save(WORK/'selection_stage1.json',dict(rule='PLAN.md',selected=selected))
        for variant in ['baseline',*[r['variant'] for r in selected]]:
            for fold in FOLDS:fit(d,variant,2026,fold)
        confirmed=[summary(v,s) for v in ['baseline',*[r['variant'] for r in selected]] for s in [42,2026]]
        save(WORK/'confirmation.json',confirmed)
        for r in selected:fit(d,r['variant'],42,'full')
    else:
        assert args.transfer and args.variant
        for fold in FOLDS:fit(d,args.variant,42,fold,args.transfer)
        save(WORK/f'{args.variant}_summary.json',summary(args.variant,42))
    print('STAGE_COMPLETE',args.stage,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['official','transfer'],default='official')
    p.add_argument('--variant');p.add_argument('--transfer',type=Path)
    main(p.parse_args())
