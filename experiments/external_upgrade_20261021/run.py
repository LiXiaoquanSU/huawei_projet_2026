"""Room-disjoint supervised external transfer, with matched source-only controls."""
import argparse,csv,gc,hashlib,json,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from net import ROOT,fresh,TemporalBank,bank_input
from prepare import WORK,save
from model import signal_targets
from utils import task_loss,write_csv
FOLDS=['a1','c1','g1']
VARIANTS=['source_control','external_counts','external_multitask','conditional_multitask']
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seed(value=42):random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def weights(rows,labels):
    counts={};domains={};records={}
    for r,y in zip(rows,labels):
        y=int(y);room=r['room'] if r['room']!='unknown' else r.get('identifier',r.get('record'));record=r.get('identifier',r.get('record'))
        counts[(y,room,record)]=counts.get((y,room,record),0)+1
        domains.setdefault(y,set()).add(room);records.setdefault((y,room),set()).add(record)
    w=[]
    for r,y in zip(rows,labels):
        y=int(y);room=r['room'] if r['room']!='unknown' else r.get('identifier',r.get('record'));record=r.get('identifier',r.get('record'))
        w.append(1/len(domains[y])/len(records[(y,room)])/counts[(y,room,record)])
    w=np.asarray(w,dtype=np.float64);return w/w.sum()
def load():
    d=torch.load(CACHE,map_location='cpu',weights_only=True);d.pop('absolute',None)
    for name,v in d['source_hashes'].items():assert sha(ROOT/'src'/name)==v
    external=torch.load(WORK/'external_sequences.pt',map_location='cpu',weights_only=True)
    for source,v in external.items():
        labels=v['activity'] if source=='sharp' else v['y']
        v['weights']=weights(v['rows'],labels)
    return d,external
def source_data(d,indices):
    rows=[dict(d['rows'][i],record=d['rows'][i]['identifier']) for i in indices]
    y=d['y'][indices];return dict(x=bank_input(d['t'][indices]).contiguous(),y=y,rows=rows,weights=weights(rows,y))
def draw(data,rng,n):
    ix=rng.choice(len(data['y']),n,replace=True,p=data['weights']);a=data['x'][ix].cuda();y=data['y'][ix].cuda()
    if a.shape[1]>64:a=a[:,torch.randperm(a.shape[1],device=a.device)[:64]]
    return a,y,ix
def auxiliary(bank,count,activity,data,source,rng,n):
    a,y,ix=draw(data,rng,n);h=bank(a)
    if source=='sharp':
        z=activity(h);target=data['activity'][ix].cuda()
        loss=nn.functional.cross_entropy(z,target)+.2*nn.functional.binary_cross_entropy_with_logits(torch.logsumexp(z[:,1:],1)-z[:,0],y.float())
    else:loss=nn.functional.cross_entropy(count(h),y)
    return loss,ix
def sequence(variant,step):
    if variant=='source_control':return 'source'
    return ['radio','brescia','radio','sharp' if variant!='external_counts' else 'brescia'][step%4]
def pretrain(d,ext,indices,variant,initseed,fold):
    family='external_multitask' if variant=='conditional_multitask' else variant
    key=f'{family}_s{initseed}_{fold if family=="source_control" else "public"}'
    folder=WORK/'pretrain'/key
    if (folder/'metrics.json').exists():return folder/'final.pt'
    folder.mkdir(parents=True,exist_ok=True);seed(initseed)
    bank=TemporalBank().cuda();count=nn.Linear(256,5).cuda();activity=nn.Linear(256,4).cuda()
    source=source_data(d,indices) if family=='source_control' else None
    pool=dict(ext,source=source);params=[*bank.parameters(),*count.parameters(),*activity.parameters()]
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.01);rng=np.random.default_rng(42);history=[];seen={}
    for epoch in range(4):
        losses=[]
        for step in range(32):
            k=sequence(family,step);opt.zero_grad(set_to_none=True)
            loss,ix=auxiliary(bank,count,activity,pool[k],k,rng,32)
            seen.setdefault(k,set()).update(pool[k]['rows'][i].get('identifier',pool[k]['rows'][i]['record']) for i in ix)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(params,3);opt.step();losses.append(loss.item())
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))));print('PRETRAIN',key,history[-1],flush=True)
    torch.save(dict(bank={k:v.cpu() for k,v in bank.state_dict().items()},count={k:v.cpu() for k,v in count.state_dict().items()},activity={k:v.cpu() for k,v in activity.state_dict().items()}),folder/'final.pt')
    info=dict(family=family,seed=initseed,steps=128,batch_size=32,history=history,records_seen={k:len(v) for k,v in seen.items()},source_records=sorted({r['identifier'] for r in source['rows']}) if source else None)
    save(folder/'metrics.json',info)
    for k,v in seen.items():write_csv(folder/f'{k}_manifest.csv',[r for r in pool[k]['rows'] if r.get('identifier',r['record']) in v])
    del bank,count,activity,opt,params,source;gc.collect();torch.cuda.empty_cache();return folder/'final.pt'
def prototype_loss(z,y,rooms):
    vectors=[];labels=[];domains=[]
    for label in torch.unique(y).tolist():
        for room in torch.unique(rooms).tolist():
            if room<0:continue
            take=(y==label)&(rooms==room)
            if int(take.sum())<2:continue
            vectors.append(z[take].mean(0));labels.append(label);domains.append(room)
    if len(vectors)<2:return z.sum()*0
    v=nn.functional.normalize(torch.stack(vectors),dim=1);label=torch.tensor(labels,device=z.device);domain=torch.tensor(domains,device=z.device)
    valid=domain[:,None]!=domain[None,:];positive=valid&(label[:,None]==label[None,:]);active=positive.any(1)
    if not bool(active.any()):return z.sum()*0
    logits=v@v.T/.2
    return (torch.logsumexp(logits.masked_fill(~valid,-10000),1)-torch.logsumexp(logits.masked_fill(~positive,-10000),1))[active].mean()
def evaluate(model,d,indices):
    model.eval();pred=[]
    with torch.no_grad():
        for start in range(0,len(indices),32):
            ix=indices[start:start+32];pred.append(model.forward_features(d['t'][ix].cuda(),d['s'][ix].cuda()).cpu())
    p=torch.cat(pred);y=d['y'][indices];cm=torch.zeros(5,5,dtype=torch.long)
    for a,b in zip(y.tolist(),p.argmax(1).tolist()):cm[a,b]+=1
    return dict(accuracy=float((p.argmax(1)==y).float().mean()),nll=float(nn.functional.nll_loss(p,y)),confusion_matrix=cm.tolist()),p
def fit(d,ext,variant,initseed,fold):
    key=f'{variant}_s{initseed}_{fold}';folder=WORK/'fits'/key
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    ti=list(range(len(d['rows']))) if fold=='full' else [i for i,r in enumerate(d['rows']) if r['room'] not in ('unknown',fold)]
    vi=[] if fold=='full' else [i for i,r in enumerate(d['rows']) if r['room']==fold]
    tr=[d['rows'][i] for i in ti];ids={r['identifier'] for r in tr};assert ids.isdisjoint({d['rows'][i]['identifier'] for i in vi})
    cp=pretrain(d,ext,ti,variant,initseed,fold);model=fresh(initseed)
    state=torch.load(cp,map_location='cpu',weights_only=True);model.bank.load_state_dict(state['bank']);model.bank_count.load_state_dict(state['count']);model.bank_activity.load_state_dict(state['activity']);del state
    folder.mkdir(parents=True,exist_ok=True)
    config=dict(variant=variant,seed=initseed,fold=fold,epochs=5,lr=.0004,weight_decay=.01,batch_size=32,pretrain=str(cp),pretrain_sha256=sha(cp),official_U=False,training_records=len(ids),training_windows=len(ti),validation_records=len({d['rows'][i]['identifier'] for i in vi}),parameter_count=sum(p.numel() for p in model.parameters()),training_seed=42,auxiliary_batch_size=16,auxiliary_weight=.1,bank_official_CE_weight=.1,conditional_weight=.05 if variant=='conditional_multitask' else 0,platform_score=None)
    save(folder/'config.json',config);write_csv(folder/'train_manifest.csv',tr)
    if vi:write_csv(folder/'validation_manifest.csv',[d['rows'][i] for i in vi])
    model.cuda();seed();rng=np.random.default_rng(42);auxrng=np.random.default_rng(43)
    source=source_data(d,ti) if variant=='source_control' else None;pool=dict(ext,source=source)
    opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01)
    ww=weights(tr,d['y'][ti]);record_indices={r['identifier']:[j for j,v in enumerate(tr) if v['identifier']==r['identifier']] for r in tr}
    roommap={r:i for i,r in enumerate(sorted({r['room'] for r in tr if r['room']!='unknown'}))};room=torch.tensor([roommap.get(r['room'],-1) for r in tr])
    target=signal_targets(d['t'][ti]);history=[];seen=set();auxseen={};globalstep=0
    print('START',key,flush=True)
    for epoch in range(5):
        t0=time.perf_counter();model.train();indices=[int(rng.choice(v)) for v in record_indices.values()]
        indices+=rng.choice(len(ti),len(ti)-len(indices),replace=True,p=ww).tolist();rng.shuffle(indices)
        losses=[];correct=total=0;align=[]
        for start in range(0,len(indices),32):
            local=indices[start:start+32];ix=[ti[j] for j in local];seen.update(ix)
            opt.zero_grad(set_to_none=True);y=d['y'][ix].cuda();tasks=model.forward_tasks(d['t'][ix].cuda(),d['s'][ix].cuda())
            base,_=task_loss(tasks,y,target[local].cuda(),'physical');bc=nn.functional.cross_entropy(tasks['bank_logits'],y)
            k=sequence(variant,globalstep);aux,ai=auxiliary(model.bank,model.bank_count,model.bank_activity,pool[k],k,auxrng,16)
            auxseen.setdefault(k,set()).update(pool[k]['rows'][i].get('identifier',pool[k]['rows'][i]['record']) for i in ai)
            al=prototype_loss(tasks['latent'],y,room[local].cuda()) if variant=='conditional_multitask' else base*0
            loss=base+.1*bc+.1*aux+(.05*al if variant=='conditional_multitask' else 0)
            assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),3);assert torch.isfinite(norm);opt.step()
            losses.append(loss.item());align.append(al.item());correct+=int((tasks['logits'].argmax(1)==y).sum());total+=len(y);globalstep+=1
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),alignment=float(np.mean(align)),sampled_train_accuracy=correct/total,distinct_windows=len(seen),distinct_records=len({d['rows'][i]['identifier'] for i in seen}),seconds=time.perf_counter()-t0))
        write_csv(folder/'history.csv',history);print('EPOCH',key,history[-1],flush=True)
    val,p=evaluate(model,d,vi) if vi else (None,None)
    if vi:write_csv(folder/'predictions.csv',[dict(**d['rows'][i],prediction=int(p[j].argmax()),**{f'log_probability_{k}':float(v) for k,v in enumerate(p[j])}) for j,i in enumerate(vi)])
    train_eval,_=evaluate(model,d,ti) if fold=='full' else (None,None)
    result=dict(config=config,validation=val,training_evaluation=train_eval,distinct_records_seen=len({d['rows'][i]['identifier'] for i in seen}),distinct_windows_seen=len(seen),auxiliary_records_seen={k:len(v) for k,v in auxseen.items()},platform_score=None)
    torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),folder/'final.pt');save(folder/'metrics.json',result)
    print('DONE',key,val if vi else train_eval,flush=True)
    del model,opt,source,pool;gc.collect();torch.cuda.empty_cache();return result
def summarize(variant,initseed):
    results=[read(WORK/'fits'/f'{variant}_s{initseed}_{fold}'/'metrics.json') for fold in FOLDS]
    cm=np.asarray([r['validation']['confusion_matrix'] for r in results]).sum(0);mean=float(np.mean([r['validation']['accuracy'] for r in results]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    return dict(variant=variant,seed=initseed,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst_room=min(r['validation']['accuracy'] for r in results),nll=float(np.mean([r['validation']['nll'] for r in results])),rooms={f:r['validation']['accuracy'] for f,r in zip(FOLDS,results)},confusion_matrix=cm.tolist())
def ensemble_decision(variant,best):
    cm=np.zeros((5,5),dtype=np.int64);folds={};nll=[]
    for fold in FOLDS:
        sources=[]
        for s in (42,2026):
            with (WORK/'fits'/f'{variant}_s{s}_{fold}'/'predictions.csv').open(encoding='utf-8-sig') as f:sources.append(list(csv.DictReader(f)))
        assert [(r['identifier'],r['start']) for r in sources[0]]==[(r['identifier'],r['start']) for r in sources[1]]
        lp=[np.asarray([[float(r[f'log_probability_{k}']) for k in range(5)] for r in rr]) for rr in sources]
        p=(np.exp(lp[0])+np.exp(lp[1]))/2;y=np.asarray([int(r['headcount']) for r in sources[0]]);prediction=p.argmax(1)
        np.add.at(cm,(y,prediction),1);folds[fold]=float(np.mean(y==prediction));nll.append(float(-np.log(p[np.arange(len(y)),y].clip(1e-8)).mean()))
    mean=float(np.mean(list(folds.values())));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    result=dict(variant=variant,seeds=[42,2026],room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst_room=min(folds.values()),nll=float(np.mean(nll)),rooms=folds,confusion_matrix=cm.tolist(),weights=[.5,.5])
    result['selected']=bool(result['selection_score']>=best['selection_score']+.01 and result['worst_room']>=best['worst_room'])
    result['best_single']=best;save(WORK/'ensemble_decision.json',result);return result
def anchor_decision(variant,best):
    previous=Path('D:/huawei2026-experiments/motion_external_20261021/fits');cm=np.zeros((5,5),dtype=np.int64);basecm=cm.copy();folds={};basefolds={};nll=[]
    for fold in FOLDS:
        paths=[previous/f'baseline_s42_{fold}/predictions.csv',WORK/'fits'/f"{variant}_s{best['seed']}_{fold}/predictions.csv"]
        sources=[]
        for path in paths:
            with path.open(encoding='utf-8-sig') as f:sources.append(list(csv.DictReader(f)))
        assert [(r['identifier'],r['start']) for r in sources[0]]==[(r['identifier'],r['start']) for r in sources[1]]
        lp=[np.asarray([[float(r[f'log_probability_{k}']) for k in range(5)] for r in rr]) for rr in sources]
        p=(np.exp(lp[0])+np.exp(lp[1]))/2;y=np.asarray([int(r['headcount']) for r in sources[0]])
        np.add.at(cm,(y,p.argmax(1)),1);np.add.at(basecm,(y,lp[0].argmax(1)),1)
        folds[fold]=float(np.mean(y==p.argmax(1)));basefolds[fold]=float(np.mean(y==lp[0].argmax(1)));nll.append(float(-np.log(p[np.arange(len(y)),y].clip(1e-8)).mean()))
    score=min(float(np.mean(list(folds.values()))),float(np.mean(np.diag(cm)/cm.sum(1))))
    base_score=min(float(np.mean(list(basefolds.values()))),float(np.mean(np.diag(basecm)/basecm.sum(1))))
    result=dict(variant=variant,external_seed=best['seed'],anchor='original width platform0.37 seed42',room_mean=float(np.mean(list(folds.values()))),macro_recall=float(np.mean(np.diag(cm)/cm.sum(1))),selection_score=score,worst_room=min(folds.values()),nll=float(np.mean(nll)),rooms=folds,confusion_matrix=cm.tolist(),anchor_selection_score=base_score,anchor_rooms=basefolds,weights=[.5,.5],selected=bool(score>=max(best['selection_score'],base_score)+.01 and min(folds.values())>=min(basefolds.values())))
    save(WORK/'anchor_ensemble_decision.json',result);return result
def main(args):
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;assert torch.cuda.is_available()
    WORK.mkdir(parents=True,exist_ok=True);snap=WORK/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
    d,ext=load()
    if args.stage=='screen':
        for variant in VARIANTS:
            for fold in FOLDS:fit(d,ext,variant,42,fold)
        result=[summarize(v,42) for v in VARIANTS];save(WORK/'screen_results.json',result)
        eligible=[r for r in result if r['variant']!='source_control'];eligible.sort(key=lambda r:(-r['selection_score'],-r['worst_room'],r['nll']))
        save(WORK/'selection_stage1.json',dict(selected=eligible[0]['variant'],results=result));print('SCREEN_COMPLETE',json.dumps(result),flush=True)
    elif args.stage=='replicate':
        chosen=read(WORK/'selection_stage1.json')['selected']
        for variant in ['source_control',chosen]:
            for fold in FOLDS:fit(d,ext,variant,2026,fold)
        result=[summarize(v,s) for v in ['source_control',chosen] for s in (42,2026)];save(WORK/'replicate_results.json',result)
        best=max([r for r in result if r['variant']==chosen],key=lambda r:(r['selection_score'],r['worst_room'],-r['nll']));chosen_seed=best['seed']
        ensemble=ensemble_decision(chosen,best)
        anchor=anchor_decision(chosen,best)
        ensemble_kind='external_pair' if ensemble['selected'] else 'single'
        if anchor['selected'] and (ensemble_kind=='single' or anchor['selection_score']>ensemble['selection_score']):ensemble_kind='anchor_pair'
        save(WORK/'selection.json',dict(variant=chosen,seed=chosen_seed,results=result,ensemble_selected=ensemble_kind=='external_pair',ensemble_kind=ensemble_kind,platform_score=None));print('REPLICATE_COMPLETE',json.dumps(dict(results=result,ensemble=ensemble,anchor=anchor)),flush=True)
    elif args.stage=='full':
        s=read(WORK/'selection.json');fit(d,ext,s['variant'],s['seed'],'full');fit(d,ext,'source_control',s['seed'],'full')
        if s.get('ensemble_selected'):fit(d,ext,s['variant'],2026 if s['seed']==42 else 42,'full')
        print('FULL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','replicate','full'],required=True);main(p.parse_args())
