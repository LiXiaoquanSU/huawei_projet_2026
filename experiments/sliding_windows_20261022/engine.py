import gc,time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from common import ROOT,WORK,FeatureStore,WindowDataset,build,seed,save,read,write_csv,old_data
from model import signal_targets
from utils import task_loss

def ordered_indices(store,recipe,fold,training):
    # Keep old source-record order; original anchor order then matches the old cache.
    identities=list(dict.fromkeys(r['identifier'] for r in old_data()['rows']));order={k:i for i,k in enumerate(identities)}
    return sorted(store.indices(recipe,fold,training),key=lambda i:(order[store.rows[i]['identifier']],store.rows[i]['start']))

def evaluate(model,store,indices):
    model.eval();outputs=[];labels=[];order=[]
    loader=DataLoader(WindowDataset(store,indices),batch_size=32)
    with torch.no_grad():
        for t,s,y,ix in loader:
            outputs.append(model.forward_features(t.cuda(),s.cuda()).cpu());labels.append(y);order.extend(ix.tolist())
    p=torch.cat(outputs);y=torch.cat(labels);pred=p.argmax(1);cm=torch.zeros(5,5,dtype=torch.long);by_record=defaultdict(list)
    for j,(a,b) in enumerate(zip(y.tolist(),pred.tolist())):cm[a,b]+=1;by_record[store.rows[order[j]]['identifier']].append(j)
    records=[];same=pairs=0
    for ident,ix in by_record.items():
        chronological=sorted(ix,key=lambda j:store.rows[order[j]]['start']);pp=pred[chronological]
        same+=int((pp[1:]==pp[:-1]).sum());pairs+=len(pp)-1
        records.append(dict(identifier=ident,room=store.rows[order[ix[0]]]['room'],headcount=int(y[ix[0]]),windows=len(ix),accuracy=float((pred[ix]==y[ix]).float().mean())))
    metric=dict(accuracy=float((pred==y).float().mean()),record_mean_accuracy=float(np.mean([r['accuracy'] for r in records])),nll=float(torch.nn.functional.nll_loss(p,y)),confusion_matrix=cm.tolist(),windows=len(indices),records=len(records),adjacent_agreement=same/max(pairs,1))
    predictions=[dict(**store.rows[i],prediction=int(pred[j]),**{f'log_probability_{k}':float(v) for k,v in enumerate(p[j])}) for j,i in enumerate(order)]
    return metric,predictions,records

def fit(store,model_name,recipe,s,fold):
    folder=WORK/'fits'/f'{model_name}_{recipe}_s{s}_{fold}'
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    ti=ordered_indices(store,recipe,fold,True);vi=ordered_indices(store,'baseline',fold,False) if fold!='full' else []
    base=ordered_indices(store,'baseline',fold,True);tr=[store.rows[i] for i in ti]
    train_records={r['identifier'] for r in tr};validation_records={store.rows[i]['identifier'] for i in vi}
    assert train_records.isdisjoint(validation_records)
    if fold!='full':assert all(r['room'] not in ('unknown',fold) for r in tr)
    windows_per_record=defaultdict(int);group_records=defaultdict(set)
    domain=lambda r:r['identifier'] if r['room']=='unknown' else r['room']
    for r in tr:
        windows_per_record[r['identifier']]+=1;group_records[(int(r['headcount']),domain(r))].add(r['identifier'])
    class_groups={c:len({g for k,g in group_records if k==c}) for c in range(5)}
    weights=[1/(windows_per_record[r['identifier']]*len(group_records[(int(r['headcount']),domain(r))])*class_groups[int(r['headcount'])]) for r in tr]
    # Same samples and update count per epoch as the old nonoverlapping experiment.
    sampler=WeightedRandomSampler(weights,len(base),generator=torch.Generator().manual_seed(42))
    loader=DataLoader(WindowDataset(store,ti),batch_size=32,sampler=sampler)
    seed(s);model=build(model_name).cuda();epochs=12 if model_name=='mixer' else 5
    config=dict(architecture=model_name,recipe=recipe,seed=s,fold=fold,epochs=epochs,lr=.0004,batch_size=32,weight_decay=.01,parameters=sum(p.numel() for p in model.parameters()),training_records=len(train_records),training_windows=len(ti),samples_per_epoch=len(base),optimizer_steps_per_epoch=(len(base)+31)//32,external_data=False,initialization='fresh',selection='fixed final epoch; no validation checkpoint selection',record_balanced_sampling=True,platform_score=None)
    folder.mkdir(parents=True,exist_ok=True);save(folder/'config.json',config);write_csv(folder/'train_manifest.csv',tr)
    if vi:write_csv(folder/'validation_manifest.csv',[store.rows[i] for i in vi])
    opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01);seed(42);seen=set();history=[];print('START',folder.name,'available',len(ti),'samples/epoch',len(base),flush=True)
    for epoch in range(epochs):
        model.train();losses=[];correct=total=0;start=time.perf_counter()
        for t,summary,y,ix in loader:
            seen.update(ix.tolist());target=signal_targets(t) if model_name!='mixer' else None
            t,summary,y=t.cuda(),summary.cuda(),y.cuda();opt.zero_grad(set_to_none=True)
            tasks=model.forward_tasks(t,summary)
            loss=torch.nn.functional.cross_entropy(tasks['logits'],y,label_smoothing=.05) if model_name=='mixer' else task_loss(tasks,y,target.cuda(),'physical')[0]
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3);assert torch.isfinite(norm);opt.step()
            losses.append(float(loss));correct+=int((tasks['logits'].argmax(1)==y).sum());total+=len(y)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),sampled_train_accuracy=correct/total,seconds=time.perf_counter()-start));write_csv(folder/'history.csv',history)
        print('EPOCH',folder.name,history[-1],flush=True)
    primary,pp,records=evaluate(model,store,vi or base)
    if fold!='full':
        dense_ix=ordered_indices(store,'step_0p5s',fold,False);dense,dp,dr=evaluate(model,store,dense_ix)
        write_csv(folder/'dense_validation_manifest.csv',[store.rows[i] for i in dense_ix]);write_csv(folder/'dense_predictions.csv',dp);write_csv(folder/'dense_record_metrics.csv',dr)
    else:dense=None
    write_csv(folder/('predictions.csv' if vi else 'training_predictions.csv'),pp);write_csv(folder/'primary_record_metrics.csv',records)
    actual_records=len({store.rows[i]['identifier'] for i in seen})
    if fold=='full':assert actual_records==48
    result=dict(config=config,validation=primary if vi else None,training_evaluation=primary if not vi else None,dense_validation=dense,actual_records=actual_records,actual_windows=len(seen),baseline_validation_unchanged=True,platform_score=None)
    torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),folder/'final.pt');save(folder/'metrics.json',result);print('DONE',folder.name,'primary',primary['accuracy'],'dense',None if dense is None else dense['record_mean_accuracy'],flush=True)
    del model,opt,loader;gc.collect();torch.cuda.empty_cache();return result
