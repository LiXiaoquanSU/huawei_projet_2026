"""Architecture-only paired fits, with fixed epochs and record-disjoint folds."""
import csv,gc,hashlib,json,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from network import ROOT,build
from model import signal_targets
from utils import task_loss,write_csv
WORK=Path('D:/huawei2026-experiments/fresh_backbones_20261022')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
FOLDS=['a1','c1','g1']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def load():
    d=torch.load(CACHE,map_location='cpu',weights_only=True);d.pop('absolute',None)
    for k,v in d['source_hashes'].items():assert sha(ROOT/'src'/k)==v
    return d
def evaluate(model,d,ix):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(ix),32):
            j=ix[start:start+32];out.append(model.forward_features(d['t'][j].cuda(),d['s'][j].cuda()).cpu())
    p=torch.cat(out);y=d['y'][ix];cm=torch.zeros(5,5,dtype=torch.long)
    for a,b in zip(y.tolist(),p.argmax(1).tolist()):cm[a,b]+=1
    return dict(accuracy=float((p.argmax(1)==y).float().mean()),nll=float(torch.nn.functional.nll_loss(p,y)),confusion_matrix=cm.tolist()),p
def fit(d,v,s,f):
    folder=WORK/'fits'/f'{v}_s{s}_{f}'
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    ti=list(range(len(d['rows']))) if f=='full' else [i for i,r in enumerate(d['rows']) if r['room'] not in ('unknown',f)]
    vi=[] if f=='full' else [i for i,r in enumerate(d['rows']) if r['room']==f]
    tr=[d['rows'][i] for i in ti];assert {r['identifier'] for r in tr}.isdisjoint({d['rows'][i]['identifier'] for i in vi})
    seed(s);model=build(v).cuda();folder.mkdir(parents=True,exist_ok=True)
    config=dict(architecture=v,seed=s,fold=f,epochs=12,lr=.0004,batch_size=32,weight_decay=.01,parameters=sum(p.numel() for p in model.parameters()),training_records=len({r['identifier'] for r in tr}),training_windows=len(ti),external_data=False,platform_score=None)
    config['loss']='original_composite' if v=='context_control' else 'cross_entropy_label_smoothing_0.05'
    save(folder/'config.json',config);write_csv(folder/'train_manifest.csv',tr)
    if vi:write_csv(folder/'validation_manifest.csv',[d['rows'][i] for i in vi])
    domain=lambda r:r['identifier'] if r['room']=='unknown' else r['room'];groups={}
    for r,i in zip(tr,ti):key=int(d['y'][i]),domain(r);groups[key]=groups.get(key,0)+1
    counts={c:len({g for label,g in groups if label==c}) for c in range(5)}
    w=[1/(groups[(int(d['y'][i]),domain(r))]*counts[int(d['y'][i])]) for r,i in zip(tr,ti)]
    sampler=WeightedRandomSampler(w,len(ti),generator=torch.Generator().manual_seed(42))
    loader=DataLoader(TensorDataset(d['t'][ti],d['s'][ti],d['y'][ti],signal_targets(d['t'][ti]),torch.arange(len(ti))),batch_size=32,sampler=sampler)
    opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01);seed(42);history=[];seen=set();print('START',folder.name,config['parameters'],flush=True)
    for epoch in range(12):
        model.train();losses=[];correct=total=0;t0=time.perf_counter()
        for t,summary,y,target,local in loader:
            seen.update(local.tolist());t,summary,y,target=t.cuda(),summary.cuda(),y.cuda(),target.cuda();opt.zero_grad(set_to_none=True)
            tasks=model.forward_tasks(t,summary);loss=task_loss(tasks,y,target,'physical')[0] if v=='context_control' else torch.nn.functional.cross_entropy(tasks['logits'],y,label_smoothing=.05);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3);assert torch.isfinite(norm);opt.step()
            losses.append(loss.item());correct+=int((tasks['logits'].argmax(1)==y).sum());total+=len(y)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),sampled_train_accuracy=correct/total,seconds=time.perf_counter()-t0));write_csv(folder/'history.csv',history);print('EPOCH',folder.name,history[-1],flush=True)
    metric,p=evaluate(model,d,vi or ti)
    result=dict(config=config,validation=metric if vi else None,training_evaluation=metric if not vi else None,actual_records=len({tr[i]['identifier'] for i in seen}),actual_windows=len(seen),platform_score=None)
    if f=='full':assert result['actual_records']==48
    write_csv(folder/('predictions.csv' if vi else 'training_predictions.csv'),[dict(**d['rows'][i],prediction=int(p[j].argmax()),**{f'log_probability_{k}':float(x) for k,x in enumerate(p[j])}) for j,i in enumerate(vi or ti)])
    torch.save(dict(model_state={k:x.detach().cpu() for k,x in model.state_dict().items()},config=config),folder/'final.pt');save(folder/'metrics.json',result);print('DONE',folder.name,metric,flush=True)
    del model,opt,loader;gc.collect();torch.cuda.empty_cache();return result
