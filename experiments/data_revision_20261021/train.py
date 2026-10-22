"""Keep the0.37 topology; transfer only a real single-Rx dynamic CNN trunk."""
import argparse,csv,gc,hashlib,json,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler,DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from model import StructuredModel,EvidenceEncoder,signal_targets
from utils import task_loss,write_csv
from acquire import WORK,save
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
FOLDS=['a1','c1','g1','e1'];TRIAD=FOLDS[:3]
VARIANTS=['source_transfer','ehu_transfer','ehu_radio','ehu_fullrate']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seed(s=42):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def balance(rows,y):
    groups={};rooms={}
    domain=lambda r:r['room'] if r['room']!='unknown' else r.get('identifier',r.get('record'))
    for r,label in zip(rows,y):
        key=int(label),domain(r),r.get('identifier',r.get('record'));groups[key]=groups.get(key,0)+1;rooms.setdefault(int(label),{}).setdefault(domain(r),set()).add(key[2])
    w=np.array([1/len(rooms[int(label)])/len(rooms[int(label)][domain(r)])/groups[(int(label),domain(r),r.get('identifier',r.get('record')))] for r,label in zip(rows,y)])
    return w/w.sum()
def load():
    d=torch.load(CACHE,map_location='cpu',weights_only=True);d.pop('absolute',None)
    for k,v in d['source_hashes'].items():assert sha(ROOT/'src'/k)==v
    return d
def source_rx(d,ix):
    a=d['t'][ix,384:768].reshape(-1,3,128,150).reshape(-1,1,128,150)
    a=a-a.mean(-1,keepdim=True);a=a/(a.square().mean((1,2,3),keepdim=True)+1e-6).sqrt();a=a.clamp(-6,6)
    rr=[dict(d['rows'][i],record=d['rows'][i]['identifier'],actual_rx=rx) for i in ix for rx in range(3)]
    return dict(x=a,y=d['y'][ix,None].expand(-1,3).reshape(-1),rows=rr)
def pool_weights(v):v['w']=balance(v['rows'],v['y']);return v
def pretrain(d,ix,variant,s,fold):
    family='ehu_transfer' if variant=='ehu_fullrate' else variant
    key=f'{family}_s{s}_{fold if family=="source_transfer" else "public"}';folder=WORK/'pretrain'/key
    if (folder/'metrics.json').exists():return folder/'final.pt'
    folder.mkdir(parents=True,exist_ok=True)
    if family=='source_transfer':datasets={'source':pool_weights(source_rx(d,ix))}
    else:
        ext=torch.load(WORK/'external_dynamic.pt',map_location='cpu',weights_only=True)
        datasets={'ehu':pool_weights(ext['ehu_train'])}
        if family=='ehu_radio':datasets['radio']=pool_weights(ext['radio'])
    seed(s);enc=EvidenceEncoder(1,[64,128,256,384],256,style=False,frequency_bins=None).cuda();head=nn.Linear(256,5).cuda()
    parameters=[*enc.parameters(),*head.parameters()];opt=torch.optim.AdamW(parameters,lr=.0004,weight_decay=.01);rng=np.random.default_rng(42);history=[];seen={}
    for epoch in range(4):
        losses=[];enc.train();head.train()
        for step in range(32):
            name=list(datasets)[step%len(datasets)];v=datasets[name];ii=rng.choice(len(v['y']),32,replace=True,p=v['w'])
            a=v['x'][ii].cuda();y=v['y'][ii].cuda();opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(enc(a)),y);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(parameters,3);opt.step();losses.append(loss.item())
            seen.setdefault(name,set()).update(v['rows'][j].get('identifier',v['rows'][j].get('record')) for j in ii)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))));print('PRETRAIN',key,history[-1],flush=True)
    external_validation=None
    if family!='source_transfer':
        enc.eval();head.eval();v=ext['ehu_test'];pred=[]
        with torch.no_grad():
            for start in range(0,len(v['y']),32):pred.append(head(enc(v['x'][start:start+32].cuda())).argmax(1).cpu())
        pred=torch.cat(pred);cm=torch.zeros(5,5,dtype=torch.long)
        for a,b in zip(v['y'].tolist(),pred.tolist()):cm[a,b]+=1
        external_validation=dict(accuracy=float((pred==v['y']).float().mean()),confusion_matrix=cm.tolist(),support=torch.bincount(v['y'],minlength=5).tolist(),room='Office1; entire physical room never trained')
    torch.save(dict(encoder={k:v.detach().cpu() for k,v in enc.state_dict().items()},head={k:v.detach().cpu() for k,v in head.state_dict().items()}),folder/'final.pt')
    save(folder/'metrics.json',dict(seed=s,family=family,history=history,updates=128,actual_records={k:len(v) for k,v in seen.items()},external_validation=external_validation))
    for k,v in datasets.items():write_csv(folder/f'{k}_train_manifest.csv',v['rows'])
    del enc,head,opt,parameters,datasets;gc.collect();torch.cuda.empty_cache();return folder/'final.pt'
def evaluate(model,d,ix):
    model.eval();pred=[]
    with torch.no_grad():
        for start in range(0,len(ix),32):
            j=ix[start:start+32];pred.append(model.forward_features(d['t'][j].cuda(),d['s'][j].cuda()).cpu())
    p=torch.cat(pred);y=d['y'][ix];cm=torch.zeros(5,5,dtype=torch.long)
    for a,b in zip(y.tolist(),p.argmax(1).tolist()):cm[a,b]+=1
    return dict(accuracy=float((p.argmax(1)==y).float().mean()),nll=float(nn.functional.nll_loss(p,y)),confusion_matrix=cm.tolist()),p
def fit(d,variant,s,fold):
    folder=WORK/'fits'/f'{variant}_s{s}_{fold}'
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    ti=list(range(len(d['rows']))) if fold=='full' else [i for i,r in enumerate(d['rows']) if r['room'] not in ('unknown',fold)]
    vi=[] if fold=='full' else [i for i,r in enumerate(d['rows']) if r['room']==fold];tr=[d['rows'][i] for i in ti]
    assert {r['identifier'] for r in tr}.isdisjoint({d['rows'][i]['identifier'] for i in vi})
    cp=None if variant=='baseline' else pretrain(d,ti,variant,s,fold)
    torch.manual_seed(s);model=StructuredModel('spectral','physical')
    if cp:
        state=torch.load(cp,map_location='cpu',weights_only=True)['encoder'];target=model.dynamic.layers.state_dict()
        for k in target:
            v=state['layers.'+k]
            if v.shape!=target[k].shape:
                assert k in ('0.body.0.weight','0.skip.weight');v=v.repeat(1,3,1,1)/3
            assert v.shape==target[k].shape;target[k]=v
        model.dynamic.layers.load_state_dict(target)
    model.cuda();folder.mkdir(parents=True,exist_ok=True);warm=variant not in ('baseline','ehu_fullrate')
    config=dict(variant=variant,seed=s,fold=fold,pretrain=str(cp) if cp else None,pretrain_sha256=sha(cp) if cp else None,epochs=5,batch_size=32,lr=.0004,dynamic_trunk_lr=.00004 if warm else .0004,freeze_first_epoch=warm,training_records=len({r['identifier'] for r in tr}),training_windows=len(ti),parameters=sum(p.numel() for p in model.parameters()),platform_score=None)
    save(folder/'config.json',config);write_csv(folder/'train_manifest.csv',tr)
    if vi:write_csv(folder/'validation_manifest.csv',[d['rows'][i] for i in vi])
    # Original0.37 sampler protocol, including its separate RNG.
    domains=lambda r:r['identifier'] if r['room']=='unknown' else r['room']
    groups={};counts={}
    for r,i in zip(tr,ti):key=int(d['y'][i]),domains(r);groups[key]=groups.get(key,0)+1
    for c in range(5):counts[c]=len({g for label,g in groups if label==c})
    w=[1/(groups[(int(d['y'][i]),domains(r))]*counts[int(d['y'][i])]) for r,i in zip(tr,ti)]
    sampler=WeightedRandomSampler(w,len(ti),generator=torch.Generator().manual_seed(42))
    target=signal_targets(d['t'][ti]);loader=DataLoader(TensorDataset(d['t'][ti],d['s'][ti],d['y'][ti],target,torch.arange(len(ti))),batch_size=32,sampler=sampler)
    trunk=list(model.dynamic.layers.parameters());ids={id(p) for p in trunk};rest=[p for p in model.parameters() if id(p) not in ids]
    opt=torch.optim.AdamW([dict(params=rest,lr=.0004),dict(params=trunk,lr=config['dynamic_trunk_lr'])],weight_decay=.01)
    seed();history=[];seen=set();print('START',folder.name,flush=True)
    for epoch in range(5):
        for p in trunk:p.requires_grad_(not(warm and epoch==0))
        model.train();losses=[];correct=total=0;t0=time.perf_counter()
        for t,summary,y,target,local in loader:
            seen.update(local.tolist());t,summary,y,target=t.cuda(),summary.cuda(),y.cuda(),target.cuda();opt.zero_grad(set_to_none=True)
            tasks=model.forward_tasks(t,summary);loss,_=task_loss(tasks,y,target,'physical');assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),3);assert torch.isfinite(norm);opt.step()
            losses.append(loss.item());correct+=int((tasks['logits'].argmax(1)==y).sum());total+=len(y)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),sampled_train_accuracy=correct/total,records_seen=len({tr[i]['identifier'] for i in seen}),windows_seen=len(seen),seconds=time.perf_counter()-t0));write_csv(folder/'history.csv',history);print('EPOCH',folder.name,history[-1],flush=True)
    val,p=evaluate(model,d,vi) if vi else (None,None);trainval,_=evaluate(model,d,ti) if not vi else (None,None)
    if vi:write_csv(folder/'predictions.csv',[dict(**d['rows'][i],prediction=int(p[j].argmax()),**{f'log_probability_{k}':float(v) for k,v in enumerate(p[j])}) for j,i in enumerate(vi)])
    result=dict(config=config,validation=val,training_evaluation=trainval,actual_records=len({tr[i]['identifier'] for i in seen}),actual_windows=len(seen),platform_score=None)
    if fold=='full':assert result['actual_records']==48
    torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),folder/'final.pt');save(folder/'metrics.json',result);print('DONE',folder.name,val or trainval,flush=True)
    del model,opt,loader,rest,trunk;gc.collect();torch.cuda.empty_cache();return result
def summary(v,s):
    rr=[read(WORK/'fits'/f'{v}_s{s}_{f}/metrics.json') for f in TRIAD];cm=np.asarray([r['validation']['confusion_matrix'] for r in rr]).sum(0)
    mean=float(np.mean([r['validation']['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    e=WORK/'fits'/f'{v}_s{s}_e1/metrics.json'
    return dict(variant=v,seed=s,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst=min(r['validation']['accuracy'] for r in rr),nll=float(np.mean([r['validation']['nll'] for r in rr])),rooms={f:r['validation']['accuracy'] for f,r in zip(TRIAD,rr)},confusion_matrix=cm.tolist(),e1=read(e)['validation'] if e.exists() else None)
def main(args):
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;assert torch.cuda.is_available();WORK.mkdir(parents=True,exist_ok=True)
    snap=WORK/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.md','.py'):shutil.copy2(p,snap/p.name)
    d=load()
    if args.stage=='control':
        fit(d,'baseline',42,'e1')
        for f in FOLDS:fit(d,'source_transfer',42,f)
    elif args.stage=='screen':
        for v in VARIANTS[1:]:
            for f in FOLDS:fit(d,v,42,f)
        rr=[summary(v,42) for v in VARIANTS];save(WORK/'screen_results.json',rr)
        selected=max([r for r in rr if r['variant']!='source_transfer'],key=lambda r:(r['selection_score'],r['worst'],-r['nll']))
        save(WORK/'selection_stage1.json',selected);print('SCREEN_COMPLETE',json.dumps(rr),flush=True)
    elif args.stage=='replicate':
        chosen=read(WORK/'selection_stage1.json')['variant']
        for v in ('source_transfer',chosen):
            for f in TRIAD:fit(d,v,2026,f)
        rr=[summary(v,s) for v in ('source_transfer',chosen) for s in (42,2026)];save(WORK/'replicate_results.json',rr)
        best=max([r for r in rr if r['variant']==chosen],key=lambda r:(r['selection_score'],r['worst'],-r['nll']))
        save(WORK/'selection.json',dict(variant=chosen,seed=best['seed'],results=rr));print('REPLICATE_COMPLETE',json.dumps(rr),flush=True)
    elif args.stage=='full':
        s=read(WORK/'selection.json');fit(d,s['variant'],s['seed'],'full');fit(d,'source_transfer',s['seed'],'full');print('FULL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['control','screen','replicate','full'],required=True);main(p.parse_args())
