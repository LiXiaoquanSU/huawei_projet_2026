import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from network import ROOT,build
WORK=Path('D:/huawei2026-experiments/random_dictionary_20261022')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load():
    d=torch.load(CACHE,map_location='cpu',weights_only=True);d.pop('absolute',None)
    for name,h in d['source_hashes'].items():assert sha(ROOT/'src'/name)==h
    return d
def write_csv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def features(d,s):
    torch.manual_seed(s);m=build().eval();path=WORK/f'features_s{s}.pt'
    if path.exists():return m,torch.load(path,weights_only=True)
    with torch.no_grad():z=torch.cat([m.features(d['t'][i:i+16],d['s'][i:i+16]) for i in range(0,len(d['t']),16)])
    assert z.shape==(960,1169) and torch.isfinite(z).all();torch.save(z,path);return m,z
def fit(d,s,f):
    folder=WORK/'fits'/f'random_dictionary_s{s}_{f}'
    if (folder/'metrics.json').exists():return read(folder/'metrics.json')
    m,x=features(d,s);ti=list(range(960)) if f=='full' else [i for i,r in enumerate(d['rows']) if r['room'] not in ('unknown',f)];vi=[] if f=='full' else [i for i,r in enumerate(d['rows']) if r['room']==f]
    tr=[d['rows'][i] for i in ti];assert {r['identifier'] for r in tr}.isdisjoint({d['rows'][i]['identifier'] for i in vi})
    groups={};domain=lambda r:r['identifier'] if r['room']=='unknown' else r['room']
    for i,r in zip(ti,tr):key=(int(d['y'][i]),domain(r));groups[key]=groups.get(key,0)+1
    counts={c:len({g for k,g in groups if k==c}) for c in range(5)}
    w=torch.tensor([1/(groups[(int(d['y'][i]),domain(r))]*counts[int(d['y'][i])]) for i,r in zip(ti,tr)],dtype=torch.float64);w=w/w.mean()
    xt=x[ti].double();center=(xt*w[:,None]).sum(0)/w.sum();scale=(((xt-center).square()*w[:,None]).sum(0)/w.sum()).sqrt().clamp_min(.001)
    z=(xt-center)/scale;y=torch.nn.functional.one_hot(d['y'][ti],5).double();intercept=(y*w[:,None]).sum(0)/w.sum()
    a=z*w.sqrt()[:,None];b=(y-intercept)*w.sqrt()[:,None]
    coef=a.T@torch.linalg.solve(a@a.T+10*torch.eye(len(ti),dtype=torch.float64),b)
    assert torch.isfinite(coef).all()
    m.center.copy_(center.float());m.scale.copy_(scale.float());m.coef.copy_(coef.float());m.intercept.copy_(intercept.float())
    ix=vi or ti;p=(((x[ix]-m.center)/m.scale)@m.coef+m.intercept).log_softmax(-1);labels=d['y'][ix];cm=torch.zeros(5,5,dtype=torch.long)
    for c,pred in zip(labels,p.argmax(1)):cm[c,pred]+=1
    metric=dict(accuracy=float((p.argmax(1)==labels).float().mean()),nll=float(torch.nn.functional.nll_loss(p,labels)),confusion_matrix=cm.tolist())
    folder.mkdir(parents=True,exist_ok=True);config=dict(architecture='random_dictionary',seed=s,fold=f,parameters=sum(p.numel() for p in m.parameters()),optimizer='weighted_ridge_dual',ridge_lambda=10,external_data=False,training_records=len({r['identifier'] for r in tr}),training_windows=len(ti),platform_score=None)
    write_csv(folder/'train_manifest.csv',tr)
    if vi:write_csv(folder/'validation_manifest.csv',[d['rows'][i] for i in vi])
    write_csv(folder/('predictions.csv' if vi else 'training_predictions.csv'),[dict(**d['rows'][i],prediction=int(p[j].argmax()),**{f'log_probability_{k}':float(v) for k,v in enumerate(p[j])}) for j,i in enumerate(ix)])
    result=dict(config=config,validation=metric if vi else None,training_evaluation=metric if not vi else None,actual_records=config['training_records'],actual_windows=len(ti),platform_score=None)
    if f=='full':assert result['actual_records']==48
    torch.save(dict(model_state=m.state_dict(),config=config),folder/'final.pt');save(folder/'metrics.json',result);print('DONE',folder.name,metric,flush=True);return result
def summary(s):
    rr=[read(WORK/'fits'/f'random_dictionary_s{s}_{f}/metrics.json')['validation'] for f in ('a1','c1','g1')];cm=np.asarray([r['confusion_matrix'] for r in rr]).sum(0)
    mean=float(np.mean([r['accuracy'] for r in rr]));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
    return dict(architecture='random_dictionary',seed=s,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),worst=min(r['accuracy'] for r in rr),nll=float(np.mean([r['nll'] for r in rr])),confusion_matrix=cm.tolist())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');args=parser.parse_args();torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True);d=load()
    for s in (42,2026):
        for f in ('a1','c1','g1','e1'):fit(d,s,f)
    rr=[summary(s) for s in (42,2026)];best=max(rr,key=lambda r:(r['selection_score'],r['worst'],-r['nll']))
    item=dict(architecture='random_dictionary',seed=best['seed'],two_seed_room_mean=float(np.mean([r['room_mean'] for r in rr])),two_seed_selection=float(np.mean([r['selection_score'] for r in rr])),results=rr,e1=[read(WORK/'fits'/f'random_dictionary_s{s}_e1/metrics.json')['validation'] for s in (42,2026)])
    save(WORK/'selection.json',dict(candidates=[item],target_60_reached=item['two_seed_room_mean']>=.6,platform_best=.39));print('SUMMARY',json.dumps(item),flush=True)
    if args.full:fit(d,best['seed'],'full')
    snap=WORK/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.md'):shutil.copy2(p,snap/p.name)
if __name__=='__main__':main()
