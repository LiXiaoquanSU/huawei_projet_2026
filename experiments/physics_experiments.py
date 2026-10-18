import argparse,hashlib,json
from pathlib import Path
import blosc2,numpy as np,torch
from physics_model import PhysicsPreprocessor,PhysicsClassifier
from train_baseline import read_csv,write_csv,window,metrics


def bank():
    root=Path('artifacts/physics');root.mkdir(parents=True,exist_ok=True)
    rows=read_csv('artifacts/baseline_a1_seed42/train_windows.csv')+read_csv('artifacts/baseline_a1_seed42/val_windows.csv')
    identity=hashlib.sha256(Path('physics_model.py').read_bytes()).hexdigest()
    path=root/'features.pt'
    if path.exists():
        d=torch.load(path,weights_only=True)
        assert d['identity']==identity
        return d['features'].cuda(),d['labels'].cuda(),rows
    model=PhysicsPreprocessor().cuda().eval();groups={}
    for j,r in enumerate(rows):groups.setdefault(r['identifier'],[]).append((j,r))
    result=[];order=[]
    with torch.no_grad():
        for n,(ident,group) in enumerate(groups.items()):
            csi=blosc2.load_array(group[0][1]['csi_path']);ati=blosc2.load_array(group[0][1]['ati_path'])
            for start in range(0,len(group),4):
                chunk=group[start:start+4];arrays=[window(csi,ati,r) for _,r in chunk]
                x=torch.from_numpy(np.stack([a[0] for a in arrays])).cuda();dt=torch.from_numpy(np.stack([a[1] for a in arrays])).cuda()
                f=model(x,dt);assert torch.isfinite(f).all();result.append(f.cpu());order.extend(j for j,_ in chunk)
            del csi,ati
            print('Physics cache',n+1,'/',len(groups),flush=True)
    features=torch.cat(result)[np.argsort(order)].contiguous();labels=torch.tensor([int(r['headcount']) for r in rows])
    torch.save(dict(features=features,labels=labels,identity=identity),path)
    return features.cuda(),labels.cuda(),rows


def main():
    torch.set_num_threads(4)
    f,y,rows=bank();out=Path('artifacts/physics');results=[]
    print('Physics feature dimensions',f.shape,flush=True)
    # One shared configuration selected by mean held-room accuracy, never a1 alone.
    for dimension in [109,f.shape[1]]:
        for kernel in ['linear','rbf']:
            for regularization in [1.,10.,100.]:
                for bandwidth in ([.3,1.,3.] if kernel=='rbf' else [1.]):
                    name=f'd{dimension}_{kernel}_l{regularization}_b{bandwidth}';scores=[]
                    folder=out/name;folder.mkdir(exist_ok=True)
                    for room in ['a1','c1','g1']:
                        train=torch.tensor([r['room']!=room for r in rows],device='cuda');val=~train
                        x,labels=f[train,:dimension],y[train];vx=f[val,:dimension]
                        mean=x.mean(0);scale=x.std(0).clamp_min(.03)
                        x=((x-mean)/scale).clamp(-8,8);vx=((vx-mean)/scale).clamp(-8,8)
                        # Class balanced regularized least squares (not fitted on validation).
                        freq=torch.bincount(labels,minlength=5).float()
                        sw=(len(labels)/(5*freq[labels])).sqrt()
                        target=torch.nn.functional.one_hot(labels,5).float()
                        if kernel=='linear':
                            design=torch.cat([x,torch.ones_like(x[:,:1])],1);vdesign=torch.cat([vx,torch.ones_like(vx[:,:1])],1)
                            designw=design*sw[:,None]
                            weights=torch.linalg.solve(designw.T@designw+regularization*torch.eye(dimension+1,device='cuda'),designw.T@(target*sw[:,None]))
                        else:
                            gamma=bandwidth/dimension
                            distance=(x.square().sum(1)[:,None]+x.square().sum(1)[None,:]-2*x@x.T).clamp_min(0)
                            design=torch.exp(-gamma*distance)
                            weighted=design*sw[:,None]*sw[None,:]
                            weights=torch.linalg.solve(weighted+regularization*torch.eye(len(x),device='cuda'),target*sw[:,None])*sw[:,None]
                            vd=(vx.square().sum(1)[:,None]+x.square().sum(1)[None,:]-2*vx@x.T).clamp_min(0)
                            vdesign=torch.exp(-gamma*vd)
                        logits=vdesign@weights;pred=logits.argmax(1)
                        result=metrics(y[val].cpu().numpy(),pred.cpu().numpy());result.update(room=room,name=name,dimension=dimension,kernel=kernel,regularization=regularization,bandwidth=bandwidth)
                        model=PhysicsClassifier(dimension,kernel,len(x)).cuda()
                        model.mean.copy_(mean);model.scale.copy_(scale);model.weights.copy_(weights)
                        if kernel=='rbf':model.support.copy_(x);model.gamma.fill_(gamma)
                        torch.save(dict(model_state=model.cpu().state_dict(),config=result),folder/f'{room}.pt')
                        vi=torch.where(val)[0].cpu().tolist()
                        write_csv(folder/f'{room}_predictions.csv',[dict(identifier=rows[j]['identifier'],room=room,start=rows[j]['start'],target=int(y[j]),predicted=int(p),**{f'logit_{k}':float(v) for k,v in enumerate(l)}) for j,p,l in zip(vi,pred.cpu().numpy(),logits.cpu().numpy())])
                        (folder/f'{room}_metrics.json').write_text(json.dumps(result,indent=2));scores.append(result['accuracy'])
                    entry=dict(name=name,mean_accuracy=float(np.mean(scores)),a1=scores[0],c1=scores[1],g1=scores[2]);results.append(entry)
                    print(entry,flush=True)
    results.sort(key=lambda r:r['mean_accuracy'],reverse=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2));print('BEST',results[:5],flush=True)


if __name__=='__main__':main()
