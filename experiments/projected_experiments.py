import json
from pathlib import Path
import numpy as np,torch
from run_iterations import load_bank
from improved_models import ImprovedModel
from phase_models import PhaseModel
from projected_model import ProjectedClassifier
from train_baseline import metrics,write_csv


def descriptor_cache():
    out=Path('artifacts/projected');out.mkdir(parents=True,exist_ok=True)
    original=load_bank();relative=load_bank(True)
    rows=original[3]
    p=torch.load('artifacts/physics/features.pt',weights_only=True)['features']
    sources={'physics':p}
    for mode,bank in [('amplitude',original),('phase',relative)]:
        model=(ImprovedModel('spectrum') if mode=='amplitude' else PhaseModel('phase_spectrum')).cuda().eval()
        result=[]
        with torch.no_grad():
            for start in range(0,len(rows),32):
                t=bank[0][start:start+32].cuda()
                f=model.spectral_features(model.signal(t),t) if mode=='amplitude' else model.spectral(t[:,:288],t)
                result.append(f.cpu())
        sources[mode]=torch.cat(result)
    sources['all']=torch.cat(list(sources.values()),dim=1)
    torch.save(dict(sources=sources,labels=original[2]),out/'descriptors.pt')
    return {k:v.cuda() for k,v in sources.items()},original[2].cuda(),rows


def fit(f,y,rows,room,rank,strength,kernel,lam,band):
    ti=torch.tensor([r['room']!=room for r in rows],device='cuda');vi=~ti
    x=f[ti];vx=f[vi];labels=y[ti]
    mean=x.mean(0);scale=x.std(0).clamp_min(.03)
    x=((x-mean)/scale).clamp(-8,8);vx=((vx-mean)/scale).clamp(-8,8)
    train_rows=[r for r in rows if r['room']!=room]
    residuals=[]
    for c in range(5):
        rooms=sorted({r['room'] for r in train_rows if int(r['headcount'])==c})
        if len(rooms)<2:continue
        centers=[]
        for source_room in rooms:
            idx=torch.tensor([r['room']==source_room and int(r['headcount'])==c for r in train_rows],device='cuda')
            centers.append(x[idx].mean(0))
        centers=torch.stack(centers);residuals.append(centers-centers.mean(0))
    residual=torch.cat(residuals)
    _,singular,vh=torch.linalg.svd(residual.cpu(),full_matrices=False)
    available=int((singular>1e-4).sum())
    directions=vh[:min(rank,available)].cuda()
    x=x-strength*(x@directions.T)@directions;vx=vx-strength*(vx@directions.T)@directions
    freq=torch.bincount(labels,minlength=5).float();sw=(len(labels)/(5*freq[labels])).sqrt()
    target=torch.nn.functional.one_hot(labels,5).float()
    if kernel=='linear':
        design=torch.cat([x,torch.ones_like(x[:,:1])],1);vd=torch.cat([vx,torch.ones_like(vx[:,:1])],1)
        a=design*sw[:,None]
        weights=torch.linalg.solve(a.T@a+lam*torch.eye(design.shape[1],device='cuda'),a.T@(target*sw[:,None]))
    else:
        gamma=band/x.shape[1]
        distance=(x.square().sum(1)[:,None]+x.square().sum(1)[None]-2*x@x.T).clamp_min(0)
        gram=torch.exp(-gamma*distance)
        weights=torch.linalg.solve(gram*sw[:,None]*sw[None]+lam*torch.eye(len(x),device='cuda'),target*sw[:,None])*sw[:,None]
        distance=(vx.square().sum(1)[:,None]+x.square().sum(1)[None]-2*vx@x.T).clamp_min(0)
        vd=torch.exp(-gamma*distance)
    logits=vd@weights;result=metrics(y[vi].cpu().numpy(),logits.argmax(1).cpu().numpy())
    state=dict(mean=mean,scale=scale,directions=directions,strength=torch.tensor(strength,device='cuda'),
               support=x,gamma=torch.tensor(band/x.shape[1],device='cuda'),weights=weights)
    return result,state,logits.detach().cpu(),torch.where(vi)[0].cpu().tolist()


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    sources,y,rows=descriptor_cache();out=Path('artifacts/projected');results=[];best=-1
    for source,f in sources.items():
        for rank in [0,1,2,4]:
            for kernel in ['linear','rbf']:
                for lam in [1.,10.,100.]:
                    band=1.;strength=1.
                    all_fits=[fit(f,y,rows,room,rank,strength,kernel,lam,band) for room in ['a1','c1','g1']]
                    scores=[v[0]['accuracy'] for v in all_fits]
                    entry=dict(source=source,rank=rank,kernel=kernel,regularization=lam,bandwidth=band,
                               mean_accuracy=float(np.mean(scores)),a1=scores[0],c1=scores[1],g1=scores[2])
                    results.append(entry)
                    if entry['mean_accuracy']>best:
                        best=entry['mean_accuracy'];print('NEW BEST',entry,flush=True)
                        for room,(m,state,logits,vi) in zip(['a1','c1','g1'],all_fits):
                            config=dict(entry,dimension=f.shape[1],room=room,n_support=state['support'].shape[0],actual_rank=state['directions'].shape[0])
                            torch.save(dict(buffers={k:v.cpu() for k,v in state.items()},config=config,metrics=m),out/f'best_{room}.pt')
                            write_csv(out/f'best_{room}_predictions.csv',[dict(identifier=rows[j]['identifier'],room=room,start=rows[j]['start'],target=int(y[j]),predicted=int(l.argmax()),**{f'logit_{k}':float(v) for k,v in enumerate(l)}) for j,l in zip(vi,logits)])
                    (out/'summary.json').write_text(json.dumps(sorted(results,key=lambda r:r['mean_accuracy'],reverse=True),indent=2))
        print('Completed source',source,flush=True)
    print('TOP',sorted(results,key=lambda r:r['mean_accuracy'],reverse=True)[:8],flush=True)


if __name__=='__main__':main()
