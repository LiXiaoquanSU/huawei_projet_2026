"""Room-disjoint ablations for explicit evidence routes and counting subtasks."""
import argparse,json,hashlib,random
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from structured_model import StructuredModel,load_structured_bank,signal_targets
from train_baseline import metrics,write_csv


def task_loss(tasks,y,target,mode):
    ce=nn.functional.nll_loss(tasks['direct'].clamp_min(1e-8).log(),y)
    expert=nn.functional.cross_entropy(tasks['experts'].reshape(-1,5),y[:,None].expand(-1,4).reshape(-1))
    ordinal=js=aux=ce*0
    if mode!='direct':
        k=torch.arange(4,device=y.device)[None]
        mask=(y[:,None]>=k).to(tasks['logits'].dtype)
        labels=(y[:,None]>k).to(mask.dtype)
        binary=nn.functional.binary_cross_entropy_with_logits(tasks['conditional_logits'],labels,reduction='none')
        ordinal=((binary*mask).sum(0)/mask.sum(0).clamp_min(1)).mean()
        p,q=tasks['direct'].clamp_min(1e-7),tasks['ordinal'].clamp_min(1e-7);m=(p+q)*.5
        js=.5*((p*(p.log()-m.log())).sum(-1)+(q*(q.log()-m.log())).sum(-1)).mean()
    if mode in ['auxiliary','physical']:aux=nn.functional.mse_loss(tasks['signal'],target)
    return ce+.15*expert+.5*ordinal+.1*js+.1*aux,dict(ce=ce.item(),expert=expert.item(),ordinal=ordinal.item(),consistency=js.item(),signal=aux.item())


def assess(model,t,s,y,indices):
    model.eval();all_tasks={}
    with torch.no_grad():
        for batch in indices.split(32):
            tasks=model.forward_tasks(t[batch].cuda(),s[batch].cuda())
            for k,v in tasks.items():all_tasks.setdefault(k,[]).append(v.cpu())
    return {k:torch.cat(v) for k,v in all_tasks.items()}


def run(size,mode,room,bank,args):
    out=args.output/f'{size}_{mode}_{room}_seed{args.seed}'
    if (out/'metrics.json').exists():return json.loads((out/'metrics.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    t,s,y,rows=bank
    vi=torch.tensor([i for i,r in enumerate(rows) if r['room']==room]);ti=torch.tensor([i for i,r in enumerate(rows) if r['room']!=room])
    assert not {rows[i]['identifier'] for i in ti.tolist()}&{rows[i]['identifier'] for i in vi.tolist()}
    assert all(rows[i]['room']!='unknown' for i in ti.tolist()+vi.tolist())
    groups={}
    for i in ti.tolist():groups[(int(y[i]),rows[i]['room'])]=groups.get((int(y[i]),rows[i]['room']),0)+1
    counts={c:len({g for a,g in groups if a==c}) for c in range(5)}
    weights=[1/(groups[(int(y[i]),rows[i]['room'])]*counts[int(y[i])]) for i in ti.tolist()]
    targets=signal_targets(t)
    sampler=WeightedRandomSampler(weights,len(ti),generator=torch.Generator().manual_seed(args.seed))
    loader=DataLoader(TensorDataset(t[ti],s[ti],y[ti],targets[ti]),batch_size=32,sampler=sampler)
    model=StructuredModel(size,mode).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0007 if size=='small' else .0004,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=4,min_lr=1e-5)
    config=dict(size=size,mode=mode,room=room,seed=args.seed,max_epochs=args.epochs,parameters=sum(p.numel() for p in model.parameters()),
        proxy_targets='six amplitude/phase coherence measures, NOT human motion labels',room_area_labels=False,
        ordered_tasks='conditional P(y>k | y>=k), k=0..3',train_windows=len(ti),val_windows=len(vi),
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['structured_model.py','train_structured.py']})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print('START',size,mode,room,config['parameters'],'parameters',flush=True)
    history=[];best=(-1,-1e20);best_epoch=0
    for epoch in range(1,args.epochs+1):
        start=perf_counter();model.train();total=correct=0;losses=[]
        for x,summary,label,target in loader:
            x,summary,label,target=x.cuda(),summary.cuda(),label.cuda(),target.cuda()
            opt.zero_grad(set_to_none=True);tasks=model.forward_tasks(x,summary)
            loss,parts=task_loss(tasks,label,target,mode);assert torch.isfinite(loss)
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),3);assert torch.isfinite(norm);opt.step()
            total+=len(label);correct+=(tasks['logits'].argmax(1)==label).sum().item();losses.append(parts)
        tasks=assess(model,t,s,y,vi);logits=tasks['logits'];va=(logits.argmax(1)==y[vi]).float().mean().item()
        vl=nn.functional.cross_entropy(logits,y[vi]).item();scheduler.step(vl)
        history.append(dict(epoch=epoch,train_accuracy=correct/total,val_accuracy=va,val_loss=vl,seconds=perf_counter()-start,**{k:float(np.mean([v[k] for v in losses])) for k in losses[0]}))
        if (va,-vl)>best:
            best=(va,-vl);best_epoch=epoch
            torch.save(dict(model_state=model.state_dict(),config=config,epoch=epoch),out/'best.pt');best_tasks=tasks
        if epoch==1 or epoch%5==0:print(size,mode,room,epoch,'train',round(correct/total,3),'val',round(va,4),'best',round(best[0],4),'seconds',round(perf_counter()-start,1),flush=True)
        if epoch>=20 and epoch-best_epoch>=10:break
    tasks=best_tasks;logits=tasks['logits'];result=metrics(y[vi].numpy(),logits.argmax(1).numpy())
    result.update(size=size,mode=mode,room=room,seed=args.seed,best_epoch=best_epoch,epochs_run=len(history),parameters=config['parameters'],
        direct_accuracy=float((tasks['direct'].argmax(1)==y[vi]).float().mean()),
        ordinal_accuracy=float((tasks['ordinal'].argmax(1)==y[vi]).float().mean()) if mode!='direct' else None,
        decision_agreement=float((tasks['ordinal'].argmax(1)==tasks['direct'].argmax(1)).float().mean()) if mode!='direct' else None,
        gate_means=tasks['gate'].mean(0).tolist(),route_accuracies=[float((tasks['experts'][:,k].argmax(1)==y[vi]).float().mean()) for k in range(4)],
        signal_proxy_mae=float((tasks['signal']-targets[vi]).abs().mean()) if mode in ['auxiliary','physical'] else None)
    (out/'metrics.json').write_text(json.dumps(result,indent=2));write_csv(out/'history.csv',history)
    predictions=[]
    for j,i in enumerate(vi.tolist()):
        row=dict(identifier=rows[i]['identifier'],room=room,start=rows[i]['start'],target=int(y[i]),predicted=int(logits[j].argmax()),direct=int(tasks['direct'][j].argmax()),ordinal=int(tasks['ordinal'][j].argmax()))
        row.update({f'logit_{k}':float(v) for k,v in enumerate(logits[j])});row.update({f'p_ge_{k+1}':float(v) for k,v in enumerate(tasks['cumulative'][j])});row.update({f'gate_{k}':float(v) for k,v in enumerate(tasks['gate'][j])})
        predictions.append(row)
    write_csv(out/'predictions.csv',predictions)
    print('DONE',size,mode,room,result['accuracy'],flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sizes',nargs='+',choices=['small','large','spectral'],default=['small']);p.add_argument('--modes',nargs='+',choices=['direct','ordinal','auxiliary','physical'],default=['direct','ordinal','auxiliary']);p.add_argument('--rooms',nargs='+',default=['a1','c1','g1']);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=35);p.add_argument('--output',type=Path,default=Path('artifacts/structured_20261019'));args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    results=[]
    for size in args.sizes:
        bank=load_structured_bank(128 if size=='spectral' else 32)
        for mode in args.modes:
            for room in args.rooms:
                results.append(run(size,mode,room,bank,args));(args.output/'summary.json').write_text(json.dumps(results,indent=2))
