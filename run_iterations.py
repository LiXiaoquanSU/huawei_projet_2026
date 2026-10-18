"""Room-disjoint development experiments on existing fixed feature caches."""
import argparse,csv,hashlib,json,random
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from improved_models import ImprovedModel
from phase_models import PhaseModel
from tf_model import TFModel
from roomsense_model import RoomSenseModel
from train_baseline import read_csv,write_csv,evaluate,metrics


def load_bank(phase=False,tf=False):
    root=Path('artifacts/baseline_a1_seed42')
    ts,ss,ys,rows=[],[],[],[]
    for split in ['train','val']:
        location=Path('artifacts/tf_cache') if tf else Path('artifacts/phase_cache') if phase else root
        d=torch.load(location/f'{split}_features.pt',weights_only=True,map_location='cpu')
        ts.append(d['temporal']);ss.append(d['summary']);ys.append(d['labels'])
        rows+=read_csv(root/f'{split}_windows.csv')
    return torch.cat(ts),torch.cat(ss),torch.cat(ys),rows


def run(mode,room,seed,out,bank,epochs):
    folder=out/f'{mode}_{room}_seed{seed}'
    if (folder/'metrics.json').exists():
        return json.loads((folder/'metrics.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    t,s,y,rows=bank
    vi=torch.tensor([i for i,r in enumerate(rows) if r['room']==room]);ti=torch.tensor([i for i,r in enumerate(rows) if r['room']!=room])
    assert not {rows[i]['room'] for i in ti.tolist()} & {rows[i]['room'] for i in vi.tolist()}
    model=(RoomSenseModel() if mode=='original' else TFModel(mode) if mode.startswith('tfcnn') else PhaseModel(mode) if mode.startswith('phase') else ImprovedModel(mode)).cuda()
    train=TensorDataset(t[ti],s[ti],y[ti]);val=TensorDataset(t[vi],s[vi],y[vi])
    if mode=='original':
        loader=DataLoader(train,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
        weights=torch.bincount(y[ti],minlength=5).float().rsqrt();weights/=weights.mean()
        loss_fn=nn.CrossEntropyLoss(weight=weights.cuda(),label_smoothing=.05)
    else:
        # Equal class probability; within a class equal room probability.
        groups={}
        for i in ti.tolist():groups[(int(y[i]),rows[i]['room'])]=groups.get((int(y[i]),rows[i]['room']),0)+1
        room_counts={c:len({r for a,r in groups if a==c}) for c in range(5)}
        weights=[1/(groups[(int(y[i]),rows[i]['room'])]*room_counts[int(y[i])]) for i in ti.tolist()]
        sampler=WeightedRandomSampler(weights,len(ti),replacement=True,generator=torch.Generator().manual_seed(seed))
        loader=DataLoader(train,batch_size=32,sampler=sampler)
        loss_fn=nn.CrossEntropyLoss(label_smoothing=.1)
    val_loader=DataLoader(val,batch_size=64)
    train_eval=DataLoader(train,batch_size=64)
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=4,min_lr=1e-5)
    config=dict(mode=mode,room=room,seed=seed,epochs=epochs,selection='val accuracy, tie val CE',
                train_windows=len(ti),val_windows=len(vi),transductive=False,
                source_sha256=hashlib.sha256(Path('improved_models.py').read_bytes()).hexdigest())
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    history=[];best=(-1.,-1e10);best_epoch=0
    for epoch in range(1,epochs+1):
        start=perf_counter();model.train();total=correct=0;loss_total=0.
        for a,b,c in loader:
            a,b,c=a.cuda(),b.cuda(),c.cuda();opt.zero_grad(set_to_none=True)
            logits=model.forward_features(a,b);loss=loss_fn(logits,c)
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);opt.step()
            total+=len(c);correct+=(logits.argmax(1)==c).sum().item();loss_total+=loss.item()*len(c)
        vl,va,logits=evaluate(model,val_loader);scheduler.step(vl)
        history.append(dict(epoch=epoch,train_loss=loss_total/total,train_accuracy_online=correct/total,
                            val_loss=vl,val_accuracy=va,seconds=perf_counter()-start))
        score=(va,-vl)
        if score>best:
            best=score;best_epoch=epoch
            torch.save(dict(model_state=model.state_dict(),config=config,epoch=epoch),folder/'best.pt')
        if epoch%5==0 or epoch==1: print(f'{mode}/{room}/{seed} epoch{epoch}: train={correct/total:.3f} val={va:.3f} best={best[0]:.3f}',flush=True)
        if epoch>=15 and epoch-best_epoch>=10:break
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True)['model_state'])
    vl,va,logits=evaluate(model,val_loader);_,ta,_=evaluate(model,train_eval)
    result=metrics(y[vi].numpy(),logits.argmax(1))
    result.update(mode=mode,room=room,seed=seed,best_epoch=best_epoch,epochs_run=len(history),
                  train_accuracy=ta,val_loss=vl,parameters=sum(p.numel() for p in model.parameters()),
                  macro_recall=float(np.mean([v for v in result['recall'].values() if v is not None])))
    (folder/'metrics.json').write_text(json.dumps(result,indent=2))
    write_csv(folder/'history.csv',history)
    write_csv(folder/'predictions.csv',[dict(identifier=rows[i]['identifier'],room=room,start=rows[i]['start'],
                target=int(y[i]),predicted=int(p),**{f'logit_{j}':float(v) for j,v in enumerate(logit)})
                for i,p,logit in zip(vi.tolist(),logits.argmax(1),logits)])
    print('DONE',mode,room,seed,va,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',default=['original','whiten','decorrelated','spectrum'])
    p.add_argument('--rooms',nargs='+',default=['a1','c1','g1']);p.add_argument('--seeds',nargs='+',type=int,default=[42]);p.add_argument('--epochs',type=int,default=45)
    p.add_argument('--output',type=Path,default=Path('artifacts/iterations_v2'));a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    bank=load_bank(phase=all(m.startswith('phase') for m in a.modes),tf=all(m.startswith('tfcnn') for m in a.modes));results=[];a.output.mkdir(parents=True,exist_ok=True)
    for mode in a.modes:
        for room in a.rooms:
            for seed in a.seeds:
                results.append(run(mode,room,seed,a.output,bank,a.epochs))
                (a.output/'summary.json').write_text(json.dumps(results,indent=2))
