"""Refit one frozen structured hypothesis on all 48 labelled traces."""
import argparse,json,random,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
import blosc2
from structured_model import StructuredModel,load_structured_bank,signal_targets
from train_structured import task_loss
from train_baseline import read_csv,window,write_csv
from export_structured import export

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['small','large','spectral'],required=True);p.add_argument('--mode',choices=['direct','ordinal','auxiliary','physical'],required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'final.pt').exists():raise FileExistsError('Preserve refit; use export_structured.py to re-export')
    epochs=[json.loads(Path(f'artifacts/structured_20261019/balanced/{args.size}_{args.mode}_{r}_seed42/metrics.json').read_text())['best_epoch'] for r in ['a1','c1','g1']]
    budget=int(np.median(epochs));model=StructuredModel(args.size,args.mode).cuda()
    t,s,y,rows=load_structured_bank(128 if args.size=='spectral' else 32);extra_t=[];extra_s=[];extra_rows=[]
    with torch.no_grad():
        for row in read_csv('D:/huawei2026-data/labeled/samples.csv'):
            if row['room']!='unknown':continue
            csi=blosc2.load_array(f'D:/huawei2026-data/labeled/CSI_{row["identifier"]}.b2');ati=blosc2.load_array(f'D:/huawei2026-data/labeled/ATI_{row["identifier"]}.b2')
            for start in range(0,12000,2400):
                entries=[dict(row,start=i,stop=i+600) for i in range(start,start+2400,600)]
                arrays=[window(csi,ati,r) for r in entries]
                a=torch.from_numpy(np.stack([v[0] for v in arrays])).cuda();dt=torch.from_numpy(np.stack([v[1] for v in arrays])).cuda()
                tt,ss=model.preprocess(a,dt);extra_t.append(tt.cpu());extra_s.append(ss.cpu());extra_rows.extend(entries)
    t=torch.cat([t,*extra_t]);s=torch.cat([s,*extra_s]);y=torch.cat([y,torch.tensor([int(r['headcount']) for r in extra_rows])]);rows+=extra_rows
    assert len(rows)==960 and len(set(r['identifier'] for r in rows))==48
    targets=signal_targets(t)
    def group(r):return r['room'] if r['room']!='unknown' else r['identifier']
    groups={}
    for i,r in enumerate(rows):groups[(int(y[i]),group(r))]=groups.get((int(y[i]),group(r)),0)+1
    counts={c:len({g for a,g in groups if a==c}) for c in range(5)}
    weights=[1/(groups[(int(y[i]),group(r))]*counts[int(y[i])]) for i,r in enumerate(rows)]
    loader=DataLoader(TensorDataset(t,s,y,targets),batch_size=32,sampler=WeightedRandomSampler(weights,len(y),generator=torch.Generator().manual_seed(42)))
    opt=torch.optim.AdamW(model.parameters(),lr=.0007 if args.size=='small' else .0004,weight_decay=.01)
    history=[]
    for epoch in range(1,budget+1):
        model.train();correct=total=0;losses=[]
        for x,summary,label,target in loader:
            x,summary,label,target=x.cuda(),summary.cuda(),label.cuda(),target.cuda()
            opt.zero_grad(set_to_none=True);tasks=model.forward_tasks(x,summary);loss,parts=task_loss(tasks,label,target,args.mode)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
            correct+=(tasks['logits'].argmax(1)==label).sum().item();total+=len(label);losses.append(loss.item())
        history.append(dict(epoch=epoch,sampled_train_accuracy=correct/total,objective=float(np.mean(losses))))
        print('FULL REFIT',args.size,args.mode,epoch,budget,correct/total,flush=True)
    config=dict(size=args.size,mode=args.mode,seed=42,epochs=budget,fold_best_epochs=epochs,records=48,windows=960,independent_accuracy=None,room_area_labels=False,human_motion_labels=False,signal_proxy='input-derived coherence')
    torch.save(dict(model_state=model.cpu().state_dict(),config=config),args.output/'final.pt')
    (args.output/'config.json').write_text(json.dumps(config,indent=2));write_csv(args.output/'history.csv',history)
    export(model,args.output)
