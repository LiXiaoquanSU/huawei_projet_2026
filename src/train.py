"""Retained full-data baseline/SAM recipes. Cleanup does not run training."""
import argparse,json,random
from pathlib import Path
import blosc2,numpy as np,torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from model import StructuredModel,signal_targets
from utils import read_csv,write_csv,window,task_loss
ROOT=Path(__file__).resolve().parents[1]

def main(args):
    if args.output.exists():raise FileExistsError('Use a new output directory; preserve saved models')
    if not torch.cuda.is_available():raise RuntimeError('Training requires the configured CUDA environment')
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=StructuredModel('spectral','physical').cuda()
    rows=read_csv(ROOT/'manifests/train_windows.csv');assert len(rows)==960
    ts=[];ss=[];current=None
    with torch.no_grad():
        for start in range(0,len(rows),4):
            rr=rows[start:start+4];assert len({r['identifier'] for r in rr})==1
            identifier=rr[0]['identifier']
            if identifier!=current:
                csi=blosc2.load_array(str(args.data_root/'labeled'/f'CSI_{identifier}.b2'))
                ati=blosc2.load_array(str(args.data_root/'labeled'/f'ATI_{identifier}.b2'));current=identifier
            arrays=[window(csi,ati,r) for r in rr]
            x=torch.from_numpy(np.stack([v[0] for v in arrays])).cuda();dt=torch.from_numpy(np.stack([v[1] for v in arrays])).cuda()
            t,s=model.preprocess(x,dt);ts.append(t.cpu());ss.append(s.cpu())
    t,s=torch.cat(ts),torch.cat(ss);y=torch.tensor([int(r['headcount']) for r in rows]);targets=signal_targets(t)
    group=lambda r:r['identifier'] if r['room']=='unknown' else r['room']
    groups={}
    for i,r in enumerate(rows):groups[(int(y[i]),group(r))]=groups.get((int(y[i]),group(r)),0)+1
    counts={c:len({g for label,g in groups if label==c}) for c in range(5)}
    weights=[1/(groups[(int(y[i]),group(r))]*counts[int(y[i])]) for i,r in enumerate(rows)]
    sampler=WeightedRandomSampler(weights,960,generator=torch.Generator().manual_seed(42))
    loader=DataLoader(TensorDataset(t,s,y,targets),batch_size=32,sampler=sampler)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.01);history=[]
    args.output.mkdir(parents=True)
    for epoch in range(1,6):
        model.train();correct=total=0;losses=[]
        for x,summary,label,target in loader:
            x,summary,label,target=x.cuda(),summary.cuda(),label.cuda(),target.cuda()
            rng,crng=torch.get_rng_state(),torch.cuda.get_rng_state()
            optimizer.zero_grad(set_to_none=True);tasks=model.forward_tasks(x,summary);loss,_=task_loss(tasks,label,target,'physical')
            assert torch.isfinite(loss);loss.backward()
            if args.method=='sam':
                params=[p for p in model.parameters() if p.grad is not None]
                norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params]));assert torch.isfinite(norm)
                saved=[p.detach().clone() for p in params]
                with torch.no_grad():
                    for p in params:p.add_(p.grad,alpha=float(.05/(norm+1e-12)))
                optimizer.zero_grad(set_to_none=True);torch.set_rng_state(rng);torch.cuda.set_rng_state(crng)
                try:
                    robust,_=task_loss(model.forward_tasks(x,summary),label,target,'physical');assert torch.isfinite(robust);robust.backward()
                finally:
                    with torch.no_grad():
                        for p,v in zip(params,saved):p.copy_(v)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3);assert torch.isfinite(norm);optimizer.step()
            correct+=(tasks['logits'].argmax(1)==label).sum().item();total+=len(label);losses.append(loss.item())
        history.append(dict(epoch=epoch,sampled_train_accuracy=correct/total,loss=float(np.mean(losses))));print(history[-1],flush=True)
    config=dict(method=args.method,size='spectral',mode='physical',epochs=5,seed=42,records=48,windows=960,independent_accuracy=None,note='Consolidated raw-feature pipeline; original binaries/checkpoints remain authoritative; bit-identical retraining is not guaranteed')
    torch.save(dict(model_state=model.cpu().state_dict(),config=config),args.output/'final.pt')
    (args.output/'config.json').write_text(json.dumps(config,indent=2));write_csv(args.output/'history.csv',history)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['baseline','sam'],default='baseline');p.add_argument('--data-root',type=Path,default=Path('D:/huawei2026-data'));p.add_argument('--output',type=Path,required=True);main(p.parse_args())
