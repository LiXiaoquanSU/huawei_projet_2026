"""Audit training execution differences against archived037 recipe, fixed budget."""
import sys,random,gc,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/nested_context_20261107'));import run as base
W=base.BASE/'rng_pipeline_20261108'
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True)
 rows=base.load(base.CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda');t=torch.from_numpy(np.array(np.load(base.CACHE/'t.npy',mmap_mode='r')[:,0])).cuda();s=torch.from_numpy(np.array(np.load(base.CACHE/'s.npy',mmap_mode='r')[:,0])).cuda();targets=base.signal_targets(t);splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 base.save(W/'protocol.json',dict(architecture='original037 spectral physical',variants=['continue_rng','native_loader'],control='nested_context baseline epoch5 manual float32 sampler and post-init seed reset',seed=42,epochs=5,lr=.0004,weight_decay=.01,batch=32,loss='original full physical objective',selection='Fixed5 only. No target-based checkpoint or calibration. No seed search.',differences={'continue_rng':'same manual float32 sampling as control, but keep post-initialization global RNG state','native_loader':'keep RNG state; DataLoader num_workers=0, WeightedRandomSampler double weights derived from original group counts; more faithful archived iteration protocol'},limits='Not exact historic run reproduction: current audited ONNX-feature cache, deterministic settings and consolidated model code. Three reused record folds, no hidden accuracy.',splits=splits))
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];fitids=sorted({rows[i]['identifier'] for i in fi});assert not set(fitids)&{rows[i]['identifier'] for i in te};w=base.weights(rows,y.cpu(),fi)
  keys=[(int(y[i]),rows[i]['identifier'] if rows[i]['room']=='unknown' else rows[i]['room']) for i in fi];counts={k:keys.count(k) for k in set(keys)};domains={c:sum(k[0]==c for k in counts) for c in range(5)};native_w=[1/(counts[k]*domains[k[0]]) for k in keys]
  for variant in ['continue_rng','native_loader']:
   dest=W/f'{fold}_{variant}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():results.append(base.load(dest/'result.json'));continue
   seed();m=base.build('baseline').cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0004,weight_decay=.01);gen=torch.Generator().manual_seed(42);history=[];all_draws=[]
   if variant=='native_loader':loader=DataLoader(TensorDataset(torch.tensor(fi)),batch_size=32,sampler=WeightedRandomSampler(native_w,len(fi),replacement=True,generator=gen),num_workers=0)
   for epoch in range(1,6):
    m.train();losses=[];draw=[]
    batches=(batch[0] for batch in loader) if variant=='native_loader' else torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)].split(32)
    for ix in batches:
     draw.extend(ix.tolist());task=m.forward_tasks(t[ix],s[ix]);loss,_=base.task_loss(task,y[ix],targets[ix],'physical');assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
    all_draws.append(draw);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),sampled_indices_sha256=hashlib.sha256(np.array(draw,dtype=np.int64).tobytes()).hexdigest()));print('TRAIN',fold,variant,history[-1],flush=True)
   m.eval();parts=[]
   with torch.no_grad():
    for start in range(0,len(te),32):ix=te[start:start+32];parts.append(m.forward_features(t[ix],s[ix]).cpu())
   z=torch.cat(parts);yy=y[te].cpu();r=dict(fold=fold,variant=variant,history=history,metrics=base.metrics(z,yy));torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=5,fit_identifiers=fitids),dest/'final.pt');torch.save(dict(logits=z,labels=yy,indices=te,fit_identifiers=fitids,sampled_indices=all_draws),dest/'predictions.pt');base.save(dest/'result.json',r);results.append(r);base.save(W/'results.json',results);print('RESULT',fold,variant,r['metrics'],flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
