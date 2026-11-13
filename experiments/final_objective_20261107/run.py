"""Frozen recipe control: supervise the deployed mixture, not only its branches."""
import sys,random,gc,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/nested_context_20261107'))
import run as base
W=base.BASE/'final_objective_20261107'
def main():
 global W
 pa=argparse.ArgumentParser();pa.add_argument('--seed',type=int,default=42);pa.add_argument('--variants',nargs='+',default=['fused_supervision','final_only']);pa.add_argument('--full',action='store_true');args=pa.parse_args()
 if args.seed!=42:W=Path(str(W)+f'_seed{args.seed}')
 if args.full:W=Path(str(W)+'_full')
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(parents=True,exist_ok=True)
 rows=base.load(base.CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda');t=torch.from_numpy(np.array(np.load(base.CACHE/'t.npy',mmap_mode='r')[:,0])).cuda();s=torch.from_numpy(np.array(np.load(base.CACHE/'s.npy',mmap_mode='r')[:,0])).cuda();target=base.signal_targets(t)
 splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 if args.full:splits={'full':dict(source=list(range(len(rows))),test=list(range(len(rows))))}
 base.save(W/'protocol.json',dict(architecture='context_events',seed=args.seed,epochs=5,lr=.0004,weight_decay=.01,batch=32,variants=args.variants,selection='fixed5; outer labels do not select epoch, calibration, bias or loss weights',full_data_fit=args.full,losses={'fused_supervision':'original objective, direct NLL replaced by final fused NLL; other four terms unchanged','final_only':'final fused NLL + 0.1 signal proxy MSE'},sampling='same class-room balanced recipe as nested_context',splits=splits,limitations='Reused development record folds; not blind or platform validation; full run if requested is training diagnostic only.'))
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];fitids=sorted({rows[i]['identifier'] for i in fi});testids=sorted({rows[i]['identifier'] for i in te})
  if not args.full:assert not set(fitids)&set(testids)
  weights=base.weights(rows,y.cpu(),fi)
  for variant in args.variants:
   dest=W/f'{fold}_{variant}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():results.append(base.load(dest/'result.json'));continue
   seed=args.seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=base.build('context_events').cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0004,weight_decay=.01);gen=torch.Generator().manual_seed(seed)
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);history=[]
   for epoch in range(1,6):
    m.train();draw=torch.tensor(fi)[torch.multinomial(weights,len(fi),replacement=True,generator=gen)];losses=[]
    for ix in draw.split(32):
     tasks=m.forward_tasks(t[ix],s[ix]);final=nn.functional.nll_loss(tasks['logits'],y[ix])
     if variant=='fused_supervision':
      original,_=base.task_loss(tasks,y[ix],target[ix],'physical');direct=nn.functional.nll_loss(tasks['direct'].clamp_min(1e-8).log(),y[ix]);loss=original-direct+final
     elif variant=='final_only':loss=final+.1*nn.functional.mse_loss(tasks['signal'],target[ix])
     elif variant=='control':loss,_=base.task_loss(tasks,y[ix],target[ix],'physical')
     else:raise ValueError(variant)
     assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
    history.append(dict(epoch=epoch,loss=float(np.mean(losses))));print('TRAIN',fold,variant,history[-1],flush=True)
   torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=5,fit_identifiers=fitids,seed=seed,variant=variant),dest/'final.pt')
   m.eval();parts={k:[] for k in ['logits','direct','ordinal']}
   with torch.no_grad():
    for start in range(0,len(te),32):
     ix=te[start:start+32];out=m.forward_tasks(t[ix],s[ix])
     for k in parts:parts[k].append(out[k].cpu())
   outputs={k:torch.cat(v) for k,v in parts.items()};labels=y[te].cpu();r=dict(fold=fold,variant=variant,seed=seed,history=history,metrics=base.metrics(outputs['logits'],labels),components={k:base.metrics(outputs[k].clamp_min(1e-8).log(),labels) for k in ['direct','ordinal']},evaluation='training fit only' if args.full else 'held-out complete records')
   torch.save(dict(**outputs,labels=labels,indices=te,fit_identifiers=fitids),dest/'predictions.pt');base.save(dest/'result.json',r);results.append(r);base.save(W/'results.json',results);print('RESULT',fold,variant,r['metrics'],flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
 base.save(W/'results.json',results);print('COMPLETE',flush=True)
if __name__=='__main__':main()
