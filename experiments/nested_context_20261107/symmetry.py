"""Fixed-five-epoch context control versus matched compute symmetry augmentation."""
import sys,random,gc,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
import run as base
sys.path.insert(0,str(base.ROOT/'experiments/tta_policy_20261024'))
from common import permutation_indices
from itertools import permutations
W=base.BASE/'context_symmetry_20261107'
def main():
 global W
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=42);parser.add_argument('--sampling',choices=['room','class'],default='room');parser.add_argument('--variants',nargs='+',default=['identity','time_reverse','rx_time']);args=parser.parse_args();seed=args.seed
 if seed!=42:W=base.BASE/f'context_symmetry_20261107_seed{seed}'
 if args.sampling=='class':W=Path(str(W)+'_class_sampling')
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 W.mkdir(parents=True,exist_ok=True);rows=base.load(base.CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda')
 t=torch.from_numpy(np.array(np.load(base.CACHE/'t.npy',mmap_mode='r'))).cuda();s=torch.from_numpy(np.array(np.load(base.CACHE/'s.npy',mmap_mode='r'))).cuda()
 perms=[tuple(v.cuda() for v in permutation_indices(p)) for p in permutations(range(3))]
 splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 base.save(W/'protocol.json',dict(variants=args.variants,sampling=args.sampling,architecture='context_events',seed=seed,epochs=5,lr=.0004,weight_decay=.01,batch=32,selection='fixed5 only; no outer checkpoint or bias selection',training='One view per sample: identity or random direction, and for rx_time one random RX permutation per minibatch. Identical number of sampled examples and optimizer steps. Original physical loss, recomputed proxy targets on transformed inputs.',evaluation='identity and uniform12 view probability means; uncalibrated',splits=splits,limitations='Repeated development record folds, not an untouched test or platform score. No full-data054 weights used.'))
 for fold in range(3):
  spec=splits[f'record{fold}'];fi=spec['source'];te=spec['test'];w=base.weights(rows,y.cpu(),fi)
  if args.sampling=='class':
   yy=y[fi].cpu();w=1/torch.bincount(yy,minlength=5)[yy].float();w/=w.sum()
  for variant in args.variants:
   folder=W/f'record{fold}_{variant}';folder.mkdir(exist_ok=True)
   if (folder/'result.json').exists():results.extend(base.load(folder/'result.json'));continue
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=base.build('context_events').cuda()
   opt=torch.optim.AdamW(m.parameters(),lr=.0004,weight_decay=.01);gen=torch.Generator().manual_seed(seed);rng=np.random.default_rng(seed)
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   if variant=='identity' and seed==42 and args.sampling=='room':
    saved=torch.load(base.WORK/f'record{fold}_context_events_outer/epoch5.pt',weights_only=False,map_location='cpu');m.load_state_dict(saved['model_state']);history=[]
   else:
    history=[]
    for epoch in range(1,6):
     m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)];losses=[]
     for ix in draw.split(32):
      direction=torch.zeros(len(ix),dtype=torch.long,device='cuda') if variant=='identity' else torch.tensor(rng.integers(2,size=len(ix)),device='cuda');tt=t[ix,direction];ss=s[ix,direction]
      if variant=='rx_time':
       pi,sg,si=perms[int(rng.integers(6))];tt=tt.index_select(1,pi)*sg;ss=ss.index_select(1,si)
      tasks=m.forward_tasks(tt,ss);loss,_=base.task_loss(tasks,y[ix],base.signal_targets(tt),'physical');assert torch.isfinite(loss)
      opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
     history.append(dict(epoch=epoch,loss=float(np.mean(losses))));print('TRAIN',fold,variant,history[-1],flush=True)
    torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=5,fit_identifiers=sorted({rows[i]['identifier'] for i in fi})),folder/'final.pt')
   m.eval();raw=[]
   with torch.no_grad():
    for start in range(0,len(te),16):
     ix=te[start:start+16];tt=t[ix].flatten(0,1);ss=s[ix].flatten(0,1);views=[]
     for pi,sg,si in perms:views.append(m.forward_features(tt.index_select(1,pi)*sg,ss.index_select(1,si)).cpu().reshape(-1,2,5))
     raw.append(torch.stack(views,1))
   raw=torch.cat(raw);yy=y[te].cpu();out=[]
   for mode,z in [('identity',raw[:,0,0]),('time2',raw[:,0].exp().mean(1).clamp_min(1e-8).log()),('all12',raw.exp().mean((1,2)).clamp_min(1e-8).log())]:
    r=dict(fold=fold,variant=variant,mode=mode,metrics=base.metrics(z,yy),history=history);out.append(r);print('RESULT',fold,variant,mode,r['metrics']['correct'],flush=True)
   torch.save(dict(raw=raw,labels=yy,indices=te),folder/'predictions.pt');base.save(folder/'result.json',out);results.extend(out);base.save(W/'results.json',results)
   del m,opt;gc.collect();torch.cuda.empty_cache()
 base.save(W/'results.json',results);print('COMPLETE',flush=True)
if __name__=='__main__':main()
