"""Test whether ordinal head information survives the established054 policy."""
import sys,gc,argparse
from pathlib import Path
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'));import g60_common as old
sys.path.insert(0,str(ROOT/'experiments/nested_context_20261107'));import run as base
W=base.BASE/'ordinal_policy_20261107'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--actual',action='store_true');args=parser.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True)
 data=torch.load(old.CACHE,weights_only=False,map_location='cpu');reverse=torch.load(old.OLD/'reverse_features.pt',weights_only=False,map_location='cpu');assert data['rows']==reverse['rows'];rows=data['rows'];y=data['y'];t=torch.stack([data['t'],reverse['t']],1).cuda();s=torch.stack([data['s'],reverse['s']],1).cuda();del data,reverse;perms=[tuple(v.cuda() for v in old.permutation_indices(p)) for p in old.PERMS];results=[]
 base.save(W/'protocol.json',dict(hypothesis='Context recordfold audit: ordinal487 versus mixed477, gains in all3folds. Test transfer to original037 architecture with old054 room-excluded proxies.',methods=['baseline','ordinal','ordinal75'],selection='fixed formulas; frozen old source calibration and054 routing; no target fitting or new bias',limitations='Four reused leave-room development proxies; not actual full054 platform accuracy. No guarantee calibration optimal for changed head.'))
 for room in (['actual'] if args.actual else ['a1','c1','g1','e1']):
  dest=W/room;dest.mkdir(exist_ok=True)
  if (dest/'results.json').exists():results.extend(base.load(dest/'results.json'));continue
  fi,cal,te=old.split(room,rows);cp=old.checkpoint(room);m,_=old.build(cp,full=room=='actual');m=m.cuda().eval();bias,temp=old.calibration(room);raw={k:[] for k in ['baseline','ordinal','ordinal75']}
  with torch.no_grad():
   for start in range(0,len(te),16):
    ix=te[start:start+16];tt=t[ix].flatten(0,1);ss=s[ix].flatten(0,1);parts={k:[] for k in raw}
    for pi,sg,si in perms:
     task=m.forward_tasks(tt.index_select(1,pi)*sg,ss.index_select(1,si));parts['baseline'].append(task['logits'].cpu().reshape(-1,2,5));parts['ordinal'].append(task['ordinal'].clamp_min(1e-8).log().cpu().reshape(-1,2,5));parts['ordinal75'].append((.75*task['ordinal']+.25*task['direct']).clamp_min(1e-8).log().cpu().reshape(-1,2,5))
    for k in raw:raw[k].append(torch.stack(parts[k],1))
  raw={k:torch.cat(v) for k,v in raw.items()};cached=torch.load(old.WORK/f'{room}.pt',weights_only=False,map_location='cpu');assert [r['identifier'] for r in cached['rows']]==[r['identifier'] for r in rows]
  error=float((raw['baseline']-cached['raw'][te]).abs().max());assert error<.005,error
  out=[];preds={}
  for method,z in raw.items():
   prediction=old.policy(z,bias,temp);r=dict(room=room,method=method,metrics=base.metrics(prediction.log_softmax(-1),y[te]),baseline_reproduction_error=error);out.append(r);preds[method]=prediction;print('RESULT',room,method,r['metrics']['correct'],'/',len(te),flush=True)
  torch.save(dict(raw=raw,predictions=preds,indices=te,labels=y[te]),dest/'predictions.pt');base.save(dest/'results.json',out);results.extend(out);base.save(W/'results.json',results);del m;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
