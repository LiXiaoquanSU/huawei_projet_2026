import sys,json,gc,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/g60_20261025'));import g60_common as g
D=Path('D:/huawei2026-experiments');W=D/'branch_audit_20261110'
def main():
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;W.mkdir(exist_ok=True)
 rows=g.read(D/'next053_20261025/rows.json');ix=[i for i,r in enumerate(rows) if int(r['start'])==0];tt=[torch.from_numpy(np.array(np.load(D/'next053_20261025/t.npy',mmap_mode='r')[ix]))];ss=[torch.from_numpy(np.array(np.load(D/'next053_20261025/s.npy',mmap_mode='r')[ix]))];meta=[dict(r,domain='S') for i,r in enumerate(rows) if i in ix]
 for path in sorted((D/'u_ssl_20261025/u_features').glob('*.pt')):
  q=torch.load(path,map_location='cpu',weights_only=True);tt.append(q['t'][:1]);ss.append(q['s'][:1]);meta.append(dict(q['rows'][0],domain='U'))
 assert len(meta)==198;t=torch.cat(tt).cuda();s=torch.cat(ss).cuda();g.save(W/'protocol.json',dict(rows=meta,selection='First nonoverlap window per48S/150U record, no labels used to select',precision='Each network rounded to original044 fp16 storage pattern. S current float32 ONNX preprocessing; U archived float64->float32 preprocessing. Same inputs acrossmodels; descriptive comparison, not deploy reproduction.',limits='S training-seen for all models; U no labels, unknown environment mix. No claim of independent accuracy. Original vsseed changes many training details, not isolated seed causal test.'))
 checkpoints={'original':ROOT/'models/best_037/final.pt'}
 for seed in [42,0,304]:checkpoints[f'seed{seed}']=ROOT/f'2026-10-25_037配方与种子复现/weights/checkpoints/base037_full_s{seed}.pt'
 summary={}
 for name,path in checkpoints.items():
  m,bias=g.build(path);m=m.cuda().eval();parts={k:[] for k in ['logits','direct','ordinal','experts','gate']}
  with torch.no_grad():
   for perm in g.PERMS:
    idx,sign,si=g.permutation_indices(perm);idx,sign,si=idx.cuda(),sign.cuda(),si.cuda()
    for rev in range(2):
     chunks={k:[] for k in parts}
     for lo in range(0,len(t),16):
      q=m.forward_tasks(t[lo:lo+16,rev].index_select(1,idx)*sign,s[lo:lo+16,rev].index_select(1,si))
      for k in chunks:chunks[k].append(q[k].cpu())
     for k in parts:parts[k].append(torch.cat(chunks[k]))
  data={k:torch.stack(v,1) for k,v in parts.items()};raw=data['logits'].reshape(len(t),6,2,5);data['policy']=g.policy(raw,bias);data['meta']=meta;torch.save(data,W/f'{name}.pt')
  info={}
  for domain in ['S','U']:
   ii=[i for i,r in enumerate(meta) if r['domain']==domain];a={k:v[ii] for k,v in data.items() if torch.is_tensor(v)};out={}
   for head in ['logits','direct','ordinal','policy']:
    p=a[head] if head=='policy' else a[head][:,0];pred=p.argmax(-1);out[head]=dict(hist=torch.bincount(pred,minlength=5).tolist())
    if domain=='S':out[head]['seen_correct']=int((pred==torch.tensor([meta[i]['headcount'] for i in ii])).sum())
   out['gate_mean']=a['gate'][:,0].mean(0).tolist();out['expert_hist']=[torch.bincount(a['experts'][:,0,k].argmax(-1),minlength=5).tolist() for k in range(4)];out['view_agreement_to_identity']=float((a['logits'].argmax(-1)==a['logits'][:,0].argmax(-1)[:,None]).float().mean());out['direct_ordinal_agreement']=float((a['direct'][:,0].argmax(-1)==a['ordinal'][:,0].argmax(-1)).float().mean());info[domain]=out
  summary[name]=dict(checkpoint=str(path),sha256=g.sha(path),domains=info);g.save(W/'summary.json',summary);print('AUDIT',name,info['U'],flush=True);del m;gc.collect();torch.cuda.empty_cache()
 g.save(ROOT/'experiments/branch_audit_20261110/results.json',summary);print('COMPLETE',flush=True)
if __name__=='__main__':main()
