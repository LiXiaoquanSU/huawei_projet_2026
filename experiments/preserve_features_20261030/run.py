"""Matched continuation from source-selected early weights; no target selection."""
import sys,copy,random,json,gc
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from train_chain import EvidenceChain,predict,metric,sample_weights,invariance,old,CACHE
BASE=Path('D:/huawei2026-experiments/texture_chain_20261028')
WORK=Path('D:/huawei2026-experiments/preserve_features_20261030')
EPOCHS=8

def main():
 torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 WORK.mkdir(parents=True,exist_ok=True)
 old.save(WORK/'protocol.json',dict(variants=['joint','frozen','anchored'],epochs=EPOCHS,seed=42,lr=.0001,weight_decay=.02,anchor='1.0 times mean squared difference divided by per-sample teacher feature variance; prehead1664, fit only',selection='minimum source calibration balanced NLL including unchanged epoch0; report fixed8 also',limits='Reused dev folds; earlier checkpoint itself source-selected; not independent test. Same init/draw/dropout seed; frozen backbone same architecture. No U/external/platform labels.',code_sha256=old.sha(Path(__file__))))
 rows=old.read(CACHE/'rows.json');y=torch.tensor([int(r['headcount']) for r in rows],device='cuda')
 t=torch.from_numpy(np.array(np.load(CACHE/'t.npy',mmap_mode='r'))).cuda();s=torch.from_numpy(np.array(np.load(CACHE/'s.npy',mmap_mode='r'))).cuda()
 for fold in range(3):
  source=BASE/f'record{fold}_full_s42';d=torch.load(source/'predictions.pt',weights_only=False,map_location='cpu');state=torch.load(source/'final.pt',weights_only=False,map_location='cpu')
  fi,cal,te=[d[k] for k in ('fit_indices','calibration_indices','test_indices')]
  sets=[{rows[i]['identifier'] for i in ix} for ix in (fi,cal,te)];assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
  teacher=EvidenceChain().cuda().eval();teacher.load_state_dict(state['model_state']);cap={}
  handle=teacher.head.register_forward_pre_hook(lambda mod,args:cap.update(x=args[0].detach()))
  reference=torch.zeros((len(rows),2,1664),device='cuda')
  with torch.inference_mode():
   for offset in range(0,len(fi),24):
    jj=fi[offset:offset+24];teacher(t[jj].flatten(0,1),s[jj].flatten(0,1));reference[jj]=cap['x'].reshape(-1,2,1664)
  handle.remove();del teacher
  for variant in ('joint','frozen','anchored'):
   folder=WORK/f'record{fold}_{variant}_s42';folder.mkdir(exist_ok=True)
   if (folder/'result.json').exists():continue
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
   m=EvidenceChain().cuda();m.load_state_dict(state['model_state']);capture={}
   hook=m.head.register_forward_pre_hook(lambda mod,args:capture.update(x=args[0]))
   if variant=='frozen':
    for name,param in m.named_parameters():param.requires_grad_(name.startswith('head.'))
   opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0001,weight_decay=.02)
   sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00001)
   w=sample_weights(rows,y.cpu(),fi);gen=torch.Generator().manual_seed(42)
   def assess():
    z=predict(m,t,s,cal);yy=y[cal].cpu();cw=1/torch.bincount(yy,minlength=5)[yy].float();cw/=cw.sum()
    return float((-z[range(len(cal)),yy]*cw).sum())
   err=float((predict(m,t,s,list(range(len(rows)))).exp()-d['logits'].exp()).abs().max());assert err<1e-5,err
   best=dict(epoch=0,nll=assess(),state=copy.deepcopy(state['model_state']));history=[]
   for epoch in range(1,EPOCHS+1):
    m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)];losses=[];drifts=[]
    for jj in draw.split(24):
     z=m(t[jj].flatten(0,1),s[jj].flatten(0,1));x=capture['x'];target=reference[jj].flatten(0,1)
     drift=((x-target).square().mean(-1)/target.var(-1,unbiased=False).clamp_min(1e-4)).mean()
     p=z.exp().reshape(-1,2,5);q=p.mean(1,keepdim=True)
     loss=F.nll_loss(z,y[jj,None].expand(-1,2).flatten())+.05*(p*(p.clamp_min(1e-8).log()-q.clamp_min(1e-8).log())).sum(-1).mean()
     if variant=='anchored':loss=loss+drift
     assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()));drifts.append(float(drift.detach()))
    sched.step();nll=assess();history.append(dict(epoch=epoch,loss=float(np.mean(losses)),drift=float(np.mean(drifts)),calibration_nll=nll))
    if nll<best['nll']:best=dict(epoch=epoch,nll=nll,state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()}))
    if epoch in (1,4,8):print('TRAIN',fold,variant,epoch,'cal',round(nll,4),'drift',round(np.mean(drifts),4),flush=True)
   last=predict(m,t,s,list(range(len(rows))));torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=EPOCHS),folder/'last.pt')
   m.load_state_dict(best['state']);selected=predict(m,t,s,list(range(len(rows))));inv=invariance(m,t[:2,0],s[:2,0]);hook.remove()
   torch.save(dict(model_state=best['state'],epoch=best['epoch'],source_sha256=old.sha(source/'final.pt')),folder/'final.pt')
   torch.save(dict(logits=selected,last_logits=last,labels=y.cpu(),rows=rows,fit_indices=fi,calibration_indices=cal,test_indices=te),folder/'predictions.pt')
   r=dict(fold=fold,variant=variant,epoch=best['epoch'],calibration_nll=best['nll'],test=metric(selected[te],y[te].cpu()),last_test=metric(last[te],y[te].cpu()),history=history,invariance=inv,reproduction_error=err,source_epoch=state['epoch'])
   old.save(folder/'result.json',r);print('DONE',fold,variant,r['epoch'],r['test']['correct'],r['last_test']['correct'],flush=True)
   del m,opt;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
