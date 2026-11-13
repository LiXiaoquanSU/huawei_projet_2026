"""Separate static and normalized dynamic evidence under matched architecture."""
import sys,random,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ordinal_feedback_20261108'));from ratio_network import Model,base,W as OLD
W=base.BASE/'dynamic_evidence_20261108'
def input_view(x,mode):
 if mode=='static':return x.mean(-1,keepdim=True).expand_as(x)
 if mode=='dynamic':
  d=(x-x.mean(-1,keepdim=True)).reshape(-1,3,3,32,150);scale=(d.square().mean((2,3,4),keepdim=True)+1e-6).sqrt();return (d/scale).clamp(-6,6).reshape(-1,9,32,150)
 return x
def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True)
 rows=base.load(base.CACHE/'rows.json');assert base.load(OLD/'features_complete.json')['rows']==rows;y=torch.tensor([r['headcount'] for r in rows],device='cuda');x=torch.from_numpy(np.array(np.load(OLD/'joint.npy',mmap_mode='r'))).cuda();summary=torch.zeros((len(rows),17),device='cuda');splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 base.save(W/'protocol.json',dict(variants=['raw','static','dynamic'],architecture='same Model from complex_ratio_network; summary17 always zero for all variants',seed=42,epochs=12,lr=.0003,weight_decay=.01,batch=24,loss='NLL',sampling='fixed class-room balanced replacement draw; original source window count per epoch',dynamic='remove per-coordinate per-frequency time mean; divide by shared per-pair RMS over3coords/frequency/time with variance epsilon1e-6; clamp[-6,6]',static='repeat time mean over150positions',selection='fixed12, no target bias or checkpoint selection',limitations='Same reused development record folds. Static/dynamic are signal decompositions, not human activity labels. Within-room breakdown is not proof of causal occupancy evidence.',splits=splits))
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];assert {rows[i]['identifier'] for i in fi}.isdisjoint({rows[i]['identifier'] for i in te});weights=base.weights(rows,y.cpu(),fi)
  for mode in ['raw','static','dynamic']:
   dest=W/f'{fold}_{mode}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():results.append(base.load(dest/'result.json'));continue
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);m=Model().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);gen=torch.Generator().manual_seed(42);history=[]
   for epoch in range(1,13):
    m.train();draw=torch.tensor(fi)[torch.multinomial(weights,len(fi),replacement=True,generator=gen)];losses=[]
    for ix in draw.split(24):
     logits=m(input_view(x[ix],mode),summary[ix]);loss=F.nll_loss(logits,y[ix]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
    history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
    if epoch in [1,6,12]:print('TRAIN',fold,mode,history[-1],flush=True)
   m.eval();parts=[];fitcorrect=0
   with torch.no_grad():
    for lo in range(0,len(fi),24):ii=fi[lo:lo+24];fitcorrect+=int((m(input_view(x[ii],mode),summary[ii]).argmax(-1)==y[ii]).sum())
    for lo in range(0,len(te),24):ii=te[lo:lo+24];parts.append(m(input_view(x[ii],mode),summary[ii]).cpu())
   z=torch.cat(parts);labels=y[te].cpu();rooms={}
   for room in sorted({rows[i]['room'] for i in te}):
    ix=[k for k,i in enumerate(te) if rows[i]['room']==room];rooms[room]=base.metrics(z[ix],labels[ix])
   r=dict(fold=fold,mode=mode,parameters=sum(p.numel() for p in m.parameters()),history=history,metrics=base.metrics(z,labels),rooms=rooms,source_fit=dict(correct=fitcorrect,total=len(fi)));base.save(dest/'result.json',r);torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},mode=mode,fit_identifiers=sorted({rows[i]['identifier'] for i in fi}),epoch=12),dest/'final.pt');torch.save(dict(logits=z,labels=labels,indices=te),dest/'predictions.pt');results.append(r);base.save(W/'results.json',results);print('RESULT',fold,mode,r['metrics'],flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
