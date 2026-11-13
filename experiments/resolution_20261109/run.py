"""Separate static and normalized dynamic evidence under matched architecture."""
import sys,random,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ordinal_feedback_20261108'));from ratio_network import Model,base,W as OLD
W=base.BASE/'resolution_20261109'
class ResolutionModel(Model):
 def forward(self,x,s):
  n,_,f,t=x.shape;x=x.reshape(n,3,3,f,t);reverse=x*x.new_tensor([1,-1,-1])[None,None,:,None,None];x=torch.stack([x,reverse],2).reshape(-1,3,f,t);z=self.conv(x)
  z=torch.cat([z.mean(-1),z.std(-1,unbiased=False),z.amax(-1)],1)
  if z.shape[-1]!=4:z=F.adaptive_avg_pool1d(z,4)
  z=self.project(z.flatten(1)).reshape(n,6,192);v=torch.cat([z.mean(1),z.std(1,unbiased=False),s],1);return self.head(v).log_softmax(-1)
def input_view(x,mode):return x
def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True)
 rows=base.load(base.CACHE/'rows.json');assert base.load(OLD/'features_complete.json')['rows']==rows;y=torch.tensor([r['headcount'] for r in rows],device='cuda');summary=torch.zeros((len(rows),17),device='cuda');splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 base.save(W/'protocol.json',dict(variants=['time300','freq64'],architecture='Same convolution weights and parameter count as raw367/960 baseline; adaptive pooling frequency statistics to4 bins only when needed; summary17zero. Full trajectories without dynamic normalization.',seed=42,epochs=12,lr=.0003,weight_decay=.01,batch=24,loss='NLL',sampling='fixed class-room balanced replacement draw; original source window count per epoch',resolution='time300:freq32/time300 versus freq64:freq64/time150; raw baseline32/150 reused after exact forward equivalence check',selection='fixed12, no target bias or checkpoint selection',limitations='Reused development folds, not independent validation. Equal trainable parameter count and examples, but increased resolution has more compute and changes receptive field extent. No additional labels or external data.',splits=splits))
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];assert {rows[i]['identifier'] for i in fi}.isdisjoint({rows[i]['identifier'] for i in te});weights=base.weights(rows,y.cpu(),fi)
  for mode in ['time300','freq64']:
   x=torch.from_numpy(np.array(np.load(W/f'{mode}.npy',mmap_mode='r'))).cuda()
   dest=W/f'{fold}_{mode}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():results.append(base.load(dest/'result.json'));continue
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);m=ResolutionModel().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);gen=torch.Generator().manual_seed(42);history=[]
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
   r=dict(fold=fold,mode=mode,parameters=sum(p.numel() for p in m.parameters()),history=history,metrics=base.metrics(z,labels),rooms=rooms,source_fit=dict(correct=fitcorrect,total=len(fi)));base.save(dest/'result.json',r);torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},mode=mode,fit_identifiers=sorted({rows[i]['identifier'] for i in fi}),epoch=12),dest/'final.pt');torch.save(dict(logits=z,labels=labels,indices=te),dest/'predictions.pt');results.append(r);base.save(W/'results.json',results);print('RESULT',fold,mode,r['metrics'],flush=True);del m,opt,x;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
