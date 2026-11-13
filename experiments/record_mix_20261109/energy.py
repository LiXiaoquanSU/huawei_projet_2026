"""Conditional low/high count decomposition with matched dynamic-only control."""
import importlib.util,sys,random,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
spec=importlib.util.spec_from_file_location('record_mix_stage',Path(__file__).with_name('run.py'));stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
base=stage.base;old=stage.old;W=base.BASE/'conditional_energy_20261109'
class Encoder(stage.Model):
 def __init__(self):super().__init__();del self.head
 def forward(self,x,s):return self.features(x,s)
class Network(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.low=Encoder();self.high=Encoder();self.lowhead=nn.Sequential(nn.Linear(401,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.3),nn.Linear(128,3));self.highhead=nn.Sequential(nn.Linear(401,128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.3),nn.Linear(128,3))
 def forward(self,x):
  summary=x.new_zeros((len(x),17));dyn=old.input_view(x,'dynamic');high_input=old.input_view(x,'static') if self.mode=='static_dynamic_energy' else dyn
  strength=(x-x.mean(-1,keepdim=True)).reshape(len(x),3,3,32,150).square().mean((2,3,4)).add(1e-12).sqrt();low_summary=torch.cat([torch.log1p(1000*strength),summary[:,3:]],1)
  lo=self.lowhead(self.low(dyn,low_summary)).log_softmax(-1);hi=self.highhead(self.high(high_input,summary)).log_softmax(-1)
  return torch.cat([lo[:,:2],lo[:,2:]+hi],1)
def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W.mkdir(exist_ok=True);rows=base.load(base.CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda');x=torch.from_numpy(np.array(np.load(old.OLD/'joint.npy',mmap_mode='r'))).cuda();splits=base.load(base.WORK/'protocol.json')['splits'];results=[]
 base.save(W/'protocol.json',dict(variants=['static_dynamic_energy'],seed=42,epochs=12,lr=.0003,weight_decay=.01,batch=24,loss='Exact fiveclass conditional log probability NLL; low predicts0,1,>=2; high predicts2,3,4 given>=2',architecture='Two equal pair encoders; independent low and high3way heads. Compared to static_dynamic, only three low summary inputs change; parameter count unchanged. Low summary restores three log1p(1000*RMS) pair strengths; other14zero, high summaryzero. Fixed transform chosen before this experiment, no target fitting. No auxiliary loss or new bias.',selection='Fixed12, no outer checkpoint or threshold search. Formula chosen after prior development confusion audit; not blind validation.',motivation='Previous single-head static stronger3/4, dynamic stronger0/1. Recombining existing probabilities433/960 is exploratory, not a guarantee this factorization learns.',limitations='Conditional classes are labels, not physical static/moving human detectors.2person weak in both sources. All3record folds reused.',splits=splits))
 for fold,spec in splits.items():
  fi=spec['source'];te=spec['test'];assert {rows[i]['identifier'] for i in fi}.isdisjoint({rows[i]['identifier'] for i in te});w=base.weights(rows,y.cpu(),fi)
  for mode in ['static_dynamic_energy']:
   dest=W/f'{fold}_{mode}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():results.append(base.load(dest/'result.json'));continue
   random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);m=Network(mode).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);gen=torch.Generator().manual_seed(42);history=[]
   for epoch in range(1,13):
    m.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)];losses=[]
    for ix in draw.split(24):
     z=m(x[ix]);assert torch.allclose(z.logsumexp(-1),torch.zeros(len(ix),device='cuda'),atol=1e-5);loss=F.nll_loss(z,y[ix]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
    history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
    if epoch in [1,6,12]:print('TRAIN',fold,mode,history[-1],flush=True)
   m.eval();parts=[];fitcorrect=0
   with torch.no_grad():
    for lo in range(0,len(fi),24):ii=fi[lo:lo+24];fitcorrect+=int((m(x[ii]).argmax(-1)==y[ii]).sum())
    for lo in range(0,len(te),24):ii=te[lo:lo+24];parts.append(m(x[ii]).cpu())
   z=torch.cat(parts);labels=y[te].cpu();r=dict(fold=fold,mode=mode,history=history,parameters=sum(p.numel() for p in m.parameters()),metrics=base.metrics(z,labels),source_fit=dict(correct=fitcorrect,total=len(fi)));base.save(dest/'result.json',r);torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=12,mode=mode,fit_identifiers=sorted({rows[i]['identifier'] for i in fi})),dest/'final.pt');torch.save(dict(logits=z,labels=labels,indices=te),dest/'predictions.pt');results.append(r);base.save(W/'results.json',results);print('RESULT',fold,mode,r['metrics'],flush=True);del m,opt;gc.collect();torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
