"""Two-seed nonlinear readability control on fixed source-only dictionaries."""
import sys,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/texture_chain_20261028'))
from train_chain import old,sample_weights,metric
WORK=Path('D:/huawei2026-experiments/distribution_pool_20261030');BASE=Path('D:/huawei2026-experiments/texture_chain_20261028')

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=WORK/'nonlinear';out.mkdir(exist_ok=True)
 old.save(out/'protocol.json',dict(seeds=[42,2026],epochs=40,modes=['old','moments','distribution','old_plus_distribution'],head='input ->128 LayerNorm SiLU Dropout0.3 ->64 SiLU ->5',lr=.001,weight_decay=.02,sampling='class room record window balanced',selection='source calibration balanced NLL; fixed40 also reported',limits='Backbone and dictionaries fixed; two seeds vary head only, not backbone. Different feature dimensions mean not matched capacity. Reused development folds.'))
 results=[]
 for fold in range(3):
  d=torch.load(BASE/f'record{fold}_full_s42/predictions.pt',weights_only=False,map_location='cpu');fi,cal,te=[d[k] for k in ('fit_indices','calibration_indices','test_indices')];y=d['labels'].cuda();w=sample_weights(d['rows'],d['labels'],fi);features=np.load(WORK/f'record{fold}_features.npz')
  for mode in ('old','moments','distribution','old_plus_distribution'):
   raw=features[mode];scaler=StandardScaler().fit(raw[fi]);x=torch.tensor(scaler.transform(raw),device='cuda',dtype=torch.float32)
   for seed in (42,2026):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);gen=torch.Generator().manual_seed(seed)
    model=nn.Sequential(nn.Linear(x.shape[1],128),nn.LayerNorm(128),nn.SiLU(),nn.Dropout(.3),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,5)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,40,eta_min=.0001);best=None;history=[]
    for epoch in range(1,41):
     model.train();draw=torch.tensor(fi)[torch.multinomial(w,len(fi),replacement=True,generator=gen)]
     for ix in draw.split(32):
      loss=nn.functional.cross_entropy(model(x[ix]),y[ix]);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
     sched.step();model.eval()
     with torch.no_grad():
      z=model(x[cal]).log_softmax(-1);yy=y[cal];cw=1/torch.bincount(yy,minlength=5)[yy].float();cw/=cw.sum();nll=float((-z[range(len(cal)),yy]*cw).sum())
     history.append(dict(epoch=epoch,calibration_nll=nll))
     if best is None or nll<best['nll']:best=dict(epoch=epoch,nll=nll,state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()}))
    with torch.no_grad():last=model(x).log_softmax(-1).cpu()
    model.load_state_dict(best['state'])
    with torch.no_grad():z=model(x).log_softmax(-1).cpu()
    key=f'record{fold}_{mode}_s{seed}';r=dict(fold=fold,mode=mode,seed=seed,epoch=best['epoch'],calibration_nll=best['nll'],test=metric(z[te],d['labels'][te]),last_test=metric(last[te],d['labels'][te]),history=history);results.append(r)
    torch.save(dict(model_state=best['state'],scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,epoch=best['epoch']),out/f'{key}_head.pt');torch.save(dict(logits=z,last_logits=last,test_indices=te),out/f'{key}_predictions.pt')
    old.save(out/'results.json',dict(results=results));print(key,r['epoch'],r['test']['correct'],r['last_test']['correct'],flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
