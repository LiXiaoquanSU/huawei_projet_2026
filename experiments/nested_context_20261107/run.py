"""Nested source-only epoch selection for037 versus039; disjoint-record OOF predictions."""
import sys,json,copy,hashlib,random,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments/global_local_20261021'))
from network import build
from model import signal_targets
from utils import task_loss
BASE=Path('D:/huawei2026-experiments');WORK=BASE/'nested_context_20261107';CACHE=BASE/'next053_20261025'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def metrics(z,y):
 p=z.argmax(-1);cm=torch.bincount(y*5+p,minlength=25).reshape(5,5);cw=1/torch.bincount(y,minlength=5)[y].float();cw/=cw.sum()
 return dict(correct=int(cm.trace()),total=len(y),macro_accuracy=float((cm.diag()/cm.sum(1).clamp_min(1)).mean()),balanced_nll=float((-z[range(len(y)),y]*cw).sum()),confusion=cm.tolist())
def weights(rows,y,ix):
 keys=[(int(y[i]),rows[i]['identifier'] if rows[i]['room']=='unknown' else rows[i]['room']) for i in ix];counts={k:keys.count(k) for k in set(keys)};domains={c:sum(k[0]==c for k in counts) for c in range(5)};w=torch.tensor([1/(counts[k]*domains[k[0]]) for k in keys]);return w/w.sum()
def train(name,variant,fit,valid,t,s,y,target,rows,keep_all=False):
 folder=WORK/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return torch.load(folder/'predictions.pt',weights_only=False,map_location='cpu')
 fitids=sorted({rows[i]['identifier'] for i in fit});validids=sorted({rows[i]['identifier'] for i in valid});assert not set(fitids)&set(validids)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);m=build(variant).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0004,weight_decay=.01);w=weights(rows,y.cpu(),fit);gen=torch.Generator().manual_seed(42)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);history=[];pred=[]
 for epoch in range(1,6):
  m.train();draw=torch.tensor(fit)[torch.multinomial(w,len(fit),replacement=True,generator=gen)];losses=[]
  for ix in draw.split(32):
   tasks=m.forward_tasks(t[ix],s[ix]);loss,_=task_loss(tasks,y[ix],target[ix],'physical');assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();losses.append(float(loss.detach()))
  m.eval();out=[]
  with torch.no_grad():
   for start in range(0,len(valid),32):
    ix=valid[start:start+32];out.append(m.forward_features(t[ix],s[ix]).cpu())
  z=torch.cat(out);pred.append(z);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=metrics(z,y[valid].cpu())))
  if keep_all:torch.save(dict(model_state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=epoch,fit_identifiers=fitids),folder/f'epoch{epoch}.pt')
  print('TRAIN',name,epoch,history[-1]['validation']['correct'],'/',len(valid),flush=True)
 out=dict(logits=torch.stack(pred),indices=valid,labels=y[valid].cpu(),fit_identifiers=fitids,valid_identifiers=validids,history=history)
 torch.save(out,folder/'predictions.pt');save(folder/'complete.json',dict(variant=variant,fit_records=fitids,valid_records=validids,history=history,parameters=sum(p.numel() for p in m.parameters())))
 del m,opt;gc.collect();torch.cuda.empty_cache();return out
def main():
 torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 WORK.mkdir(parents=True,exist_ok=True);rows=load(CACHE/'rows.json');y=torch.tensor([r['headcount'] for r in rows],device='cuda');t=torch.from_numpy(np.array(np.load(CACHE/'t.npy',mmap_mode='r')[:,0])).cuda();s=torch.from_numpy(np.array(np.load(CACHE/'s.npy',mmap_mode='r')[:,0])).cuda();target=signal_targets(t)
 manifests=load(BASE/'detail_features_20261027/probe_results.json')['folds'];splits={};results=[]
 for fold in range(3):
  testids=set(manifests[f'record{fold}']['test_records']);test=[i for i,r in enumerate(rows) if r['identifier'] in testids];source=[i for i in range(len(rows)) if i not in set(test)];assignment={}
  for c in range(5):
   ids=sorted({rows[i]['identifier'] for i in source if int(y[i])==c},key=lambda v:hashlib.sha256(('nested_context:'+v).encode()).hexdigest());assert len(ids)>=2
   for j,ident in enumerate(ids):assignment[ident]=j%2
  inner=[([i for i in source if assignment[rows[i]['identifier']]!=k],[i for i in source if assignment[rows[i]['identifier']]==k]) for k in range(2)];splits[f'record{fold}']=dict(source=source,test=test,inner=inner,assignment=assignment)
 save(WORK/'protocol.json',dict(architectures=['baseline','context_events'],seed=42,epochs=5,lr=.0004,weight_decay=.01,batch=32,training='original full task loss and intrinsic augmentations; forward view only; class-room balanced sampling with unknown grouped byrecord; samples_per_epoch=fit_size',primary_selection='source OOF macro_accuracy maximum; tie -> lowerbalancedNLL; further tie -> earlier epoch',secondary_selection='source OOF balancedNLL minimum',limits='Existing outer development folds reused; inner teacher excludes each OOF record. Selected epoch uses inner labels. Outer test never selects epoch or bias. Actual054 all-data weights not used.',splits=splits))
 for fold in range(3):
  spec=splits[f'record{fold}'];source=spec['source'];test=spec['test']
  for variant in ('baseline','context_events'):
   inner=[]
   for k,(fit,valid) in enumerate(spec['inner']):inner.append(train(f'record{fold}_{variant}_inner{k}',variant,fit,valid,t,s,y,target,rows))
   indices=[i for d in inner for i in d['indices']];assert sorted(indices)==sorted(source);oof=torch.cat([d['logits'] for d in inner],1);labels=torch.cat([d['labels'] for d in inner]);scores=[metrics(oof[e],labels) for e in range(5)]
   acc=max(range(5),key=lambda e:(scores[e]['macro_accuracy'],-scores[e]['balanced_nll'],-e));nll=min(range(5),key=lambda e:scores[e]['balanced_nll']);save(WORK/f'record{fold}_{variant}_selection.json',dict(accuracy_epoch=acc+1,nll_epoch=nll+1,scores=scores));torch.save(dict(logits=oof,indices=indices,labels=labels),WORK/f'record{fold}_{variant}_oof.pt')
   outer=train(f'record{fold}_{variant}_outer',variant,source,test,t,s,y,target,rows,True)
   for criterion,e in [('accuracy',acc),('nll',nll),('fixed5',4)]:
    r=dict(fold=fold,variant=variant,criterion=criterion,epoch=e+1,metrics=metrics(outer['logits'][e],outer['labels']));results.append(r);print('RESULT',r,flush=True)
   save(WORK/'results.json',dict(results=results))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
