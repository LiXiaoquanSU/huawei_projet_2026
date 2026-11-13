"""Frozen054 conditional routing and bounded, class-agnostic view reliability."""
import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tta_policy_20261024'))
from common import build,permutation_indices,precision_spec
WORK=Path('D:/huawei2026-experiments/g60_20261025')
OLD=Path('D:/huawei2026-experiments/protected_context_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
PERMS=((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))
SOURCE=ROOT/'models/teammate054_20261025/gprime_full_group_fallback_web054.onnx'
SOURCE_SHA='2b189c950412d0d770d82a0f118ebd95699e18b5b6706f56a8539c59a04b4534'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checkpoint(room):return ROOT/'models/best_037/final.pt' if room=='actual' else OLD/'fits'/f'baseline_s42_{room}'/'final.pt'
def calibration(room):
    if room=='actual':return precision_spec()[1],1.
    c=read(OLD/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline'];return torch.tensor(c['bias']),c['temperature']
def split(room,rows):
    if room=='actual':return list(range(len(rows))),[],list(range(len(rows)))
    m=read(OLD/'results'/f's42_{room}'/'manifest.json');ids={k:{r['identifier'] for r in m[k]} for k in ('fit','calibration','test')}
    assert all(ids[a].isdisjoint(ids[b]) for a,b in (('fit','test'),('fit','calibration'),('calibration','test')))
    return tuple([i for i,r in enumerate(rows) if r['identifier'] in ids[k]] for k in ('fit','calibration','test'))
def components(raw,b,T=1.):
    logq=raw.exp().mean(2).clamp_min(1e-12).log()/T+b
    q=logq.exp();p=q/q.sum(-1,keepdim=True)
    time=(raw/T+b).softmax(-1);tm=time.mean(2)
    js=(time*(time.clamp_min(1e-8).log()-tm[:,:,None].clamp_min(1e-8).log())).sum(-1).mean(2)
    return logq,q,p,js
def finish(raw,b,T,fallback):
    v0=raw[:,0,0]/T+b;v45=raw[:,0].exp().mean(1).clamp_min(1e-12).log()/T+b
    inner=torch.where(((v45.argmax(-1)==0)|(v45.argmax(-1)==2))[:,None],v45,fallback)
    return torch.where((v0.argmax(-1)==1)[:,None],v0,inner)+raw.new_tensor([0,.35,0,0,0])
def features(raw,b,T=1.):
    l,q,p,js=components(raw,b,T);mean=p.mean(1,keepdim=True);lp=p.clamp_min(1e-8).log();lm=mean.clamp_min(1e-8).log()
    top=p.topk(2,dim=-1).values;ent=-(p*lp).sum(-1)/np.log(5)
    time=(raw/T+b).softmax(-1);mass=l.logsumexp(-1);mass=mass-mass.mean(1,keepdim=True)
    # All features unchanged by a shared relabeling of the5 classes; no Rx/room ID.
    return torch.stack([ent,top[:,:,0],top[:,:,0]-top[:,:,1],(p*mean).sum(-1),
      (p*(lp-lm)).sum(-1),js,mass.clamp(-2,2),(time[:,:,0]-time[:,:,1]).abs().sum(-1),p.square().sum(-1)],-1)
class Reliability(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(9,16),nn.SiLU(),nn.Linear(16,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return (.5*self.net(x).squeeze(-1).tanh()).softmax(-1)
def policy(raw,b,T=1.,method='baseline',gates=None):
    l,q,p,js=components(raw,b,T)
    if method in ('consensus_bridge','rescue01','rescue_empty'):
        base=finish(raw,b,T,q.mean(1).clamp_min(1e-12).log());full=q.mean(1);prob=full/full.sum(-1,keepdim=True);winner=prob.argmax(-1)
        v0=(raw[:,0,0]/T+b).argmax(-1);v45=l[:,0].argmax(-1);votes=(p.argmax(-1)==winner[:,None]).sum(1)
        if method=='consensus_bridge':
            top=prob.topk(2,-1).values;mask=(v0!=1)&((v45==0)|(v45==2))&(votes>=5)&((top[:,0]-top[:,1])>=.2)&(winner!=v45)
            alternative=(.75*(base-raw.new_tensor([0,.35,0,0,0])).exp()+.25*full).clamp_min(1e-12).log()+raw.new_tensor([0,.35,0,0,0])
        elif method=='rescue01':
            mask=(v0!=1)&(v45==0)&(winner==1)&(votes>=5)&(prob[:,1]>=.6);alternative=full.clamp_min(1e-12).log()+raw.new_tensor([0,.35,0,0,0])
        else:
            mask=(v0==1)&(winner==0)&(votes==6)&(prob[:,0]>=.8);alternative=full.clamp_min(1e-12).log()+raw.new_tensor([0,.35,0,0,0])
        return torch.where(mask[:,None],alternative,base)
    if method=='baseline':alt=q.mean(1)
    elif method=='normalized':alt=p.mean(1)
    elif method=='power075':alt=q.pow(.75).mean(1).clamp_min(1e-12).pow(1/.75)
    elif method=='trimmed25':alt=.75*q.mean(1)+.25*q.sort(1).values[:,1:5].mean(1)
    elif method=='timegeom25':
        geometric=(raw.mean(2)/T+b).exp();alt=.75*q.mean(1)+.25*geometric.mean(1)
    elif method=='time_js':
        w=(-js/.1).exp().clamp_min(.5);w=w/w.sum(1,keepdim=True);alt=(q*w[:,:,None]).sum(1)
    elif method=='learned':
        x=features(raw,b,T);w=torch.stack([g(x) for g in gates]).mean(0);alt=(q*w[:,:,None]).sum(1)
    else:raise ValueError(method)
    return finish(raw,b,T,alt.clamp_min(1e-12).log())
def metrics(z,y,base,ix):
    ix=torch.tensor(ix);p=z[ix].argmax(-1);bp=base[ix].argmax(-1);yy=y[ix];cm=torch.bincount(yy*5+p,minlength=25).reshape(5,5)
    return dict(correct=int((p==yy).sum()),total=len(ix),accuracy=float((p==yy).float().mean()),corrected=int(((p==yy)&(bp!=yy)).sum()),broken=int(((p!=yy)&(bp==yy)).sum()),confusion_matrix=cm.tolist())
def balance(rows,y,ix):
    groups={}
    for i in ix:
        key=(int(y[i]),rows[i]['room']);groups[key]=groups.get(key,0)+1
    domains={k:sum(c==k for c,r in groups) for k in range(5)}
    w=torch.tensor([1/(groups[int(y[i]),rows[i]['room']]*domains[int(y[i])]) for i in ix]);return w/w.sum()
