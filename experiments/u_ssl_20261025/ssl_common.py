"""Fixed original initialization and trainable phase/time encoder suffixes."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ssl_adapter_shared',ROOT/'experiments/adapter049_20261024/run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
build,permutation_indices,policy=shared.build,shared.permutation_indices,shared.policy
read,save,sha=shared.read,shared.save,shared.sha
WORK=Path('D:/huawei2026-experiments/u_ssl_20261025')
OLD=Path('D:/huawei2026-experiments/protected_context_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
ROUTES=('actual','a1','c1','g1','e1')

def checkpoint(room):return ROOT/'models/best_037/final.pt' if room=='actual' else OLD/'fits'/f'baseline_s42_{room}'/'final.pt'
def old_latents(room):return Path('D:/huawei2026-experiments/evidence049_research_20261024/frozen_latents.pt') if room=='actual' else Path(f'D:/huawei2026-experiments/adapter049_20261024/{room}_latents.pt')

class Student(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.phase_project=copy.deepcopy(model.phase.project)
        self.time_block=copy.deepcopy(model.temporal.blocks[-1])
        self.time_project=copy.deepcopy(model.temporal.project)
        self.decoder=shared.AdapterHead(model).eval();self.decoder.requires_grad_(False)
        self.projector=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,128))
    def encode(self,phase,time):
        p=self.phase_project(phase);x=self.time_block(time)
        mean=x.mean(1);std=(x.sub(mean[:,None]).square().mean(1)+1e-5).sqrt()
        t=self.time_project(torch.cat([mean,std],1))
        return p,t
    def readout(self,p,t,e,s):
        new=torch.stack([e[:,0],e[:,1],p,t],1)
        return self.decoder(new,s,0.)[0]
    def forward(self,phase,time,e,s):
        p,t=self.encode(phase,time);return self.readout(p,t,e,s),torch.cat([p,t],-1)
    def selected_state(self):return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith(('decoder.','projector.'))}

def vicreg(a,b):
    a=a.float();b=b.float();invariance=F.mse_loss(a,b)
    sa=(a.var(0,unbiased=False)+1e-4).sqrt();sb=(b.var(0,unbiased=False)+1e-4).sqrt()
    variance=.5*(F.relu(1-sa).mean()+F.relu(1-sb).mean())
    ac=a-a.mean(0);bc=b-b.mean(0);den=max(a.shape[0]-1,1)
    ca=ac.T@ac/den;cb=bc.T@bc/den
    mask=1-torch.eye(a.shape[1],device=a.device)
    covariance=((ca*mask).square().sum()+(cb*mask).square().sum())/(2*a.shape[1])
    return 25*invariance+25*variance+covariance,dict(invariance=float(invariance.detach()),variance=float(variance.detach()),covariance=float(covariance.detach()))

def scores(model,data,bias,T=1.,batch=64):
    zz=[];representation=[]
    with torch.no_grad():
        for start in range(0,len(data['labels']),batch//4):
            stop=start+batch//4
            args=[data[k][start:stop].cuda().flatten(0,1) for k in ('phase','time','latents','summaries')]
            z,e=model(*args);zz.append(z.cpu().reshape(-1,4,5));representation.append(e.cpu().reshape(-1,4,512))
    raw=torch.cat(zz);return policy(raw,bias,T),raw,torch.cat(representation)

def metrics(z,y,base,indices):
    ix=torch.tensor(indices);p=z[ix].argmax(-1);bp=base[ix].argmax(-1);yy=y[ix]
    cm=torch.bincount(yy*5+p,minlength=25).reshape(5,5)
    return dict(correct=int((p==yy).sum()),total=len(ix),accuracy=float((p==yy).float().mean()),corrected=int(((p==yy)&(bp!=yy)).sum()),broken=int(((p!=yy)&(bp==yy)).sum()),confusion_matrix=cm.tolist())

def balance(rows,y,ix):
    groups={}
    for i in ix:
        room=rows[i]['room'] if rows[i]['room']!='unknown' else rows[i]['identifier']
        key=int(y[i]),room;groups[key]=groups.get(key,0)+1
    domains={label:len({room for c,room in groups if c==label}) for label in range(5)}
    w=[]
    for i in ix:
        room=rows[i]['room'] if rows[i]['room']!='unknown' else rows[i]['identifier']
        w.append(1/(groups[int(y[i]),room]*domains[int(y[i])]))
    w=torch.tensor(w);return w/w.sum()
