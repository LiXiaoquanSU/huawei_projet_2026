"""Frozen original network, exact receiver feature permutations and fixed policies."""
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/tta_policy_20261024')
OLD=Path('D:/huawei2026-experiments')
sys.path.insert(0,str(ROOT/'src'))
from model import StructuredModel

PERMS=list(itertools.permutations(range(3)))
POLICIES=('light','full','unprotected','trimmed','geometric','cyclic','consensus5')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def precision_spec():
    model=onnx.load(ROOT/'models/best_044/model.onnx')
    arrays={x.name:numpy_helper.to_array(x) for x in model.graph.initializer}
    casts={n.output[0]:n.input[0] for n in model.graph.node if n.op_type=='Cast' and n.input[0] in arrays}
    keys=[k.removeprefix('model.') for k,v in casts.items() if arrays[v].dtype==np.float16]
    bias=torch.tensor(arrays['__roomsense_bias_0_constant'].copy())
    return keys,bias,arrays,casts


def build(checkpoint,full=False):
    model=StructuredModel('spectral','physical').eval()
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(cp['model_state'])
    keys,bias,arrays,casts=precision_spec();state=model.state_dict()
    with torch.no_grad():
        for key in keys:state[key].copy_(state[key].half().float())
    if full:
        for key,value in state.items():
            name='model.'+key;source=name if name in arrays else casts.get(name)
            if source is not None:assert np.array_equal(value.numpy(),arrays[source].astype(np.float32))
            else:assert key.startswith('motion.')  # auxiliary branch not used at inference
    return model,bias


def permutation_indices(permutation):
    # [static3,dynamic3] x128 amplitude, three phase pairs x(cos,sin)x32, clock3.
    amplitude=[128*c+i for c in (*permutation,*(x+3 for x in permutation)) for i in range(128)]
    pair_lookup={(0,1):0,(0,2):1,(1,2):2};phase=[];sign=[]
    for u,v in ((0,1),(0,2),(1,2)):
        a,b=permutation[u],permutation[v];pair=pair_lookup[tuple(sorted((a,b)))]
        phase.extend([768+64*pair+i for i in range(64)])
        sign.extend([1.]*32+([1.]*32 if a<b else [-1.]*32))
    index=torch.tensor(amplitude+phase+[960,961,962],dtype=torch.long)
    signs=torch.tensor([1.]*768+sign+[1.]*3).reshape(1,963,1)
    summary=torch.tensor([3*group+c for group in range(4) for c in permutation]+list(range(12,17)),dtype=torch.long)
    return index,signs,summary


class ReceiverViews(nn.Module):
    def __init__(self):
        super().__init__()
        for k,permutation in enumerate(PERMS):
            idx,sign,summary=permutation_indices(permutation)
            self.register_buffer(f'index{k}',idx);self.register_buffer(f'sign{k}',sign);self.register_buffer(f'summary{k}',summary)

    def forward(self,t,s):
        ts=[];ss=[]
        for k in range(6):
            ts.append(t.index_select(1,getattr(self,f'index{k}'))*getattr(self,f'sign{k}'))
            ss.append(s.index_select(1,getattr(self,f'summary{k}')))
        return torch.cat(ts,0),torch.cat(ss,0)


def scores(raw,bias):
    """raw [N,6,2,5] -> time-averaged biased positive scores [N,6,5]."""
    return raw.exp().mean(-2)*bias.exp()


def decide(q,policy):
    """One fixed rule, q [N,6,5]. No learned constants or threshold search."""
    identity=q[:,0];pair=(q[:,0]+q[:,5])*.5
    a=identity.argmax(-1);light=torch.where(((a==0)|(a==2))[:,None],identity,pair)
    a=light.argmax(-1);protect=(a==0)|(a==2)
    mean=q.mean(1)
    if policy=='light':return light
    if policy=='full':alternative=mean
    elif policy=='unprotected':return mean
    elif policy=='trimmed':alternative=q.sort(dim=1).values[:,1:5].mean(1)
    elif policy=='geometric':alternative=q.clamp_min(1e-12).log().mean(1).exp()
    elif policy=='cyclic':alternative=q[:,[0,3,4]].mean(1)
    elif policy=='consensus5':
        winner=mean.argmax(-1)
        votes=(q.argmax(-1)==winner[:,None]).sum(-1)
        top=(mean/mean.sum(-1,keepdim=True)).topk(2,dim=-1).values
        rescue=(votes>=5)&((top[:,0]-top[:,1])>=.20)&(winner!=a)
        protect=protect&~rescue;alternative=mean
    else:raise ValueError(policy)
    return torch.where(protect[:,None],light,alternative)


class Submission(nn.Module):
    def __init__(self,model,bias,policy):
        super().__init__();self.model=model;self.views=ReceiverViews();self.policy=policy
        self.register_buffer('bias',bias)
        self.register_buffer('reverse_frames',torch.arange(599,-1,-1))
        self.register_buffer('reverse_intervals',torch.cat([torch.arange(598,-1,-1),torch.tensor([599])]))

    def forward(self,frames,intervals):
        x=torch.stack([frames,frames.index_select(0,self.reverse_frames)])
        dt=torch.stack([intervals,intervals.index_select(0,self.reverse_intervals)])
        t,s=self.model.preprocess(x.double(),dt.double());t,s=self.views(t.float(),s.float())
        if self.policy=='light':
            # Only the identity/reversed-receiver inputs are needed for this rule.
            t=torch.cat([t[:2],t[-2:]],0);s=torch.cat([s[:2],s[-2:]],0)
            raw=self.model.forward_features(t,s).reshape(1,2,2,5)
            q=scores(raw,self.bias);identity=q[:,0];average=q.mean(1);label=identity.argmax(-1)
            return torch.where(((label==0)|(label==2))[:,None],identity,average)[0].clamp_min(1e-12).log()
        raw=self.model.forward_features(t,s).reshape(1,6,2,5)
        return decide(scores(raw,self.bias),self.policy)[0].clamp_min(1e-12).log()
