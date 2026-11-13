"""Next round uses the actual054ONNX preprocessing, not historical native caches."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
WORK=Path('D:/huawei2026-experiments/next053_20261025')
read,save,sha=old.read,old.save,old.sha
SOURCE=old.SOURCE;SOURCE_SHA=old.SOURCE_SHA
PERMS=old.PERMS
def make_session(path):
    import onnxruntime as ort
    o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])
def opinions(raw,b,T=1.):
    baseline=old.policy(raw,b,T)
    q=raw.exp().mean(2).clamp_min(1e-12).log()/T+b
    gnp=old.finish(raw,b,T,q[:,1:5].exp().mean(1).clamp_min(1e-12).log())
    # Actual teammate's six U-calibration vectors, provenance retained; not newly fit.
    bc=read(WORK/'gcal_bias.json');bias=q.new_tensor([bc[p] for p in ('012','021','102','120','201','210')])
    gcal=old.finish(raw,b,T,(q+bias).exp().mean(1).clamp_min(1e-12).log())
    return baseline,gnp,gcal

def transfer(base,student,kind):
    p=base.softmax(-1);s=student.softmax(-1)
    if kind=='all_mean':return ((p+s)*.5).clamp_min(1e-12).log()
    if kind=='low_guard':return torch.where((base.argmax(-1)<2)[:,None],student,base)
    if kind=='binary01':
        # Preserve total teacher01mass and teacherclasses2/3/4; update only01ratio.
        conditional=s[:,:2]/s[:,:2].sum(-1,keepdim=True).clamp_min(1e-12)
        low=p[:,:2].sum(-1,keepdim=True)*(.5*conditional+.5*p[:,:2]/p[:,:2].sum(-1,keepdim=True).clamp_min(1e-12))
        proposed=torch.cat([low,p[:,2:]],-1).clamp_min(1e-12).log()
        return torch.where((base.argmax(-1)<2)[:,None],proposed,base)
    if kind=='presence':
        present=.5*(1-p[:,:1])+.5*(1-s[:,:1]);conditional=p[:,1:]/p[:,1:].sum(-1,keepdim=True).clamp_min(1e-12)
        proposed=torch.cat([1-present,present*conditional],-1).clamp_min(1e-12).log()
        return torch.where((base.argmax(-1)<2)[:,None],proposed,base)
    raise ValueError(kind)

def pool(raw,b,T,method):
    base,gnp,gcal=opinions(raw,b,T);p=torch.stack([base,gnp,gcal]).softmax(-1)
    if method=='baseline':return base
    if method=='gnp_mean':return (.5*p[0]+.5*p[1]).clamp_min(1e-12).log()
    if method in ('rx_ordinal25','rx_ordinal50'):
        _,q,rx,_=old.components(raw,b,T)
        cumulative=rx.flip(-1).cumsum(-1).flip(-1)[:,:,1:].clamp(1e-5,1-1e-5)
        c=torch.logit(cumulative).mean(1).sigmoid()
        ordinal=torch.cat([1-c[:,:1],c[:,:-1]-c[:,1:],c[:,-1:]],-1).clamp_min(1e-12)
        categorical=q.mean(1);categorical=categorical/categorical.sum(-1,keepdim=True)
        amount=.25 if method=='rx_ordinal25' else .5
        return old.finish(raw,b,T,((1-amount)*categorical+amount*ordinal).clamp_min(1e-12).log())
    if method=='opinion_mean':return (.5*p[0]+.25*p[1]+.25*p[2]).clamp_min(1e-12).log()
    if method=='opinion_median':
        v=p.median(0).values;return (v/v.sum(-1,keepdim=True)).clamp_min(1e-12).log()
    if method=='ordinal_logit':
        # Pool four nested exceedance probabilities, preserving count ordering.
        cumulative=p.flip(-1).cumsum(-1).flip(-1)[:,:,1:].clamp(1e-5,1-1e-5)
        weights=p.new_tensor([.5,.25,.25])[:,None,None]
        c=(torch.logit(cumulative)*weights).sum(0).sigmoid()
        v=torch.cat([1-c[:,:1],c[:,:-1]-c[:,1:],c[:,-1:]],-1)
        return v.clamp_min(1e-12).log()
    raise ValueError(method)

def mirror_features(t):
    a=t[:,:768].reshape(-1,6,128,150).flip(2).reshape(-1,768,150)
    p=t[:,768:960].reshape(-1,6,32,150).flip(2).reshape(-1,192,150)
    return torch.cat([a,p,t[:,960:]],1)

def spectral(raw,mirrored,b,T,amount):
    q=raw.exp().mean(2).clamp_min(1e-12).log()/T+b
    qm=mirrored.exp().mean(2).clamp_min(1e-12).log()/T+b
    return old.finish(raw,b,T,((1-amount)*q.exp().mean(1)+amount*qm.exp().mean(1)).clamp_min(1e-12).log())
