"""Cross-fitted reliability arbitration; original054 remains the default action."""
import gc,hashlib,importlib.util,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/next053_20261025'))
from n53_common import old,make_session
from common import ReceiverViews
WORK=Path('D:/huawei2026-experiments/protection_arbiter_20261025')
PREV=Path('D:/huawei2026-experiments/next053_20261025')
EVID=Path('D:/huawei2026-experiments/subnet_expectation_20261025/evidence')
save,read,sha=old.save,old.read,old.sha
ROOMS=('a1','c1','g1','e1')

def opinions(raw,b,T):
    base=old.policy(raw,b,T)
    offset=raw.new_tensor([0,.35,0,0,0])
    singles=raw/T+b+offset
    pairs=raw.exp().mean(2).clamp_min(1e-12).log()/T+b+offset
    group=(pairs-offset).exp().mean(1).clamp_min(1e-12).log()+offset
    # Baseline, full group, 6 time-paired Rx opinions, 12 single views.
    return torch.cat([base[:,None],group[:,None],pairs,singles.flatten(1,2)],1).softmax(-1)

def features(prob):
    winner=prob.argmax(-1);top=prob.topk(2,dim=-1).values
    base=prob[:,0];full=prob[:,1];view=prob[:,2:8]
    gather=lambda x:x.gather(-1,winner).clamp_min(1e-8)
    bp=gather(base);fp=gather(full)
    votes=(view.argmax(-1)[:,None,:]==winner[:,:,None]).float().mean(-1)
    # Explicit relative uncertainty; no input room ID or latent room descriptor.
    entropy=-(prob*prob.clamp_min(1e-8).log()).sum(-1)/np.log(5)
    jsbase=(prob*(prob.clamp_min(1e-8).log()-base[:,None].clamp_min(1e-8).log())).sum(-1)
    jsfull=(prob*(prob.clamp_min(1e-8).log()-full[:,None].clamp_min(1e-8).log())).sum(-1)
    values=[top[:,:,0],top[:,:,0]-top[:,:,1],entropy,bp,fp,votes,
            jsbase.clamp_max(10)/10,jsfull.clamp_max(10)/10,
            (winner==winner[:,:1]).float(),(winner==winner[:,1:2]).float()]
    return torch.stack(values,-1)

def main():
    summary=[]
    for room in ROOMS:
        d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room);p=opinions(d['raw'],b,T)
        ix=old.split(room,d['rows'])[2];y=d['labels'][ix];z=p[ix].argmax(-1)
        base=z[:,0]==y;anyright=(z==y[:,None]).any(1)
        summary.append(dict(room=room,windows=len(ix),baseline=int(base.sum()),
                            additional_oracle=int((~base&anyright).sum()),
                            full_group_rescue=int((~base&(z[:,1]==y)).sum())))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
