"""Conditional high-count readout vs fixed temporal-consensus lowering."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/high_count_20261024')
OLD=Path('D:/huawei2026-experiments/presence_rescue_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
sys.path.insert(0,str(ROOT/'experiments/presence_rescue_20261024'))
from run import proxy_prediction


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def evidence(e):
    x=e['features'][:,:172].reshape(-1,4,43);a=[];b=[]
    for view in range(4):
        for offset in (0,5,18):
            p=x[:,view,offset:offset+5].clamp_min(1e-6)
            a.append(((p[:,3]+p[:,4]).log()-p[:,2].log()).clamp(-6,6))
            b.append((p[:,4].log()-p[:,3].log()).clamp(-6,6))
    return torch.stack([torch.stack(a,1),torch.stack(b,1)],1)


class Boundaries(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed)
        self.raw=nn.Parameter(torch.randn(2,12)*.02);self.bias=nn.Parameter(torch.zeros(2))
    def forward(self,x):return (x*self.raw.softmax(-1)[None]).sum(-1)+2*self.bias.tanh()


def probs(z):
    a,b=z.sigmoid().unbind(-1)
    return torch.stack([1-a,a*(1-b),a*b],1)


def metric(pred,base,y,rows,ix):
    ix=torch.tensor(ix);p,b,y=pred[ix],base[ix],y[ix]
    cm=torch.bincount(y*5+p,minlength=25).reshape(5,5)
    records=[]
    for identifier in sorted({rows[i]['identifier'] for i in ix.tolist()}):
        j=torch.tensor([j for j,i in enumerate(ix.tolist()) if rows[i]['identifier']==identifier])
        records.append(dict(identifier=identifier,truth=int(y[j[0]]),windows=len(j),
            corrected=int(((p[j]==y[j])&(b[j]!=y[j])).sum()),broken=int(((p[j]!=y[j])&(b[j]==y[j])).sum())))
    return dict(corrected=int(((p==y)&(b!=y)).sum()),broken=int(((p!=y)&(b==y)).sum()),
        correct=int((p==y).sum()),total=len(ix),confusion_matrix=cm.tolist(),records=records)


def fit(x,y,rows,indices,seed,pair,folder):
    ix=[i for i in indices if y[i]>=2];ids=sorted({rows[i]['identifier'] for i in ix})
    assert set(y[ix].tolist())=={2,3,4}
    m=Boundaries(seed);optimizer=torch.optim.Adam(m.parameters(),lr=.02)
    groups={}
    for i in ix:
        key=int(y[i]),rows[i]['room'];groups[key]=groups.get(key,0)+1
    domains={label:len({r for label0,r in groups if label0==label}) for label in (2,3,4)}
    w=torch.tensor([1/(groups[int(y[i]),rows[i]['room']]*domains[int(y[i])]) for i in ix]);w/=w.sum()
    rx=torch.stack([x[[i for i in ix if rows[i]['identifier']==identifier]].mean(0) for identifier in ids])
    rr=[rows[next(i for i in ix if rows[i]['identifier']==identifier)] for identifier in ids]
    pairsets=[[(a,b) for a,ra in enumerate(rr) for b,rb in enumerate(rr) if ra['room']==rb['room'] and
               ra['headcount']>boundary and rb['headcount']==boundary] for boundary in (2,3)]
    labels=torch.stack([(y[ix]>2).float(),(y[ix]>3).float()],1)
    history=[]
    for step in range(600):
        z=m(x[ix]);loss0=(F.binary_cross_entropy_with_logits(z[:,0],labels[:,0],reduction='none')*w).sum()
        mask=y[ix]>2;loss1=(F.binary_cross_entropy_with_logits(z[mask,1],labels[mask,1],reduction='none')*w[mask]).sum()/w[mask].sum()
        ranking=torch.zeros(())
        if pair:
            rz=m(rx);terms=[F.softplus(1-rz[a,k]+rz[b,k]) for k,pairs in enumerate(pairsets) for a,b in pairs]
            if terms:ranking=torch.stack(terms).mean()
        loss=loss0+loss1+.5*ranking+.02*(2*m.bias.tanh()).square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%100==0:history.append(float(loss.detach()))
    with torch.inference_mode():p=probs(m(x))
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=m.state_dict(),weights=m.raw.detach().softmax(-1),bias=2*m.bias.detach().tanh(),
        seed=seed,pair_loss=pair,fit_identifiers=ids),folder/'final.pt')
    save(folder/'training.json',dict(steps=600,loss=history,parameters=26,record_pairs=[len(v) for v in pairsets],confidence_gate=.6))
    return p


def main():
    torch.set_num_threads(4)
    d=torch.load(CACHE,map_location='cpu',weights_only=True);rows,y=d['rows'],d['y']
    e=torch.load(OLD/'actual_evidence.pt',weights_only=True);x=evidence(e)
    joined=read(Path('D:/huawei2026-experiments/remaining048_20261024/joined_windows.json'))
    lookup={(r['identifier'],r['start']):r for r in joined}
    base=torch.tensor([lookup[r['identifier'],r['start']]['prediction'] for r in rows]);assert int((base==y).sum())==654
    save(WORK/'plan.json',dict(base='original scored048, not temporal80',rooms=['a1','c1'],seeds=[42,2026],steps=600,
        models='positive 26-parameter conditional 2/3/4 readout; with/without within-room record ranking',
        gate='only base>=2 may be lowered, confidence>=0.6; no changes to0/1',
        limitation='High counts in only two labelled rooms; each outer fold leaves one high-count training room. Actual encoder saw labels; separate independent-proxy analysis. Repeated development data.',
        fixed_candidate='original and time reverse pre-bias argmax agree in2/3 and below base -> original pre-bias; else exact048'))
    raw=e['logits'].argmax(-1);k=raw[:,0]
    mask=(k==raw[:,1])&(k>=2)&(k<=3)&(base>k)
    fixed=torch.where(mask,k,base)
    save(WORK/'fixed_guard_diagnostic.json',metric(fixed,base,y,rows,list(range(len(y)))))
    results=[]
    for room in ('a1','c1'):
        split=read(OLD/'splits'/f'{room}.json');groups={k:{r['identifier'] for r in v} for k,v in split.items()}
        assert groups['fit'].isdisjoint(groups['test']) and groups['fit'].isdisjoint(groups['calibration'])
        fit_ix=[i for i,r in enumerate(rows) if r['identifier'] in groups['fit']]
        test_ix=[i for i,r in enumerate(rows) if r['identifier'] in groups['test']]
        proxy=torch.load(OLD/f'proxy_evidence_{room}.pt',weights_only=True);px=evidence(proxy);pb=proxy_prediction(proxy['logits'],room)
        for origin,xx,bb in [('actual_diagnostic',x,base),('independent_proxy',px,pb)]:
            for pair in (False,True):
                for seed in (42,2026):
                    folder=WORK/'fits'/f'{origin}_{room}_pair{int(pair)}_s{seed}'
                    probability=fit(xx,y,rows,fit_ix,seed,pair,folder)
                    confidence,winner=probability.max(1);winner+=2
                    pred=torch.where((bb>=2)&(winner<bb)&(confidence>=.6),winner,bb)
                    result=dict(origin=origin,room=room,pair_loss=pair,seed=seed,metrics=metric(pred,bb,y,rows,test_ix))
                    save(folder/'result.json',result);results.append(result)
                    np.savez_compressed(folder/'predictions.npz',probability=probability.numpy(),base=bb.numpy(),labels=y.numpy())
                    print('FOLD',origin,room,pair,seed,result['metrics']['corrected'],result['metrics']['broken'],flush=True)
    save(WORK/'results.json',results)
    for pair in (False,True):
        folder=WORK/'full_data'/f'pair{int(pair)}';p=fit(x,y,rows,list(range(len(y))),42,pair,folder)
        confidence,winner=p.max(1);winner+=2
        pred=torch.where((base>=2)&(winner<base)&(confidence>=.6),winner,base)
        save(folder/'diagnostic.json',dict(in_sample=True,metrics=metric(pred,base,y,rows,list(range(len(y))))))
    print('HIGH_COUNT_TRAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
