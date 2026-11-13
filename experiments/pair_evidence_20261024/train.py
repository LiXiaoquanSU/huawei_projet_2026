"""Small positive evidence readout, same-room ranking ablation, fixed .8 gate."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/pair_evidence_20261024')
OLD=Path('D:/huawei2026-experiments/presence_rescue_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')
sys.path.insert(0,str(ROOT/'experiments/presence_rescue_20261024'))
from run import metrics,proxy_prediction


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def evidence_features(evidence):
    x=evidence['features']; chunks=[]
    for view in range(4):
        v=x[:,43*view:43*(view+1)]
        for offset in (0,5,18): # direct, ordinal, static expert
            chunks.append((v[:,offset+1].clamp_min(1e-6).log()-v[:,offset].clamp_min(1e-6).log()).clamp(-6,6))
        p=v[:,10].clamp(1e-6,1-1e-6)
        chunks.append((p.log()-(1-p).log()).clamp(-6,6))
    return torch.stack(chunks,1)


class Monotone(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed)
        self.raw=nn.Parameter(torch.randn(16)*.02);self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,x):return (x*self.raw.softmax(0)).sum(1)+2*self.bias.tanh()


def fit(x,y,rows,indices,seed,pair_loss,folder):
    ix=[i for i in indices if y[i]<=1]; identifiers=sorted({rows[i]['identifier'] for i in ix})
    model=Monotone(seed)
    groups={}
    for i in ix:
        key=int(y[i]),rows[i]['room'];groups[key]=groups.get(key,0)+1
    domains={label:len({room for label0,room in groups if label0==label}) for label in (0,1)}
    weights=torch.tensor([1/(groups[(int(y[i]),rows[i]['room'])]*domains[int(y[i])]) for i in ix]);weights/=weights.sum()
    record_features=torch.stack([x[[i for i in ix if rows[i]['identifier']==identifier]].mean(0) for identifier in identifiers])
    record_rows=[rows[next(i for i in ix if rows[i]['identifier']==identifier)] for identifier in identifiers]
    pairs=[(a,b) for a,ra in enumerate(record_rows) for b,rb in enumerate(record_rows)
           if ra['room']==rb['room'] and ra['room']!='unknown' and ra['headcount']==1 and rb['headcount']==0]
    optimizer=torch.optim.Adam(model.parameters(),lr=.025);history=[]
    for step in range(600):
        prediction=model(x[ix]);classification=(F.binary_cross_entropy_with_logits(prediction,y[ix].float(),reduction='none')*weights).sum()
        ranking=torch.zeros(())
        if pairs:
            z=model(record_features)
            ranking=torch.stack([F.softplus(1-z[a]+z[b]) for a,b in pairs]).mean()
        w=model.raw.softmax(0)
        loss=classification+(.5*ranking if pair_loss else 0)+.01*(2*model.bias.tanh()).square()+.01*(w*(w*16).log()).sum()
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%100==0:history.append(dict(step=step,loss=float(loss.detach()),classification=float(classification.detach()),ranking=float(ranking.detach())))
    with torch.inference_mode():prob=model(x).sigmoid()
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(weights=model.raw.detach().softmax(0),bias=2*model.bias.detach().tanh(),
                    state=model.state_dict(),seed=seed,pair_loss=pair_loss,fit_identifiers=identifiers),folder/'final.pt')
    save(folder/'training.json',dict(history=history,steps=600,parameters=17,
        same_room_record_pairs=len(pairs),pair_loss_weight=.5 if pair_loss else 0,threshold=.8,
        inputs='16 bounded log-odds: 4 views x (direct01,ordinal01,static_expert01,presence); no raw room summaries'))
    return prob


def main():
    torch.set_num_threads(4)
    data=torch.load(CACHE,map_location='cpu',weights_only=True);rows,y=data['rows'],data['y']
    actual_e=torch.load(OLD/'actual_evidence.pt',weights_only=True);actual_x=evidence_features(actual_e)
    joined=read(Path('D:/huawei2026-experiments/remaining048_20261024/joined_windows.json'))
    lookup={(r['identifier'],r['start']):r for r in joined}
    base=torch.tensor([lookup[r['identifier'],r['start']]['prediction'] for r in rows])
    assert int((base==y).sum())==654
    save(WORK/'plan.json',dict(seeds=[42,2026],rooms=['a1','c1','g1','e1'],
        readouts=['positive_linear','positive_pair_ranking'],steps=600,fixed_threshold=.8,
        purpose='Explore complementary errors; no longer demand the prior1% empty loss gate as proof of platform inferiority',
        limitations='Actual encoder saw all records; proxy diagnostics independent only with their own proxy evidence and base. Rooms repeatedly used for development.',
        policy='only base0 may become1; preserve all other base logits on export',
        guard_candidates=['two temporal views unbiased argmax1','four views unbiased argmax1',
                          'two temporal views unbiased argmax1 AND each odds1vs0>=4 (fixed80%)']))
    all_ix=list(range(len(y)));guard_results={}
    logits=actual_e['logits']
    masks={'temporal_consensus':(logits[:,:2].argmax(-1)==1).all(1),
        'four_consensus':(logits.argmax(-1)==1).all(1),
        'temporal_consensus80':(logits[:,:2].argmax(-1)==1).all(1)&((logits[:,:2,1]-logits[:,:2,0])>=np.log(4)).all(1)}
    for name,mask in masks.items():guard_results[name]=metrics(mask.float(),.5,y,base,all_ix,rows)
    save(WORK/'guard_diagnostics.json',guard_results)
    results=[]
    for room in ('a1','c1','g1','e1'):
        split=read(OLD/'splits'/f'{room}.json');groups={k:{r['identifier'] for r in v} for k,v in split.items()}
        fit_ix=[i for i,r in enumerate(rows) if r['identifier'] in groups['fit']]
        test_ix=[i for i,r in enumerate(rows) if r['identifier'] in groups['test']]
        assert groups['fit'].isdisjoint(groups['test']) and groups['fit'].isdisjoint(groups['calibration'])
        proxy=torch.load(OLD/f'proxy_evidence_{room}.pt',weights_only=True)
        proxy_x=evidence_features(proxy);proxy_base=proxy_prediction(proxy['logits'],room)
        for origin,x,reference in [('actual_diagnostic',actual_x,base),('independent_proxy',proxy_x,proxy_base)]:
            for pair in (False,True):
                for seed in (42,2026):
                    name=f'{origin}_{room}_pair{int(pair)}_s{seed}'
                    p=fit(x,y,rows,fit_ix,seed,pair,WORK/'fits'/name)
                    result=dict(origin=origin,room=room,pair_loss=pair,seed=seed,
                        metrics=metrics(p,.8,y,reference,test_ix,rows))
                    results.append(result);save(WORK/'fits'/name/'result.json',result)
                    np.savez_compressed(WORK/'fits'/name/'predictions.npz',probability=p.numpy(),base=reference.numpy(),labels=y.numpy())
                    print('FOLD',name,'corrected/broken',result['metrics']['corrected'],result['metrics']['broken'],flush=True)
    save(WORK/'results.json',results)
    for pair in (False,True):
        folder=WORK/'full_data'/f'pair{int(pair)}'
        p=fit(actual_x,y,rows,all_ix,42,pair,folder)
        save(folder/'diagnostic.json',dict(in_sample=True,metrics=metrics(p,.8,y,base,all_ix,rows)))
    print('TRAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
