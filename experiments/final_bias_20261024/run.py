"""Exact048 cached logits: bounded final-output bias calibration, no rerouting."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
WORK=Path('D:/huawei2026-experiments/final_bias_20261024')
OLD=Path('D:/huawei2026-experiments/presence_rescue_20261024')
PREVIOUS=Path('D:/huawei2026-experiments/protected_context_20261024')
CACHE=Path('D:/huawei2026-experiments/review_20261020/archive_from_C/runs/architecture_search/features.pt')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def metric(z,y,base,rows,indices):
    ix=torch.tensor(indices);pred=z[ix].argmax(-1);b=base[ix];truth=y[ix]
    cm=torch.bincount(truth*5+pred,minlength=25).reshape(5,5)
    rr=[]
    for identifier in sorted({rows[i]['identifier'] for i in indices}):
        j=torch.tensor([j for j,i in enumerate(indices) if rows[i]['identifier']==identifier])
        rr.append(dict(identifier=identifier,truth=int(truth[j[0]]),windows=len(j),
            corrected=int(((pred[j]==truth[j])&(b[j]!=truth[j])).sum()),
            broken=int(((pred[j]!=truth[j])&(b[j]==truth[j])).sum())))
    return dict(correct=int((pred==truth).sum()),total=len(ix),accuracy=float((pred==truth).float().mean()),
        corrected=int(((pred==truth)&(b!=truth)).sum()),broken=int(((pred!=truth)&(b==truth)).sum()),
        confusion_matrix=cm.tolist(),records=rr,
        changed_from=torch.bincount(b[pred!=b],minlength=5).tolist(),changed_to=torch.bincount(pred[pred!=b],minlength=5).tolist())


def fit(z,y,rows,indices,method):
    ix=torch.tensor(indices);groups={}
    for i in indices:
        key=int(y[i]),rows[i]['room'];groups[key]=groups.get(key,0)+1
    domains={label:len({room for label0,room in groups if label0==label}) for label in range(5)}
    assert min(domains.values())>0
    w=torch.tensor([1/(groups[int(y[i]),rows[i]['room']]*domains[int(y[i])]) for i in indices]);w/=w.sum()
    delta=torch.zeros(5,requires_grad=True)
    for step in range(500):
        loss=(F.cross_entropy(z[ix]+delta,y[ix],reduction='none')*w).sum()+.2*delta.square().sum()
        gradient,=torch.autograd.grad(loss,delta)
        with torch.no_grad():
            delta-=.1*gradient
            if method=='one_positive':
                delta[[0,2,3,4]]=0;delta[1].clamp_(0,.35)
            else:delta.clamp_(-.35,.35)
    return delta.detach()


def proxy_logits(e,room):
    raw=e['logits'];cal=read(PREVIOUS/'results'/f's42_{room}'/'metrics.json')['calibration']['baseline']
    b=torch.tensor(cal['bias']);temperature=cal['temperature']
    original=raw[:,0]/temperature+b
    q=raw.exp().reshape(-1,2,2,5).mean(2)
    identity=q[:,0].clamp_min(1e-12).log()/temperature+b
    average=q.mean(1).clamp_min(1e-12).log()/temperature+b
    label=identity.argmax(-1)
    legacy=torch.where(((label==0)|(label==2))[:,None],identity,average)
    return torch.where((original.argmax(-1)==1)[:,None],original,legacy)


def main():
    torch.set_num_threads(4)
    d=torch.load(CACHE,map_location='cpu',weights_only=True);rows,y=d['rows'],d['y']
    with Path('D:/huawei2026-experiments/actual047_audit_20261024/all_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        raw={(r['identifier'],int(r['start'])):r for r in csv.DictReader(f)}
    a=torch.tensor([[float(raw[r['identifier'],r['start']][f'base044_logit_{k}']) for k in range(5)] for r in rows])
    l=torch.tensor([[float(raw[r['identifier'],r['start']][f'light047_logit_{k}']) for k in range(5)] for r in rows])
    z=torch.where((a.argmax(-1)==1)[:,None],a,l);base=z.argmax(-1)
    assert int((base==y).sum())==654
    e=torch.load(OLD/'actual_evidence.pt',weights_only=True);b=e['bias']
    fixed={'control':torch.zeros(5),'one_fixed035':torch.tensor([0.,.35,0,0,0]),
        'one_fixed050_diagnostic':torch.tensor([0.,.5,0,0,0]),'bias_shrink10':-.1*(b-b.mean()),
        'four_plus020_diagnostic':torch.tensor([0.,0,0,0,.2])}
    save(WORK/'plan.json',dict(base='original scored048, exact cached actualONNX outputs; no high-count045 head',
        methods=list(fixed)+['one_positive','all_regularized'],steps=500,learning_rate=.1,
        bounded_fit='single1 residual in[0,.35], or all5 residuals in[-.35,.35], lambda0.2 L2; class/room weights',
        folds=['a1','c1','g1','e1'],fit='existing training-side whole calibration records only; outer not used for fitting',
        caveats=['Actual048 encoder trained on all48; actual-head calibration folds are diagnostic, not independent pipeline.',
                 'Proxy backbone previously fit only training-side records; same inner calibration records reused for its earlier calibration.',
                 'The two fixed1 shifts were inspected before this run; these are development choices, not untouched preregistered thresholds.',
                 'Unknown hidden priors; no claim target uniform or pure label shift. No U-data assumption or hidden-label fitting.'],
        selection='Prefer one-class shift if stable fitted direction; do not export source-optimal all5 shift merely for local gains.'))
    all_ix=list(range(len(y)));results=[];fitted=[]
    for room in ('a1','c1','g1','e1'):
        split=read(OLD/'splits'/f'{room}.json');groups={k:{r['identifier'] for r in v} for k,v in split.items()}
        assert groups['calibration'].isdisjoint(groups['test'])
        calibration=[i for i,r in enumerate(rows) if r['identifier'] in groups['calibration']]
        test=[i for i,r in enumerate(rows) if r['identifier'] in groups['test']]
        px=proxy_logits(torch.load(OLD/f'proxy_evidence_{room}.pt',weights_only=True),room)
        for origin,zz in [('actual_diagnostic',z),('independent_proxy',px)]:
            for method in ('one_positive','all_regularized','one_fixed035'):
                delta=fixed[method] if method in fixed else fit(zz,y,rows,calibration,method)
                value=dict(room=room,origin=origin,method=method,delta=delta.tolist(),
                    calibration_records=sorted(groups['calibration']),test_records=sorted(groups['test']),
                    metrics=metric(zz+delta,y,zz.argmax(-1),rows,test))
                results.append(value)
                if origin=='actual_diagnostic' and method=='one_positive':fitted.append(float(delta[1]))
                print('CALIBRATION',room,origin,method,delta.tolist(),value['metrics']['corrected'],value['metrics']['broken'],flush=True)
    save(WORK/'fold_results.json',results)
    known=[i for i,r in enumerate(rows) if r['room']!='unknown']
    full={method:fit(z,y,rows,known,method) for method in ('one_positive','all_regularized')}
    # Median over separately fitted whole-record calibration splits, not outer labels.
    chosen=torch.tensor([0.,float(np.median(fitted)),0,0,0])
    fixed['one_calibration_median']=chosen
    full_report={method:dict(delta=delta.tolist(),metrics=metric(z+delta,y,base,rows,all_ix)) for method,delta in {**fixed,**full}.items()}
    summary=dict(platform_best=.48,high_count_platform=.45,base_correct=654,total=960,
        candidates=full_report,chosen_method='one_calibration_median',chosen_delta=chosen.tolist(),
        note='Median of overlapping training-side calibration subsets, not four independent crossfits. No outer-room labels used in each fit; source encoder label exposure remains. Proxy risk separate.')
    save(WORK/'summary.json',summary)
    np.savez_compressed(WORK/'cached_exact048.npz',logits=z.numpy(),labels=y.numpy(),delta=chosen.numpy())
    save(WORK/'rows.json',rows)
    score=full_report['one_calibration_median']['metrics']
    print('CHOSEN',chosen.tolist(),{k:score[k] for k in ('correct','corrected','broken')},flush=True)


if __name__=='__main__':main()
