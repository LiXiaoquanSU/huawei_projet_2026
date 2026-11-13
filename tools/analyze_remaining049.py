"""Source-data residual diagnosis; no training, export, or hidden-test access."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('D:/huawei2026-experiments/remaining049_20261024')
PRIOR=Path('D:/huawei2026-experiments/final_bias_20261024')
BEST=ROOT/'models/final_bias_20261024/01_final_bias_one035.onnx'
HASH='d083e3bb36f522e29b45acb88e41e5577013f9140b2f59dd2f9741c476f839df'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csv_save(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quantiles(x):return np.quantile(x,[0,.25,.5,.75,1]).tolist() if len(x) else []


def main():
    assert sha(BEST)==HASH
    rows=read(PRIOR/'rows.json')
    with np.load(PRIOR/'cached_exact048.npz') as data:z48=data['logits'].copy();y=data['labels'].copy()
    z=z48+np.array([0,.35,0,0,0],np.float32);prediction=z.argmax(1)
    assert int((prediction==y).sum())==688
    error=prediction!=y;cm=np.bincount(y*5+prediction,minlength=25).reshape(5,5)
    audit=rows_csv(Path('D:/huawei2026-experiments/actual047_audit_20261024/all_predictions.csv'))
    original={(r['identifier'],int(r['start'])):r for r in audit}
    def actual_logits(name):
        return np.array([[float(original[r['identifier'],r['start']][f'{name}_logit_{k}']) for k in range(5)] for r in rows],np.float32)
    alternatives={
        'original044_plus035':actual_logits('base044')+np.array([0,.35,0,0,0],np.float32),
        'full047_plus035':actual_logits('full047')+np.array([0,.35,0,0,0],np.float32)}
    provenance={k:'cached prior actual ONNX audit; full047 includes exact protected-branch reuse' for k in alternatives}
    evidence=torch.load(Path('D:/huawei2026-experiments/presence_rescue_20261024/actual_evidence.pt'),weights_only=True)
    f=evidence['features'].numpy()[:,:172].reshape(-1,4,43)
    raw=evidence['logits'].numpy()
    for k in range(4):
        alternatives[f'raw_unbiased_view{k}']=raw[:,k]
        alternatives[f'direct_view{k}']=f[:,k,:5]
        alternatives[f'ordinal_view{k}']=f[:,k,5:10]
    for k,name in enumerate(('static','dynamic','phase','temporal')):alternatives[f'expert_{name}_original']=f[:,0,18+k*5:23+k*5]
    for name in alternatives:
        provenance.setdefault(name,'frozen exact weights on cached preprocessed features; approximate probabilities, diagnostic only')
    old_specs=[('cross039','cross_events',42,ROOT/'models/best_039/model.onnx'),
               ('context039','context_events',2026,ROOT/'models/best_039_context/model.onnx')]
    registry=read(Path('D:/huawei2026-experiments/global_local_20261021/export_registry.json'))
    old_sources=[]
    for name,architecture,seed,path in old_specs:
        entry=next(v for v in registry if v['architecture']==architecture)
        assert sha(path)==entry['sha256']
        source=Path('D:/huawei2026-experiments/global_local_20261021/fits')/f'{architecture}_s{seed}_full'/'training_predictions.csv'
        old={(r['identifier'],int(r['start'])):r for r in rows_csv(source)}
        assert len(old)==len(rows)==960
        alternatives[name]=np.array([[float(old[r['identifier'],r['start']][f'log_probability_{k}']) for k in range(5)] for r in rows])
        assert all(int(old[r['identifier'],r['start']]['headcount'])==int(y[i]) for i,r in enumerate(rows))
        provenance[name]='historical full-training PyTorch prediction CSV; not new actual ONNX run and not independent validation'
        old_sources.append(dict(name=name,model_sha256=entry['sha256'],prediction_csv=str(source),prediction_csv_sha256=sha(source)))
    preds={name:value.argmax(1) for name,value in alternatives.items()}
    comparisons={}
    for name,p in preds.items():
        comparisons[name]=dict(source=provenance[name],correct_on_current_errors=int((error&(p==y)).sum()),
            wrong_on_current_correct=int((~error&(p!=y)).sum()),total_correct=int((p==y).sum()),
            rescued_by_class={str(k):int((error&(y==k)&(p==y)).sum()) for k in range(5)})
    view_names=['original044_plus035','full047_plus035']+[f'raw_unbiased_view{k}' for k in range(4)]
    head_names=[f'{name}_view{k}' for k in range(4) for name in ('direct','ordinal')]+[f'expert_{name}_original' for name in ('static','dynamic','phase','temporal')]
    def any_correct(names):return np.logical_or.reduce([preds[n]==y for n in names])
    view_ok=any_correct(view_names);head_ok=any_correct(head_names);old_ok=any_correct(['cross039','context039'])
    buckets=np.full(len(y),'correct',object)
    buckets[error&view_ok]='view_or_bias_available'
    buckets[error&~view_ok&head_ok]='head_only_available'
    buckets[error&~view_ok&~head_ok&old_ok]='historical_architecture_only'
    buckets[error&~view_ok&~head_ok&~old_ok]='none_of_observed_outputs_correct'
    # A1 bias cannot reorder any pair in0/2/3/4. These wrong labels cannot be
    # made correct by any choice of the one constant while other logits stay fixed.
    others=z.copy();others[:,1]=-np.inf;other_pred=others.argmax(1)
    structurally_unfixable=error&(y!=1)&(other_pred!=y)
    combined=[]
    for i,r in enumerate(rows):
        rawrow=original[r['identifier'],r['start']]
        combined.append(dict(identifier=r['identifier'],room=r['room'],truth=int(y[i]),start=r['start'],stop=r['stop'],
            prediction=int(prediction[i]),bucket=str(buckets[i]),
            correct_in_views=[name for name in view_names if preds[name][i]==y[i]],
            correct_in_heads=[name for name in head_names if preds[name][i]==y[i]],
            correct_in_historical=[name for name in ('cross039','context039') if preds[name][i]==y[i]],
            amplitude_rms=float(rawrow['amplitude_rms']),relative_frame_change=float(rawrow['relative_frame_change']),
            top_margin=float(np.sort(z[i])[-1]-np.sort(z[i])[-2]),
            one_score_deficit=float(np.max(others[i])-z[i,1])))
    by_record=[]
    for identifier in sorted({r['identifier'] for r in rows}):
        ix=np.array([i for i,r in enumerate(rows) if r['identifier']==identifier]);bad=error[ix]
        by_record.append(dict(identifier=identifier,room=rows[ix[0]]['room'],truth=int(y[ix[0]]),windows=len(ix),errors=int(bad.sum()),
            destinations={str(k):int((bad&(prediction[ix]==k)).sum()) for k in range(5) if (bad&(prediction[ix]==k)).any()},
            historical_can_rescue=int((bad&old_ok[ix]).sum()),observed_all_wrong=int((bad&~view_ok[ix]&~head_ok[ix]&~old_ok[ix]).sum())))
    by_record.sort(key=lambda r:(-r['errors'],r['identifier']))
    one_missed=error&(y==1)&(prediction==0)
    group_stats={}
    for room in ('a1','c1','g1','e1'):
        rmask=np.array([r['room']==room for r in rows])
        for name,mask in [('missed_one',one_missed),('correct_one',(y==1)&~error),('empty',y==0)]:
            ix=np.flatnonzero(rmask&mask)
            group_stats[f'{room}_{name}']=dict(windows=len(ix),records=len({rows[i]['identifier'] for i in ix}),
                relative_frame_change=quantiles(np.array([r['relative_frame_change'] for r in combined])[ix]),
                amplitude_rms=quantiles(np.array([r['amplitude_rms'] for r in combined])[ix]))
    result=dict(platform_best=.49,correct=688,total=960,remaining_errors=int(error.sum()),independent_test=False,
        confusion_matrix=cm.tolist(),by_class={str(k):dict(total=int((y==k).sum()),errors=int((error&(y==k)).sum()),
            destinations={str(j):int((error&(y==k)&(prediction==j)).sum()) for j in range(5) if (error&(y==k)&(prediction==j)).any()}) for k in range(5)},
        missed_one_as_empty=int(one_missed.sum()),missed_one_rooms={room:int((one_missed&np.array([r['room']==room for r in rows])).sum()) for room in sorted({r['room'] for r in rows})},
        errors_not_fixable_by_one_bias_alone=int(structurally_unfixable.sum()),
        oracle_buckets={name:int((buckets==name).sum()) for name in ('view_or_bias_available','head_only_available','historical_architecture_only','none_of_observed_outputs_correct')},
        comparisons=comparisons,by_record=by_record,records_with_errors=sum(r['errors']>0 for r in by_record),
        whole_records_wrong=sum(r['errors']==r['windows'] for r in by_record),signal_proxy_stats=group_stats,
        historical_provenance=old_sources,
        limitations=['Oracle grouping uses true labels; cannot directly route inference or claim recoverable hidden accuracy.',
            'All actual backbone data are training-source; historical CSV outputs not newly executed ONNX.',
            'Approximate cached head outputs may differ near boundaries; only diagnostics.',
            'No true activity or room size labels. Relative frame change is a measured signal proxy, not person speed.',
            'Candidate0.50 remains pending; no new training, export or registry candidate replacement.'])
    # Existing room-excluded paired proxy models, each with its own earlier
    # training-side record calibration. This is an oracle comparison, not fusion.
    sys.path.insert(0,str(ROOT/'experiments/final_bias_20261024'))
    from run import proxy_logits
    proxy_complement=[]
    for room in ('a1','c1','g1','e1'):
        ix=[i for i,r in enumerate(rows) if r['room']==room]
        pe=torch.load(Path('D:/huawei2026-experiments/presence_rescue_20261024')/f'proxy_evidence_{room}.pt',weights_only=True)
        pz=proxy_logits(pe,room).numpy()[ix]+np.array([0,.35,0,0,0],np.float32)
        p=pz.argmax(1)
        with np.load(Path('D:/huawei2026-experiments/protected_context_20261024/results')/f's42_{room}'/'predictions.npz') as previous:
            assert np.array_equal(previous['labels'],y[ix])
            aux=previous['context_calibrated'].argmax(1)
        truth=y[ix]
        proxy_complement.append(dict(room=room,windows=len(ix),base_correct=int((p==truth).sum()),aux_correct=int((aux==truth).sum()),
            base_wrong_aux_right=int(((p!=truth)&(aux==truth)).sum()),base_right_aux_wrong=int(((p==truth)&(aux!=truth)).sum()),
            both_wrong=int(((p!=truth)&(aux!=truth)).sum())))
    result['independent_backbone_proxy_complement']=proxy_complement
    result['one_error_extra_bias_deficits']={str(k):quantiles(np.array([r['one_score_deficit'] for r in combined])[error&(y==1)&(prediction==k)]) for k in (0,2,3,4)}
    save(WORK/'summary.json',result);save(WORK/'windows.json',combined)
    errors=[r for r in combined if r['truth']!=r['prediction']]
    csv_save(WORK/'errors.csv',[{**r,**{k:json.dumps(r[k],ensure_ascii=False) for k in ('correct_in_views','correct_in_heads','correct_in_historical')}} for r in errors])
    csv_save(WORK/'records.csv',[{**r,'destinations':json.dumps(r['destinations'])} for r in by_record])
    save(ROOT/'checks/remaining049_summary.json',result)
    print(json.dumps({k:result[k] for k in ('remaining_errors','by_class','missed_one_rooms','errors_not_fixable_by_one_bias_alone','oracle_buckets','records_with_errors','whole_records_wrong')},ensure_ascii=False,indent=2))
    print('ALTERNATIVES',json.dumps({k:comparisons[k] for k in ('cross039','context039','original044_plus035','full047_plus035')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
