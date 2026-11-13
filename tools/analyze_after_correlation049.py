"""Read-only source-error and official-unlabeled ATI audit for next-round design."""
import json,sys
from pathlib import Path
import numpy as np,torch,blosc2
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'checks/after_correlation049_20261025.json'
OLD=Path('D:/huawei2026-experiments/correlation049_20261024')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def distribution(values):
    a=np.asarray(values);return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),maximum=float(a.max())) if len(a) else None
def main():
    rows=read(OLD/'export/all_predictions.json');assert len(rows)==960
    base=np.array([r['base'] for r in rows]);new=np.array([r['prediction'] for r in rows]);y=np.array([r['truth'] for r in rows]);groups={}
    with np.load(OLD/'no_clock_replicate_actual_ensemble.npz') as v:aux=np.exp(v['head_logits'])
    data=torch.load(OLD/'features.pt',weights_only=True);lookup={(r['identifier'],r['start']):i for i,r in enumerate(data['rows'])};aux=aux[[lookup[r['identifier'],r['start']] for r in rows]]
    for name,mask in dict(missed_one=(base==0)&(y==1),correct_empty=(base==0)&(y==0),false_one=(base==1)&(y==0),correct_one=(base==1)&(y==1)).items():
        label=1 if name in ('missed_one','correct_empty') else 0
        votes=aux.argmax(-1)==label
        groups[name]=dict(windows=int(mask.sum()),aux_votes_for_alternative=int((mask&votes).sum()),
          aux_probability_for_alternative=distribution(aux[mask,label]),rooms={room:int((mask&np.array([r['room']==room for r in rows])).sum()) for room in sorted({r['room'] for r in rows})})
    changed=base!=new
    changes=[]
    for oldlabel in range(5):
        for newlabel in range(5):
            mask=changed&(base==oldlabel)&(new==newlabel)
            if mask.any():changes.append(dict(before=oldlabel,after=newlabel,windows=int(mask.sum()),corrected=int((mask&(new==y)).sum()),broken=int((mask&(base==y)).sum())))
    unlabeled=[]
    for path in sorted(Path('D:/huawei2026-data/unlabeled').glob('ATI_*.b2')):
        identifier=path.stem.removeprefix('ATI_');parts=identifier.split('_');dt=blosc2.load_array(str(path)).reshape(-1).astype(np.float64)
        valid=dt[:-1][dt[:-1]>0];csi=path.with_name('CSI_'+identifier+'.b2');assert csi.exists()
        unlabeled.append(dict(identifier=identifier,layout=parts[0],room=parts[1],packets=len(dt),csi_compressed_bytes=csi.stat().st_size,
          positive_dt_us_mean=float(valid.mean()) if len(valid) else None,positive_dt_us_median=float(np.median(valid)) if len(valid) else None,
          dt_over20ms_fraction=float((dt[:-1]>20000).mean()),nonpositive_fraction=float((dt[:-1]<=0).mean())))
    u_groups={}
    for room in sorted({r['room'] for r in unlabeled}):
        rr=[r for r in unlabeled if r['room']==room]
        u_groups[room]=dict(records=len(rr),packets=sorted({r['packets'] for r in rr}),mean_positive_dt_us=distribution([r['positive_dt_us_mean'] for r in rr if r['positive_dt_us_mean'] is not None]),gap20ms_fraction=distribution([r['dt_over20ms_fraction'] for r in rr]))
    report=dict(scored_best=.49,covariance_platform=.49,source_correct=int((new==y).sum()),source_errors=int((new!=y).sum()),
      base01_windows=int((base<=1).sum()),base01_errors=int(((base<=1)&(base!=y)).sum()),
      base01_unchanged=bool(np.array_equal(new[base<=1],base[base<=1])),source_errors_in_preserved01=int(((base<=1)&(new!=y)).sum()),
      changed_windows=int(changed.sum()),corrected=int((changed&(new==y)).sum()),broken=int((changed&(base==y)).sum()),changed_records=len({r['identifier'] for r,m in zip(rows,changed) if m}),changes=changes,groups=groups,
      unlabeled_records=len(unlabeled),unlabeled_groups=u_groups,unlabeled_record_metadata=unlabeled,
      caveats=['No hidden per-sample labels; equal rounded platform score cannot localize gained/lost cases.',
       'Aux votes on source errors use heads fit on those records; not independent rescue estimates.',
       'U ATI profiling does not infer people labels or prove U/T room matching. Room/layout parsed from filenames, not independently verified physical metadata.',
       'Only ATI and file metadata audited; no U CSI feature cache, training or new submission export.'])
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='unlabeled_record_metadata'},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
