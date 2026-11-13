"""Within-record activity/error association; no activity labels or model tuning."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
EVID=Path('D:/huawei2026-experiments/subnet_expectation_20261025/evidence')
OUT=ROOT/'docs/ACTIVITY_AUDIT_20261027.json'

def main():
    torch.set_num_threads(4);records=[]
    for room in ('a1','c1','g1','e1','actual'):
        d=torch.load(EVID/f'{room}.pt',weights_only=True);b,T=old.calibration(room);pred=old.policy(d['raw'],b,T).argmax(-1);y=d['labels'];rows=d['rows'];ix=old.split(room,rows)[2]
        s=d['s'][:,0,0];motion=s[:,6:9].mean(1).numpy();derivative=s[:,9:12].mean(1).numpy()
        for ident in sorted({rows[i]['identifier'] for i in ix}):
            jj=np.array([i for i in ix if rows[i]['identifier']==ident]);correct=(pred[jj]==y[jj]).numpy();r=dict(evaluation=room,room=rows[jj[0]]['room'],identifier=ident,label=int(y[jj[0]]),windows=len(jj),correct=int(correct.sum()),predicted_counts=torch.bincount(pred[jj],minlength=5).tolist(),activity={})
            for name,values in [('amplitude_motion',motion),('amplitude_derivative',derivative)]:
                v=values[jj];order=np.argsort(v,kind='stable');n=len(order)//2;low=order[:n];high=order[-n:];informative=bool(v.max()-v.min()>1e-6 and v[high].mean()>v[low].mean())
                r['activity'][name]=dict(mean=float(v.mean()),spread=float(v.max()-v.min()),low_accuracy=float(correct[low].mean()),high_accuracy=float(correct[high].mean()),high_minus_low=float(correct[high].mean()-correct[low].mean()),informative=informative)
            records.append(r)
    result=dict(protocol='Within each complete record compare lower/upper halves of measured motion; same room and count. Descriptive reused-room analysis. Actual backbone saw all labeled records and is training-fit only.',records=records,summary={})
    rng=np.random.default_rng(20261027)
    for population in ('heldout_rooms','actual_training_fit'):
        rr=[r for r in records if (r['evaluation']=='actual')==(population=='actual_training_fit')]
        for name in ('amplitude_motion','amplitude_derivative'):
            valid=[r for r in rr if r['activity'][name]['informative']];delta=np.array([r['activity'][name]['high_minus_low'] for r in valid]);boots=[]
            for _ in range(5000):
                pick=[]
                for room in sorted({r['room'] for r in valid}):
                    group=[r for r in valid if r['room']==room];pick.extend(group[i]['activity'][name]['high_minus_low'] for i in rng.integers(0,len(group),len(group)))
                boots.append(np.mean(pick))
            result['summary'][population+'_'+name]=dict(records=len(valid),low_accuracy=float(np.mean([r['activity'][name]['low_accuracy'] for r in valid])),high_accuracy=float(np.mean([r['activity'][name]['high_accuracy'] for r in valid])),mean_high_minus_low=float(delta.mean()),record_bootstrap_ci95=np.percentile(boots,[2.5,97.5]).tolist(),by_class={str(c):dict(records=len(q),low=float(np.mean([r['activity'][name]['low_accuracy'] for r in q])),high=float(np.mean([r['activity'][name]['high_accuracy'] for r in q]))) for c in range(5) if (q:=[r for r in valid if r['label']==c])})
    old.save(OUT,result);print(json.dumps(result['summary'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
