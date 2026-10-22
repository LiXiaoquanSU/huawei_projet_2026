"""Paired error analysis only; oracle uses labels and is not a deployable model."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OLD=Path('D:/huawei2026-experiments/structure_events_20261021');WORK=Path('D:/huawei2026-experiments/global_local_20261021')
def readrows(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    results=[];unique_windows=set()
    for seed in (42,2026):
        for fold in ('a1','c1','g1'):
            rows=[readrows(OLD/'fits'/f'{v}_s{seed}_{fold}/predictions.csv') for v in ('baseline','events')];counts=dict(both_correct=0,only_baseline_correct=0,only_events_correct=0,both_wrong=0,disagreement=0)
            for a,b in zip(*rows):
                assert tuple(a[k] for k in ('identifier','start','stop','headcount'))==tuple(b[k] for k in ('identifier','start','stop','headcount'))
                unique_windows.add(tuple(a[k] for k in ('identifier','start','stop')))
                ca=a['prediction']==a['headcount'];cb=b['prediction']==b['headcount'];key='both_correct' if ca and cb else 'only_baseline_correct' if ca else 'only_events_correct' if cb else 'both_wrong';counts[key]+=1;counts['disagreement']+=a['prediction']!=b['prediction']
            results.append(dict(seed=seed,fold=fold,windows=len(rows[0]),**counts))
    totals={k:sum(r[k] for r in results) for k in ('windows','both_correct','only_baseline_correct','only_events_correct','both_wrong','disagreement')}
    totals['unique_physical_windows']=len(unique_windows)
    totals['label_using_oracle_accuracy']=(totals['both_correct']+totals['only_baseline_correct']+totals['only_events_correct'])/totals['windows']
    out=dict(results=results,totals=totals,note='paired reused fold errors; oracle chooses by true label, cannot deploy and is not an accuracy forecast; no blend weights searched')
    (WORK/'complementarity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(totals),flush=True)
if __name__=='__main__':main()
