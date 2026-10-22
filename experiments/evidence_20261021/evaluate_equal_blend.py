"""Fixed equal-probability blend of the two platform .39 structures; no weight search."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=Path('D:/huawei2026-experiments/global_local_20261021/fits')
def rows(v,s,f):
    with (WORK/f'{v}_s{s}_{f}/predictions.csv').open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))
def main():
    result=[]
    for s1,s2 in ((42,2026),(2026,42)):
        rooms=[];cms=[]
        for f in ('a1','c1','g1'):
            a,b=rows('cross_events',s1,f),rows('context_events',s2,f);cm=np.zeros((5,5),int)
            for x,y in zip(a,b):
                assert (x['identifier'],x['start'])==(y['identifier'],y['start'])
                pa=np.exp([float(x[f'log_probability_{i}']) for i in range(5)]);pb=np.exp([float(y[f'log_probability_{i}']) for i in range(5)])
                label=int(x.get('headcount',x.get('label')));cm[label,int((pa+pb).argmax())]+=1
            rooms.append(float(np.trace(cm)/cm.sum()));cms.append(cm)
        cm=np.sum(cms,axis=0);mean=float(np.mean(rooms));macro=float(np.mean(np.diag(cm)/cm.sum(1)))
        result.append(dict(cross_seed=s1,context_seed=s2,rooms=rooms,room_mean=mean,macro_recall=macro,selection_score=min(mean,macro),confusion_matrix=cm.tolist()))
    out=dict(method='fixed 0.5 probability blend; cross42/context2026 is deployable seed pair, reversed seeds sensitivity only; no weight search',results=result)
    (ROOT/'checks/equal_blend_039_development.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
if __name__=='__main__':main()
