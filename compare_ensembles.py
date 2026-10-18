"""Equal-weight label-free score fusion; configurations selected on development folds."""
import itertools,json
from pathlib import Path
import numpy as np
from train_baseline import read_csv,write_csv,metrics


def path_for(name,room):
    if name in ['original','whiten','decorrelated','spectrum']:
        return Path(f'artifacts/iterations_v2/{name}_{room}_seed42/predictions.csv')
    if name in ['phase','phase_spectrum']:
        return Path(f'artifacts/iterations_phase/{name}_{room}_seed42/predictions.csv')
    if name.startswith('tfcnn'):
        return Path(f'artifacts/iterations_tf/{name}_{room}_seed42/predictions.csv')
    if name=='projected':return Path(f'artifacts/projected/best_{room}_predictions.csv')
    return Path(f'artifacts/physics/d493_rbf_l100.0_b3.0/{room}_predictions.csv')


def main():
    names=['original','whiten','decorrelated','spectrum','projected','physics','phase','phase_spectrum','tfcnn','tfcnn_mixstyle','tfcnn_dynamic','tfcnn_domain']
    rooms=['a1','c1','g1'];data={}
    names=[n for n in names if all(path_for(n,r).exists() for r in rooms)]
    for room in rooms:
        data[room]={};reference=None
        for name in names:
            rows=read_csv(path_for(name,room));keys=[(r['identifier'],r['start']) for r in rows]
            if reference is None:reference=keys
            assert keys==reference
            logits=np.array([[float(r[f'logit_{j}']) for j in range(5)] for r in rows])
            z=(logits-logits.mean(1,keepdims=True))/np.maximum(logits.std(1,keepdims=True),.01)
            probs=np.exp(z-z.max(1,keepdims=True));probs/=probs.sum(1,keepdims=True)
            data[room][name]=(probs,np.array([int(r['target']) for r in rows]),rows)
    results=[]
    for size in [1,2,3]:
        for members in itertools.combinations(names,size):
            scores=[];recalls=[]
            for room in rooms:
                probs=np.mean([data[room][n][0] for n in members],axis=0)
                labels=data[room][members[0]][1];pred=probs.argmax(1)
                m=metrics(labels,pred);scores.append(m['accuracy'])
                recalls.append(float(np.mean([v for v in m['recall'].values() if v is not None])))
            results.append(dict(members=list(members),mean_accuracy=float(np.mean(scores)),macro_recall=float(np.mean(recalls)),a1=scores[0],c1=scores[1],g1=scores[2]))
    results.sort(key=lambda r:r['mean_accuracy'],reverse=True)
    out=Path('artifacts/ensembles');out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    best=results[0]
    for room in rooms:
        probs=np.mean([data[room][n][0] for n in best['members']],axis=0)
        labels=data[room][best['members'][0]][1];rows=data[room][best['members'][0]][2]
        (out/f'{room}_metrics.json').write_text(json.dumps(metrics(labels,probs.argmax(1)),indent=2))
        write_csv(out/f'{room}_predictions.csv',[dict(identifier=r['identifier'],room=room,start=r['start'],target=int(y),predicted=int(p.argmax()),**{f'logit_{j}':float(v) for j,v in enumerate(p)}) for r,y,p in zip(rows,labels,probs)])
    print('Candidates',len(results),'top',results[:10],flush=True)


if __name__=='__main__':main()
