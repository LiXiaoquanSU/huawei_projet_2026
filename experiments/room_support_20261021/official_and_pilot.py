"""Audit missing room/count cells and run a fixed, room-disjoint transfer diagnostic.

This small RandomForest is a data/representation diagnostic, not the 0.37 model
and not an ONNX submission candidate. No validation-driven hyperparameter search.
"""
import csv,json
from collections import Counter
from pathlib import Path
import blosc2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from prepare import WORK,descriptor,save

def official():
    samples=list(csv.DictReader(Path('D:/huawei2026-data/labeled/samples.csv').open(encoding='utf-8-sig')))
    for r in samples:r['room']=r['room'] or 'unknown';r['headcount']=int(r['headcount'])
    table={room:[sum(r['room']==room and r['headcount']==y for r in samples) for y in range(5)] for room in sorted({r['room'] for r in samples})}
    save(WORK/'official_support.json',dict(record_counts=table,total_records=len(samples),note='Unknown room identifiers are not evidence of a single shared room.'))
    cache=WORK/'official_temporal_features.npz';meta=WORK/'official_windows.csv'
    if cache.exists():
        d=np.load(cache);rows=list(csv.DictReader(meta.open(encoding='utf-8')));return d['x'],d['y'],rows
    x=[];y=[];rows=[];rejected=[]
    for row in samples:
        identifier=row['identifier'];root=Path('D:/huawei2026-data/labeled')
        c=blosc2.load_array(str(root/f'CSI_{identifier}.b2'))
        ati=blosc2.load_array(str(root/f'ATI_{identifier}.b2'))
        for start in range(0,len(c)-599,600):
            intervals=ati[start:start+600].astype(np.float64)/1e6;t=np.r_[0,np.cumsum(intervals[:-1])]
            if np.any(intervals[:-1]<=0) or np.max(intervals[:-1])>.1 or t[-1]<2.8:
                rejected.append(dict(identifier=identifier,start=start,reason='time gap/nonpositive/short span'));continue
            x.append(descriptor(c[start:start+600],t));y.append(row['headcount'])
            rows.append(dict(dataset='official',record=identifier,room=row['room'],headcount=row['headcount'],start_frame=start,stop_frame=start+600))
        print('OFFICIAL',identifier,'windows',len(rows),flush=True)
    x=np.stack(x);y=np.array(y);np.savez_compressed(cache,x=x,y=y)
    with meta.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(WORK/'official_feature_rejections.json',rejected)
    return x,y,rows

def weights(rows,mass=1):
    # Equal class mass, then equal observed rooms, then equal records per cell.
    # This cannot fill unsupported room/count combinations.
    cells={};records=Counter(r['record'] for r in rows)
    for r in rows:cells.setdefault(int(r['headcount']),{}).setdefault(r['room'],set()).add(r['record'])
    return np.array([mass/len(cells)/len(cells[int(r['headcount'])])/len(cells[int(r['headcount'])][r['room']])/records[r['record']] for r in rows])
def metrics(y,p):
    cm=confusion_matrix(y,p,labels=list(range(5)));support=cm.sum(1);present=support>0
    return dict(accuracy=float(np.mean(y==p)),macro_recall_present=float(np.mean(np.diag(cm)[present]/support[present])),support=support.tolist(),confusion_matrix=cm.tolist())
def fit(x,y,w,test):
    model=RandomForestClassifier(n_estimators=192,max_depth=10,min_samples_leaf=3,max_features='sqrt',random_state=42,n_jobs=6)
    model.fit(x,y,sample_weight=w);return model.predict(test)
def main():
    x,y,rows=official();d=np.load(WORK/'radiopoints_temporal_features.npz');ex,ey=d['x'],d['y']
    erows=list(csv.DictReader((WORK/'radiopoints_windows.csv').open(encoding='utf-8')));results=[];predictions=[]
    for room in ('a1','b1','c1','e1','g1'):
        ti=np.array([i for i,r in enumerate(rows) if r['room'] not in (room,'unknown')]);vi=np.array([i for i,r in enumerate(rows) if r['room']==room])
        tr=[rows[i] for i in ti];assert {r['record'] for r in tr}.isdisjoint({rows[i]['record'] for i in vi})
        for variant in ('official_only','official_plus_radiopoints'):
            xx=x[ti];yy=y[ti];ww=weights(tr)
            if variant.endswith('radiopoints'):xx=np.concatenate([xx,ex]);yy=np.concatenate([yy,ey]);ww=np.concatenate([ww,weights(erows,.25)])
            p=fit(xx,yy,ww,x[vi]);m=dict(fold=room,variant=variant,train_records=len({r['record'] for r in tr}),external_records=15 if variant.endswith('radiopoints') else 0,**metrics(y[vi],p));results.append(m)
            for i,pred in zip(vi,p):predictions.append(dict(**rows[i],variant=variant,predicted=int(pred)))
            print('RESULT',json.dumps(m),flush=True)
    for room in ('RoomA','RoomB','RoomC'):
        ti=np.array([i for i,r in enumerate(erows) if r['room']!=room]);vi=np.array([i for i,r in enumerate(erows) if r['room']==room])
        p=fit(ex[ti],ey[ti],weights([erows[i] for i in ti]),ex[vi]);m=dict(fold=room,variant='radiopoints_leave_room_out',**metrics(ey[vi],p));results.append(m);print('RESULT',json.dumps(m),flush=True)
    save(WORK/'pilot_results.json',dict(seed=42,model='RandomForest192 depth10 minleaf3',external_weight=.25,features=96,results=results,interpretation='Diagnostic only. All windows of a record/room stay together. No official test labels or pseudo-labels used.'))
    with (WORK/'pilot_predictions.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
if __name__=='__main__':main()
