"""Record-level count trends: exploratory diagnostics, never target feature selection."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/g60_20261025'))
import g60_common as old
BASE=Path('D:/huawei2026-experiments/count_order_20261029');CACHE=Path('D:/huawei2026-experiments/next053_20261025')

def main():
    BASE.mkdir(exist_ok=True);rows=old.read(CACHE/'rows.json');t=np.load(CACHE/'t.npy',mmap_mode='r');s=np.load(CACHE/'s.npy')[:,0];n=len(rows)
    features={}
    for name,c in [('amplitude',np.array(t[:,0,384:768]).reshape(n,3,32,4,150).mean(3)),('phase',np.array(t[:,0,768:960]).reshape(n,3,64,150))]:
        c-=c.mean(-1,keepdims=True);cov=(c@c.swapaxes(-1,-2))/150
        ev=np.linalg.eigvalsh(cov).clip(0);p=ev/ev.sum(-1,keepdims=True).clip(1e-10)
        metrics={'effective_rank':np.exp(-(p*np.log(p.clip(1e-10))).sum(-1)),'participation_rank':1/np.square(p).sum(-1).clip(1e-10),'largest_component':p[:,:,-1]}
        power=np.square(np.abs(np.fft.rfft(c,axis=-1)))[:,:, :,1:].sum(2);power/=power.sum(-1,keepdims=True).clip(1e-10)
        metrics.update({f'cycles_{lo}_{hi}':power[:,:,lo-1:hi].sum(-1) for lo,hi in [(1,3),(4,10),(11,30),(31,75)]})
        for key,value in metrics.items():features[f'{name}_{key}']=value.mean(1)
    for group,key in enumerate(('log_amplitude','amplitude_std','motion_energy','derivative_energy')):features[key]=s[:,group*3:group*3+3].mean(1)
    names=list(features);x=np.stack(list(features.values()),1);record=[]
    for ident in sorted({r['identifier'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['identifier']==ident];r=rows[ix[0]];record.append(dict(identifier=ident,room=r['room'],count=r['headcount'],features=x[ix].mean(0).tolist()))
    trends=[];rng=np.random.default_rng(20261029)
    for j,name in enumerate(names):
        tr=dict(feature=name)
        for room in ('a1','c1'):
            rr=[r for r in record if r['room']==room and 1<=r['count']<=4];xx=np.array([r['features'][j] for r in rr]);yy=np.array([r['count'] for r in rr]);rho=float(spearmanr(xx,yy).statistic)
            null=[float(spearmanr(xx,rng.permutation(yy)).statistic) for _ in range(1000)]
            tr[room]=dict(records=len(rr),rho=rho,permutation_p=(1+int((np.abs(null)>=abs(rho)).sum()))/1001,class_means={str(k):float(xx[yy==k].mean()) for k in range(1,5)})
        tr['same_direction']=tr['a1']['rho']*tr['c1']['rho']>0;trends.append(tr)
    old.save(BASE/'trends.json',dict(features=names,records=record,trends=trends,limitations='Exploratory correlations on only11a1 and12c1 records,1-4people; unadjusted permutationp values across18features are not confirmatory. Not used to select input features on evaluation folds. Rank/spectral power from normalized cached CSI are not numbers of people or independent human sources.',cache_sha=old.sha(CACHE/'rows.json')))
    print(json.dumps([(r['feature'],r['a1']['rho'],r['c1']['rho']) for r in trends]),flush=True)
if __name__=='__main__':main()
