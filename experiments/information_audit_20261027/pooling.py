"""Descriptive amplitude pooling residuals; signal energy is not label information."""
import sys,json
from pathlib import Path
import numpy as np
import torch,blosc2
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('D:/huawei2026-data/labeled')

def main():
    torch.set_num_threads(4)
    rows=json.loads(Path('D:/huawei2026-experiments/next053_20261025/rows.json').read_text());out=[]
    for ident in sorted({r['identifier'] for r in rows}):
        rr=[r for r in rows if r['identifier']==ident];r=rr[len(rr)//2]
        raw=blosc2.load_array(str(DATA/f'CSI_{ident}.b2'));a=np.ascontiguousarray(raw[r['start']:r['stop']]).reshape(600,3,1024);power=np.abs(a).astype(np.float64)**2;valid=power>1e-8;logamp=.5*np.log1p(power);del raw,a
        # Fixed commonly active bins avoid interpreting absent-carrier padding as signal.
        active=valid.all(0);group=logamp.reshape(600,3,128,8);mask=valid.reshape(600,3,128,8);pooled=(group*mask).sum(-1)/np.maximum(mask.sum(-1),1)
        dynamic=logamp-logamp.mean(0,keepdims=True);recon=np.repeat(pooled-pooled.mean(0,keepdims=True),8,axis=-1)
        energy=float((dynamic**2*active).sum());freq_res=float(((dynamic-recon)**2*active).sum()/max(energy,1e-20))
        pd=pooled-pooled.mean(0,keepdims=True);pr=np.repeat(pd.reshape(150,4,3,128).mean(1),4,axis=0);ga=active.reshape(3,128,8).all(-1)
        temporal_res=float(((pd-pr)**2*ga).sum()/max(float((pd**2*ga).sum()),1e-20))
        residual=dynamic-recon
        def lagcorr(x,lag):
            u=x[:-lag]*active;v=x[lag:]*active
            return float((u*v).sum()/max(float(np.sqrt((u*u).sum()*(v*v).sum())),1e-20))
        correlations={f'{name}_lag{lag}':lagcorr(x,lag) for name,x in [('discarded_frequency',residual),('retained_frequency',recon)] for lag in (1,4)}
        ati=blosc2.load_array(str(DATA/f'ATI_{ident}.b2')).reshape(-1)[r['start']:r['stop']-1].astype(float);dt=ati[ati>0]/1e6
        out.append(dict(identifier=ident,room=r['room'],label=r['headcount'],start=r['start'],active_fraction=float(active.mean()),dynamic_energy=energy,frequency8_relative_squared_residual=freq_res,time4_relative_squared_residual=temporal_res,duration_seconds=float(dt.sum()),median_interval_seconds=float(np.median(dt)),max_interval_seconds=float(dt.max())))
        out[-1].update(correlations)
        if len(out)%8==0:print('AUDITED',len(out),flush=True)
    result=dict(protocol='One fixed middle 600-frame window from each of 48 labeled records. Log amplitude only; no labels used for choosing windows. Residual/energy is a signal reconstruction statistic, not lost class information; high-frequency content can be noise.',records=out,summary={k:dict(median=float(np.median([r[k] for r in out])),q25=float(np.quantile([r[k] for r in out],.25)),q75=float(np.quantile([r[k] for r in out],.75)),minimum=float(min(r[k] for r in out)),maximum=float(max(r[k] for r in out))) for k in ('frequency8_relative_squared_residual','time4_relative_squared_residual','duration_seconds','active_fraction')})
    for key in correlations:
        values=[r[key] for r in out];result['summary'][key]=dict(median=float(np.median(values)),q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)))
    (ROOT/'docs/POOLING_AUDIT_20261027.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()
