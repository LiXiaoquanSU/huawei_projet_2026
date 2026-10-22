"""Clean new counting records with measured timestamps and grouped test rooms."""
import csv,hashlib,json,re
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from acquire import WORK,save
OLD=Path('D:/huawei2026-experiments/room_support_20261021')
def integer(value,empty=None):
    a=np.asarray(value)
    return empty if a.size==0 else int(a.reshape(-1)[0])
def dynamic(v):
    v=v-v.mean(-1,keepdims=True);v=v/np.sqrt(np.mean(v*v)+1e-6)
    return np.clip(v,-6,6).astype(np.float32)
def regularize(a,t):
    dt=np.diff(t);hz=1/np.median(dt[dt>0]);grid=np.arange(0,t[-1],1/hz)
    regular=np.stack([np.interp(grid,t,a[:,k]) for k in range(a.shape[1])],1)
    ratio=Fraction(50/hz).limit_denominator(1000);out_hz=hz*ratio.numerator/ratio.denominator
    filtered=resample_poly(regular,ratio.numerator,ratio.denominator,axis=0,window=('kaiser',8.6))
    return filtered,np.arange(len(filtered))/out_hz,float(hz),float(out_hz)
def main():
    root=WORK/'ehunam';selection=json.loads((root/'selection.json').read_text());manifest={r['member']:r for r in json.loads((root/'download_manifest.json').read_text())}
    result={'ehu_train':dict(x=[],y=[],rows=[]),'ehu_test':dict(x=[],y=[],rows=[]),'radio':dict(x=[],y=[],rows=[])};audit=[];seen=set()
    existing=WORK/'external_dynamic.pt';incremental=existing.exists()
    if incremental:
        result=torch.load(existing,map_location='cpu',weights_only=True)
        for value in result.values():value['x']=[v.numpy() for v in value['x']];value['y']=value['y'].tolist()
        audit=json.loads((WORK/'cleaning_audit.json').read_text())['ehu_records'];seen={r['sha256'] for r in audit}
    completed={r['record'] for r in audit}
    for meta in selection['records']:
        file=meta['File'];entry=manifest[file];path=Path(entry['path'])
        if file in completed:continue
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest==entry['sha256'] and digest not in seen;seen.add(digest)
        d=loadmat(path,simplify_cells=True);label=integer(d['N_People']);application=d['Application']
        if application=='E':assert meta['N_People']==0 and label in (None,0);label=0
        else:assert application=='PC' and label==meta['N_People'] and label in (1,2,3,4) and len(str(d['People']))==label
        assert int(d['BW'])==80 and int(d['Rx'])==1 and d['Enviroment']==meta['Enviroment']
        assert meta['N_Machine']==0 and integer(d['N_Machine']) in (None,0)
        c=d['CSI'];t=np.asarray(d['TimeStamp'],dtype=np.float64).reshape(-1);assert c.shape==(len(t),256)
        mask=np.isfinite(t)&np.isfinite(c).all(1);mask&=t>np.maximum.accumulate(np.r_[-np.inf,t[:-1]])
        c=c[mask];t=t[mask];t-=t[0];dt=np.diff(t);duration=float(t[-1]);gaps=t[1:][dt>.1]
        # Exactly the author's MC1/80MHz mask, not SHARP's different mask.
        carriers=np.setdiff1d(np.arange(256),np.r_[0:6,127:131,251:256]);assert len(carriers)==int(d['Occupied_SC'])==241
        # Keep the physical80MHz axis, including holes:256 measured bins become
        # 128 adjacent pairs, matching the original model's80MHz/128 scale.
        valid=np.abs(c)>1e-5;valid[:,np.setdiff1d(np.arange(256),carriers)]=False
        a=.5*np.log1p(np.abs(c).astype(np.float32)**2)
        groups=np.arange(256).reshape(128,2)
        a=np.stack([(a[:,g]*valid[:,g]).sum(1)/np.maximum(valid[:,g].sum(1),1) for g in groups],1)
        filtered,clock,hz,out_hz=regularize(a,t);target='ehu_test' if meta['Enviroment']=='Office 1' else 'ehu_train';used=0;excluded=Counter()
        for start in np.arange(5,duration-8+1e-8,3):
            take=(t>=start)&(t<start+3);tw=t[take]-start
            if len(tw)<60 or tw[-1]-tw[0]<2.8:excluded['coverage_or_frames']+=1;continue
            if np.any((gaps>=start)&(gaps<start+3)):excluded['gap_gt_100ms']+=1;continue
            desired=start+np.arange(150)/50;v=np.stack([np.interp(desired,clock,filtered[:,k]) for k in range(128)])
            result[target]['x'].append(dynamic(v)[None]);result[target]['y'].append(label)
            result[target]['rows'].append(dict(source='EHUNAM_figshare28541225',record=file,room=meta['Enviroment'],session=meta['Set'],headcount=label,start_seconds=float(start),time_source='measured_seconds_TimeStamp',receiving_device=1,actual_single_rx=True,original_file=d['Original_File'],author_date=meta['Date'],source_sha256=digest,split=target))
            used+=1
        if used==0 and duration<13:excluded['too_short_after_5s_edge_trim']=1
        audit.append(dict(record=file,label=label,room=meta['Enviroment'],split=target,raw_frames=len(mask),discarded_frames=int((~mask).sum()),duration_s=duration,median_hz=hz,resampled_hz=out_hz,gaps_gt_100ms=len(gaps),windows=used,excluded=dict(excluded),label_source='ApplicationE plus author summary0' if application=='E' else 'explicitN_People and People crosscheck',original_file=d['Original_File'],author_date=meta['Date'],null_bins_removed=15,sha256=digest))
        print('EHU_CLEAN',file,label,meta['Enviroment'],round(hz,1),used,dict(excluded),flush=True)
        del d,c,a,filtered
    with (OLD/'radiopoints_windows.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    for record in dict.fromkeys(r['record'] for r in rows):
        if incremental:continue
        d=np.load(OLD/'clean'/f'{record}.npz');c,t=d['csi'],d['time_s'];a=.5*np.log1p(np.abs(c).astype(np.float32)**2);shape=a.shape
        regular,clock,_,_=regularize(a.reshape(len(a),-1),t)
        for r in [v for v in rows if v['record']==record]:
            start=float(r['start_s'])
            if start<3 or start+3>t[-1]-3:continue
            target=start+np.arange(150)/50
            v=np.stack([np.interp(target,clock,regular[:,k]) for k in range(regular.shape[1])]).reshape(shape[1],shape[2],shape[3],150)
            for tx in range(shape[1]):
                for rx in range(shape[2]):
                    result['radio']['x'].append(dynamic(v[tx,rx])[None]);result['radio']['y'].append(int(r['headcount']))
                    result['radio']['rows'].append(dict(r,source='RadioPoints',actual_tx=tx,actual_rx=rx,split='external_train',record=record))
        print('RADIO_CLEAN',record,flush=True)
    for key,v in result.items():
        v['x']=torch.from_numpy(np.stack(v['x']));v['y']=torch.tensor(v['y']);assert torch.isfinite(v['x']).all()
        assert set(v['y'].tolist())==set(range(5)),(key,torch.bincount(v['y'],minlength=5))
        print('DYNAMIC_READY',key,list(v['x'].shape),torch.bincount(v['y'],minlength=5).tolist(),flush=True)
        with (WORK/f'{key}_windows.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(v['rows'][0]));w.writeheader();w.writerows(v['rows'])
    assert set(r['room'] for r in result['ehu_train']['rows']).isdisjoint({r['room'] for r in result['ehu_test']['rows']})
    assert {r['record'] for r in result['ehu_train']['rows']}.isdisjoint({r['record'] for r in result['ehu_test']['rows']})
    torch.save(result,WORK/'external_dynamic.pt');save(WORK/'cleaning_audit.json',dict(ehu_records=audit,summary={k:dict(shape=list(v['x'].shape),records=len({r['record'] for r in v['rows']}),class_counts=torch.bincount(v['y'],minlength=5).tolist()) for k,v in result.items()},office1_held_out=True,no_synthetic_antennas=True,no_nominal_timestamps=True,edge_trim_ehu_s=5,edge_trim_radio_s=3))
if __name__=='__main__':main()
