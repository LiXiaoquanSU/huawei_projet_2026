"""Real-clock overlapping 600-frame windows; per-record memory-mapped shards."""
import gc,time
from pathlib import Path
import blosc2
import numpy as np
import torch
from common import ROOT,WORK,RAW,read,save,sha,write_csv,old_data,build
from model import StructuredPreprocessor

def starts_from_clock(ati):
    n=len(ati);clock=np.r_[0,np.cumsum(np.maximum(ati[:-1].astype(np.float64),0))/1e6]
    ticks=np.arange(int(np.floor(clock[n-600]*2))+1,dtype=np.int64)
    starts=np.searchsorted(clock,ticks*.5,side='left');half={int(k) for k in starts if k<=n-600};one={int(k) for k,t in zip(starts,ticks) if t%2==0 and k<=n-600}
    baseline=set(range(0,n-599,600));assert len(baseline)==20
    # Old anchors are retained in both expanded versions to preserve an exact control set.
    return clock,baseline,one|baseline,half|baseline

def main():
    torch.set_num_threads(4);blosc2.set_nthreads(1);WORK.mkdir(parents=True,exist_ok=True)
    old=old_data();records={r['identifier']:{k:r[k] for k in ('identifier','layout','room','headcount')} for r in old['rows']};assert len(records)==48
    assert len(list(RAW.glob('CSI_*.b2')))==48
    previous={(r['identifier'],int(r['start'])):i for i,r in enumerate(old['rows'])}
    hashes={k:sha(ROOT/'src'/k) for k in old['source_hashes']};pre=StructuredPreprocessor(128).cuda().eval();allrows=[];audits=[];started=time.perf_counter();probes={}
    for number,(ident,meta) in enumerate(sorted(records.items()),1):
        folder=WORK/'cache/shards'/ident;folder.mkdir(parents=True,exist_ok=True);done=folder/'record.json'
        if done.exists():
            item=read(done);assert item['source_hashes']==hashes;allrows.extend(item['rows']);audits.append(item['audit']);continue
        csi=blosc2.load_array(str(RAW/f'CSI_{ident}.b2'));ati=blosc2.load_array(str(RAW/f'ATI_{ident}.b2'))
        assert csi.shape==(12000,1,3,1024) and csi.dtype==np.complex64 and ati.shape==(12000,) and np.isfinite(ati).all()
        clock,base,one,half=starts_from_clock(ati);starts=sorted(half);rows=[]
        for local,k in enumerate(starts):
            rows.append(dict(**meta,start=k,stop=k+600,local_index=local,baseline=k in base,step_1s=k in one,step_0p5s=True,start_seconds=float(clock[k]),window_seconds=float(clock[k+599]-clock[k])))
        tmap=np.lib.format.open_memmap(folder/'t.partial.npy',mode='w+',dtype=np.float32,shape=(len(rows),963,150));smap=np.lib.format.open_memmap(folder/'s.partial.npy',mode='w+',dtype=np.float32,shape=(len(rows),17))
        for j,r in enumerate(rows):
            if r['baseline']:
                i=previous[(ident,r['start'])];tmap[j]=old['t'][i].numpy();smap[j]=old['s'][i].numpy()
        todo=[j for j,r in enumerate(rows) if not r['baseline']];parity=[];prediction_parity={}
        # Check two old anchors against raw double precision preprocessing per record.
        batches=[([next(i for i,r in enumerate(rows) if r['start']==k) for k in (0,11400)],True)]+[(todo[j:j+4],False) for j in range(0,len(todo),4)]
        with torch.no_grad():
            for ix,check in batches:
                xs=[];ds=[]
                for j in ix:
                    k=rows[j]['start'];xs.append(csi[k:k+600].view(np.float32).reshape(600,1,3,1024,2).copy());dt=ati[k:k+600].copy();dt[-1]=0;ds.append(dt)
                x=torch.from_numpy(np.stack(xs)).cuda().double();dt=torch.from_numpy(np.stack(ds)).cuda().double();t,s=pre(x,dt);t=t.float().cpu().numpy();s=s.float().cpu().numpy()
                assert np.isfinite(t).all() and np.isfinite(s).all()
                if check:
                    fe=float(np.max(np.abs(t-tmap[ix])));se=float(np.max(np.abs(s-smap[ix])));assert fe<.01 and se<.001,(ident,fe,se);parity=[fe,se]
                    if fe>=.001:
                        if not probes:
                            for name,seed in (('cross_events',42),('context_events',2026),('mixer',42)):
                                oldfolder='fresh_backbones_20261022' if name=='mixer' else 'global_local_20261021'
                                cp=torch.load(Path('D:/huawei2026-experiments')/oldfolder/'fits'/f'{name}_s{seed}_full/final.pt',map_location='cpu',weights_only=True)
                                model=build(name).cuda().eval();model.load_state_dict(cp['model_state']);probes[name]=model
                            del cp
                        errors={}
                        for name,model in probes.items():
                            before=model.forward_features(torch.from_numpy(tmap[ix].copy()).cuda(),torch.from_numpy(smap[ix].copy()).cuda()).softmax(-1)
                            after=model.forward_features(torch.from_numpy(t).cuda(),torch.from_numpy(s).cuda()).softmax(-1)
                            errors[name]=float((before-after).abs().max());assert errors[name]<.005,(ident,name,fe,errors[name])
                        print('PRECISION_PARITY',ident,fe,errors,flush=True)
                        prediction_parity=errors
                else:tmap[ix]=t;smap[ix]=s
        tmap.flush();smap.flush();del tmap,smap
        (folder/'t.partial.npy').replace(folder/'t.npy');(folder/'s.partial.npy').replace(folder/'s.npy')
        audit=dict(identifier=ident,baseline=len(base),step_1s=len(one),step_0p5s=len(half),duration_seconds=float(clock[-1]),negative_intervals=int((ati<0).sum()),raw_cache_errors=parity,prediction_parity_if_feature_error_over_1e_3=prediction_parity,csi_sha256=sha(RAW/f'CSI_{ident}.b2'),ati_sha256=sha(RAW/f'ATI_{ident}.b2'),cache_t_sha256=sha(folder/'t.npy'),cache_s_sha256=sha(folder/'s.npy'))
        save(done,dict(rows=rows,audit=audit,source_hashes=hashes));allrows.extend(rows);audits.append(audit)
        del csi,ati;gc.collect();print('CACHE',number,48,ident,'counts',len(base),len(one),len(half),'seconds',round(time.perf_counter()-started,1),flush=True)
    assert len({r['identifier'] for r in allrows})==48 and sum(r['baseline'] for r in allrows)==960
    counts={k:sum(r[k] for r in allrows) for k in ('baseline','step_1s','step_0p5s')}
    save(WORK/'cache/index.json',dict(rows=allrows,source_hashes=hashes,counts=counts,records=48,window_frames=600,clock_unit='seconds from nonnegative ATI microseconds',baseline_reused_exactly=True,independent_records_added=0,raw_audits=audits))
    write_csv(WORK/'cache/windows.csv',allrows);save(ROOT/'checks/sliding_cache_20261022.json',dict(status='pass',counts=counts,records=48,raw_records_verified=48,clock_strides_seconds=[1,.5],window_frames=600,baseline_reused_exactly=True,max_raw_feature_error=max(a['raw_cache_errors'][0] for a in audits)))
    print('CACHE_COMPLETE',counts,flush=True)
if __name__=='__main__':main()
