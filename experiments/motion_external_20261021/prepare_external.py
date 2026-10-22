"""Clean true0--4 labels into a single-Rx auxiliary dataset, keeping official tests separate."""
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import blosc2

WORK=Path('D:/huawei2026-experiments/motion_external_20261021')
RAW=WORK/'external/extracted'

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def scalar_refs(f,ds):return np.asarray([np.asarray(f[r]).reshape(-1)[0] for r in ds[()].reshape(-1)])
def load_capture(p,nominal_us):
    with h5py.File(p,'r') as f:
        cap=f['capture']
        if isinstance(cap,h5py.Dataset):
            v=cap[()].T
            c=(v['real']+1j*v['imag']).astype(np.complex64)
            time=np.arange(len(c),dtype=np.float64)*nominal_us/1e6
            timing='nominal_only_not_measured'
        else:
            refs=cap['csi'][()].reshape(-1);frames=[]
            for r in refs:
                v=f[r][()].reshape(-1);frames.append(v['real']+1j*v['imag'])
            c=np.asarray(frames,dtype=np.complex64)
            time=scalar_refs(f,cap['ts']).astype(np.float64)
            time-=time[0];timing='measured_pcap_timestamp'
    assert c.ndim==2 and c.shape[1]==1024
    return c,time,timing

def main():
    missing=[]
    for index in sorted((WORK/'external').glob('index_*.json')):
        for item in json.loads(index.read_text(encoding='utf-8'))['members']:
            p=Path(item['path'])
            if '__MACOSX' in p.parts or p.name.startswith('._') or p.suffix!='.mat':continue
            local=RAW/p
            if not local.exists() or local.stat().st_size!=item['bytes']:missing.append(str(p))
    if missing:raise RuntimeError(f'Download incomplete: {len(missing)} MATs missing; first={missing[:3]}')
    # Sampled exact-content anchors detect reused labeled captures even after
    # changing compression/container. This is not a proof against transformed copies.
    anchors={}
    official=Path('D:/huawei2026-data/labeled')
    for p in sorted(official.glob('CSI_*.b2')):
        c=blosc2.load_array(str(p))
        for frame in c[::100,0]:
            for rx in frame:
                key=hashlib.blake2b(np.asarray(rx,dtype=np.complex64).tobytes(),digest_size=16).hexdigest()
                anchors[key]=p.stem
        del c
    print('OFFICIAL_ANCHORS',len(anchors),flush=True)
    # Only use a real measured antenna; missing antennas are never copied.
    paths=sorted(p for p in RAW.rglob('*.mat') if '__MACOSX' not in p.parts and
                 (p.stem.endswith('core1') or 'core' not in p.stem))
    images=[];labels=[];rows=[];audit=[];seen_hashes=set()
    for p in paths:
        config=json.loads((p.parent/'config.json').read_text(encoding='utf-8-sig'))
        label=int(config['people']['num'])
        assert label in range(5) and config['bandwidth']==80 and config['modulation']=='ax'
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest in seen_hashes:
            audit.append(dict(path=str(p),excluded='duplicate SHA256',sha256=digest));continue
        seen_hashes.add(digest)
        c,time,timing=load_capture(p,config['usleep'])
        overlaps=set()
        for frame in c:
            key=hashlib.blake2b(np.asarray(frame,dtype=np.complex64).tobytes(),digest_size=16).hexdigest()
            if key in anchors:overlaps.add(anchors[key])
        if overlaps:
            audit.append(dict(path=str(p.relative_to(RAW)),sha256=digest,excluded='exact CSI anchor overlap with official labeled data',official_records=sorted(overlaps)))
            print('OVERLAP_EXCLUDED',p,sorted(overlaps),flush=True);continue
        finite=np.isfinite(c).all(1)&np.isfinite(time)
        previous=np.concatenate(([-np.inf],np.maximum.accumulate(np.where(np.isfinite(time),time,-np.inf))[:-1]))
        monotone=time>previous
        keep=finite&monotone
        removed=int((~keep).sum());c,time=c[keep],time[keep]
        time-=time[0]
        valid=np.abs(c)>1e-4
        # Authors report artifacts around MATLAB indices500--524.
        valid[:,499:524]=False
        # Match the official static route: log magnitude stabilized as .5log(1+|H|²).
        amp=.5*np.log1p(np.abs(c).astype(np.float64)**2)
        w=valid.reshape(-1,128,8);count=w.sum(-1)
        pooled=(amp.reshape(-1,128,8)*w).sum(-1)/count.clip(1)
        assert np.isfinite(pooled).all()
        delta=np.diff(time);positive=delta[delta>0]
        median_dt=float(np.median(positive));rate=float(1/median_dt)
        # Same3-second physical window when measured; explicitly nominal otherwise.
        starts=np.arange(0,max(0,time[-1]-3),3.)
        if len(starts)>20:starts=starts[np.linspace(0,len(starts)-1,20).round().astype(int)]
        session=p.parent.name
        split='external_holdout' if session.startswith('2025-08-26') else 'external_train'
        used=0;bad_gap=0
        for start in starts:
            left=int(np.searchsorted(time,start));right=int(np.searchsorted(time,start+3,side='right'))
            tt=time[left:right];aa=pooled[left:right]
            if len(tt)<30 or tt[-1]-tt[0]<2.8:continue
            if np.diff(tt).max()>max(.1,3*median_dt):bad_gap+=1;continue
            target=np.linspace(tt[0],tt[-1],150)
            # Interpolation is for the static amplitude encoder; no fast-motion claim.
            image=np.stack([np.interp(target,tt,aa[:,k]) for k in range(128)])
            center=image.mean();scale=np.sqrt(((image-center)**2).mean()+1e-4)
            image=np.clip((image-center)/scale,-6,6).astype(np.float32)
            images.append(image[None]);labels.append(label)
            rows.append(dict(source='zenodo17225694',record=str(p.relative_to(RAW)),
                             session=session,headcount=label,moving=bool(config['people']['moving']),
                             split=split,start_seconds=float(start),timing=timing,
                             antenna='core1_real_measurement',source_sha256=digest))
            used+=1
        audit.append(dict(path=str(p.relative_to(RAW)),sha256=digest,frames=len(c),removed_frames=removed,
                          label=label,median_hz=rate,timing=timing,nominal_us=config['usleep'],
                          configured_receiving_antennas=config['numAntennasRx'],selected_actual_antenna='core1',moving=config['people']['moving'],
                          notes=config.get('notes'),position_rx=config.get('posRx'),position_tx=config.get('posTx'),
                          windows=used,excluded_gap_windows=bad_gap,split=split))
        print('CLEAN',str(p.relative_to(RAW)),label,round(rate,2),used,split,flush=True)
    assert set(labels)==set(range(5)), 'Do not start training on incomplete0--4 download'
    ids_train={r['record'] for r in rows if r['split']=='external_train'}
    ids_test={r['record'] for r in rows if r['split']=='external_holdout'}
    assert ids_train.isdisjoint(ids_test)
    payload=dict(x=torch.from_numpy(np.stack(images)),y=torch.tensor(labels),rows=rows)
    torch.save(payload,WORK/'external/single_rx_features_v2.pt')
    fields=list(rows[0]);p=WORK/'external/windows_v2.csv'
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    counts={split:{str(c):sum(r['split']==split and r['headcount']==c for r in rows) for c in range(5)} for split in ['external_train','external_holdout']}
    save(WORK/'external/cleaning_audit_v2.json',dict(captures=audit,window_counts=counts,
         feature_version=2,amplitude_transform='.5*log1p(abs(CSI)**2), same as official TF route',
         previous_v1='Preserved initial log1p(abs(CSI)) illustration-style transform; not used for transfer training.',
         timestamp_note='Old denseMAT lacks timestamps; nominal times explicitly marked, not measured.',
         duplicate_check='Every100th official labeled frame, allRx, exact complex64 content against every external core1 frame; transformed/non-anchored overlap remains possible.',
         official_test_unchanged=True,no_synthetic_antennas=True,
         holdout_note='Date-heldout external validation covers only available classes; not0--4 accuracy or cross-room validation.',
         license='CC BY-SA4.0',citation='Tonini et al., Zenodo10.5281/zenodo.17225694 v2'))
    print('PREPARED',counts,flush=True)

if __name__=='__main__':main()
