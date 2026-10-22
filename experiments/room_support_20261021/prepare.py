"""Decode Intel 5300 captures and build hardware-neutral temporal descriptors.

Intel format/scaling follows Daniel Halperin's 2008--2011 csitool MATLAB
read_bfee.c/read_bf_file.m/get_scaled_csi.m; sources are preserved with data.
No missing antennas or subcarriers are synthesized.
"""
import csv, hashlib, json, struct
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
WORK=Path('D:/huawei2026-experiments/room_support_20261021')
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def decode(path):
    raw=path.read_bytes();pos=0;records=[];bad=Counter()
    while pos+3<=len(raw):
        n=int.from_bytes(raw[pos:pos+2],'big');code=raw[pos+2];pos+=2
        if n<1 or pos+n>len(raw):bad['truncated_tail']+=1;break
        b=raw[pos+1:pos+n];pos+=n
        if code!=187:bad[f'other_code_{code}']+=1;continue
        if len(b)<20:bad['short_header']+=1;continue
        nr,nt=b[8:10];length=int.from_bytes(b[16:18],'little')
        if nr not in (1,2,3) or nt not in (1,2,3) or length!=(30*(nr*nt*16+3)+7)//8 or len(b)!=20+length:
            bad['invalid_dimensions_or_length']+=1;continue
        payload=np.frombuffer(b[20:],np.uint8).astype(np.uint32)
        index=3+np.arange(30)[:,None]*(3+16*nr*nt)+np.arange(nr*nt)[None,:]*16
        byte=index//8;shift=index%8
        r=((payload[byte]>>shift)|(payload[byte+1]<<(8-shift)))&255
        i=((payload[byte+1]>>shift)|(payload[byte+2]<<(8-shift)))&255
        # Independently check bit interpretation against little-endian bit string.
        bitstream=int.from_bytes(b[20:],'little')
        for k in (int(index[0,0]),int(index[-1,-1])):
            assert ((bitstream>>k)&65535)==int((r.flatten()[0] if k==index[0,0] else r.flatten()[-1]) | ((i.flatten()[0] if k==index[0,0] else i.flatten()[-1])<<8))
        c=((r.astype(np.int16)+128)%256-128)+1j*((i.astype(np.int16)+128)%256-128)
        c=c.reshape(30,nr,nt).transpose(2,1,0).astype(np.complex64)
        perm=[(b[15]>>(2*j))&3 for j in range(nr)]
        if sorted(perm)!=list(range(nr)):bad['invalid_permutation']+=1;continue
        cp=c.copy();cp[:,perm,:]=c;c=cp
        rss=[v for v in b[10:13] if v]
        if not rss or not np.any(c):bad['zero_signal']+=1;continue
        rssi_db=10*np.log10(sum(10**(v/10) for v in rss))-44-b[14]
        scale=10**(rssi_db/10)/(np.abs(c).astype(np.float64)**2).sum()*30
        noise=struct.unpack('b',b[13:14])[0];noise=-92 if noise==-127 else noise
        factor=np.sqrt(scale/(10**(noise/10)+scale*nr*nt))
        if nt==2:factor*=np.sqrt(2)
        if nt==3:factor*=np.sqrt(10**.45)
        c*=factor
        records.append((int.from_bytes(b[:4],'little'),c,int.from_bytes(b[18:20],'little')))
    if not records:raise ValueError(f'No CSI in {path}')
    shapes=Counter(str(x[1].shape) for x in records);shape=max(shapes,key=shapes.get)
    selected=[r for r in records if str(r[1].shape)==shape];bad['nonmodal_shape']=len(records)-len(selected)
    times=[];last=None;offset=0
    for tick,_,_ in selected:
        if last is not None and tick<last and last-tick>2**31:offset+=2**32
        times.append((tick+offset)/1e6);last=tick
    times=np.asarray(times);keep=np.r_[True,np.diff(times)>0];bad['nonpositive_timestamp']=int((~keep).sum())
    c=np.stack([v[1] for v in selected])[keep];rates=np.array([v[2] for v in selected])[keep];times=times[keep];times-=times[0]
    return c,times,rates,dict(bad),dict(shapes)

def descriptor(c,t):
    """96 features: 16 temporal statistics x six link/frequency quantiles.

    Three seconds, antialiased to 50 Hz before lag/spectrum comparisons.
    Frequency and antenna axes are pooled as observed links, never resized.
    These are amplitude-modulation spectra, NOT calibrated Doppler/velocity.
    """
    # Use up to 32 genuinely observed frequency bins per link; no upsampling.
    if c.shape[-1]>32:c=c[...,np.linspace(0,c.shape[-1]-1,32).round().astype(int)]
    a=.5*np.log1p(np.abs(c.reshape(len(c),-1)).astype(np.float64)**2)
    good=np.mean(a,axis=0)>1e-6;a=a[:,good]
    if a.shape[1]<5:raise ValueError('Too few nonzero channels')
    dt=np.median(np.diff(t));hz=1/dt
    grid=np.arange(0,3,1/hz);regular=np.stack([np.interp(grid,t,a[:,j]) for j in range(a.shape[1])],1)
    # Fine regular grid followed by low-pass polyphase resampling avoids aliasing.
    from fractions import Fraction
    frac=Fraction(50/hz).limit_denominator(1000)
    regular=resample_poly(regular,frac.numerator,frac.denominator,axis=0)
    z=np.stack([np.interp(np.arange(150)/50,np.arange(len(regular))/50,regular[:,j]) for j in range(a.shape[1])],1)
    mean=z.mean(0);std=z.std(0);center=z-mean
    norm=center/np.maximum(std,1e-5)
    features=[std/(np.abs(mean)+1e-3),np.mean(np.abs(center),0)/(np.abs(mean)+1e-3)]
    for lag in (1,2,5,10,25):features.append(np.mean(norm[lag:]*norm[:-lag],0))
    for lag in (1,5,25):features.append(np.sqrt(np.mean((z[lag:]-z[:-lag])**2,0))/(std+1e-3))
    power=np.abs(np.fft.rfft(norm*np.hanning(150)[:,None],axis=0))**2
    freq=np.fft.rfftfreq(150,.02);total=power[1:].sum(0)+1e-8
    for lo,hi in ((0,.7),(.7,2),(2,5),(5,10),(10,25.01)):
        features.append(power[(freq>lo)&(freq<=hi)].sum(0)/total)
    p=power[1:]/total;features.append(-(p*np.log(p+1e-9)).sum(0)/np.log(len(p)))
    value=np.quantile(np.stack(features),[0,.1,.25,.5,.75,1],axis=1).T.reshape(-1).astype(np.float32)
    assert value.shape==(96,) and np.isfinite(value).all();return value

def main():
    manifest=json.loads((WORK/'radiopoints_manifest.json').read_text());clean=WORK/'clean';clean.mkdir(exist_ok=True)
    rows=[];vectors=[];audit=[];digests=set()
    for item in manifest:
        assert item['sha256'] not in digests;digests.add(item['sha256'])
        p=Path(item['path']);c,t,rates,bad,shapes=decode(p)
        np.savez_compressed(clean/f"{item['room']}_{item['count']}p.npz",csi=c,time_s=t,rate_flags=rates)
        windows=0;rejected=Counter();dt=np.diff(t)
        for start in np.arange(0,max(0,t[-1]-3)+1e-6,3):
            take=(t>=start)&(t<start+3);tw=t[take]-start
            if len(tw)<60 or tw[-1]-tw[0]<2.8:rejected['insufficient_coverage']+=1;continue
            if np.max(np.diff(tw))>.1:rejected['long_gap']+=1;continue
            vectors.append(descriptor(c[take],tw));rows.append(dict(dataset='radiopoints',record=f"{item['room']}_{item['count']}p",room=item['room'],headcount=item['count'],start_s=float(start),stop_s=float(start+3),group=item['room'],source_sha256=item['sha256']))
            windows+=1
        audit.append(dict(**item,shape=list(c.shape),duration_s=float(t[-1]),median_hz=float(1/np.median(dt)),gap_max_s=float(dt.max()),rate_flags=np.unique(rates).tolist(),bandwidth_mhz=sorted(set(40 if int(x)&2048 else 20 for x in rates)),discarded=bad,original_shapes=shapes,windows=windows,rejected_windows=dict(rejected)))
        print(item['room'],item['count'],c.shape,round(t[-1],2),round(1/np.median(dt),2),windows,bad,flush=True)
    np.savez_compressed(WORK/'radiopoints_temporal_features.npz',x=np.stack(vectors),y=np.array([r['headcount'] for r in rows]),room=np.array([r['room'] for r in rows]))
    with (WORK/'radiopoints_windows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(WORK/'radiopoints_audit.json',audit)
    print('TOTAL',len(rows),sum(x['bytes'] for x in manifest),flush=True)
if __name__=='__main__':main()
