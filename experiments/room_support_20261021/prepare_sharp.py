"""Adapt the public SHARP subset using author-documented stream/frequency layout.

All timing is NOMINAL (paper Tc ~=6ms); MATs lack per-packet timestamps.
SHARP is related data, not an authenticated copy of the gated 23.6GB collection.
"""
import csv,json,re
from collections import Counter
import numpy as np
from scipy.io import loadmat
from prepare import WORK,descriptor,save
from official_and_pilot import fit,metrics,weights
def main():
    root=WORK/'sharp_public';manifest=json.loads((root/'manifest.json').read_text());rows=[];x=[];audit=[]
    clean=root/'clean';clean.mkdir(exist_ok=True)
    for item in manifest:
        name=item['member'];session=name.split('/')[0];room={'S1a':'bedroom','S6a':'living_room','S7a':'laboratory'}[session]
        act=re.search(r'_(E|L|W|R)',name)[1];activity={'E':'empty','L':'sitting','W':'walking','R':'running'}[act]
        a=loadmat(item['path'])['csi_buff'];shape=a.shape;assert a.shape[1]==256
        # Format derived from author's preprocessing code, not value inference.
        a=np.fft.fftshift(a,axes=1);a=a[:len(a)//4*4].reshape(-1,1,4,256)
        a=a[...,np.setdiff1d(np.arange(256),[0,1,2,3,4,5,127,128,129,251,252,253,254,255])].astype(np.complex64)
        # Drop complete 4-stream packet groups; deleting individual rows would
        # change antenna identities. Preserve original packet index/time gaps.
        good=np.isfinite(a).all((1,2,3))&((np.abs(a)**2).sum(3)>0).all((1,2))
        packet_index=np.arange(len(a))[good];a=a[good];t=packet_index*.006
        # The source MAT is retained and hashed. Save the cleaning recipe/index
        # rather than a second ~160MB complex matrix per recording.
        np.savez(clean/(name.split('/')[-1].removesuffix('.mat')+'_index.npz'),time_s=t,packet_index=packet_index,valid_subcarriers=np.setdiff1d(np.arange(256),[0,1,2,3,4,5,127,128,129,251,252,253,254,255]))
        windows=0
        for start in np.arange(0,max(0,t[-1]-3)+1e-8,3):
            take=(t>=start)&(t<start+3);tt=t[take]-start
            if len(tt)<60 or tt[-1]-tt[0]<2.8 or np.max(np.diff(tt))>.1:continue
            x.append(descriptor(a[take],tt));rows.append(dict(dataset='sharp_public',record=name,room=room,session=session,headcount=0 if act=='E' else 1,activity=activity,start_s=float(start),timing='nominal_6ms_NOT_measured',source_sha256=item['sha256']));windows+=1
        audit.append(dict(**item,room=room,activity=activity,original_shape=list(shape),shape=list(a.shape),dropped_packet_groups=int((~good).sum()),trailing_rows=shape[0]%4,timing='nominal_6ms_NOT_measured',windows=windows))
        print('SHARP',name,a.shape,windows,flush=True)
    x=np.stack(x);y=np.array([r['headcount'] for r in rows]);activity=np.array([['empty','sitting','walking','running'].index(r['activity']) for r in rows]);rooms=np.array([r['room'] for r in rows])
    np.savez_compressed(root/'temporal_features.npz',x=x,y=y,activity=activity,room=rooms)
    with (root/'windows.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(root/'audit.json',audit);results=[]
    for room in sorted(set(rooms)):
        ti=np.where(rooms!=room)[0];vi=np.where(rooms==room)[0]
        for task,labels in [('presence',y),('activity',activity)]:
            rr=[dict(rows[i],headcount=int(labels[i])) for i in ti]
            p=fit(x[ti],labels[ti],weights(rr),x[vi]);m=dict(room=room,task=task,**metrics(labels[vi],p));results.append(m);print('SHARP_RESULT',json.dumps(m),flush=True)
    save(root/'pilot_results.json',dict(results=results,note='Only 0/1 headcount and four activity states; nominal time; fixed RF diagnostic, not a people-counting submission score.'))
if __name__=='__main__':main()
