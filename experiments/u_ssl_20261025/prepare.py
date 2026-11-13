"""Official U raw audit/cache and frozen encoder-prefix activations."""
import argparse,gc
import blosc2
from ssl_common import *
from model import StructuredPreprocessor

def raw_cache():
    folder=WORK/'u_features';folder.mkdir(parents=True,exist_ok=True)
    pre=StructuredPreprocessor(128).cuda().eval();reverse=torch.cat([torch.arange(598,-1,-1),torch.tensor([599])]).cuda()
    records=[]
    for path in sorted(Path('D:/huawei2026-data/unlabeled').glob('CSI_*.b2')):
        identifier=path.stem.removeprefix('CSI_');ati_path=path.with_name('ATI_'+identifier+'.b2');dest=folder/(identifier+'.pt');meta_path=dest.with_suffix('.json')
        if dest.exists() and meta_path.exists():records.append(read(meta_path));continue
        csi=blosc2.load_array(str(path));ati=blosc2.load_array(str(ati_path)).reshape(-1)
        assert csi.shape==(12000,1,3,1024) and csi.dtype==np.complex64,(identifier,csi.shape,csi.dtype)
        assert ati.shape==(12000,) and np.isfinite(ati).all()
        finite=int(np.isfinite(csi).sum());assert finite==csi.size
        active=float((np.abs(csi)>1e-4).mean())
        parts=identifier.split('_');meta=dict(identifier=identifier,layout=parts[0],room=parts[1],csi_sha256=sha(path),ati_sha256=sha(ati_path),csi_dtype=str(csi.dtype),shape=list(csi.shape),finite_complex_count=finite,active_power_gt1e_8_fraction=active,
          ati_nonpositive=int((ati[:-1]<=0).sum()),ati_negative=int((ati[:-1]<0).sum()),windows=20,labels=None,preprocessing='float64 raw->float32 cached; correct reverse ATI and per-window final0')
        tt=[];ss=[]
        with torch.inference_mode():
            for offset in range(0,20,2):
                starts=[offset*600,(offset+1)*600]
                x=np.stack([np.ascontiguousarray(csi[s:s+600]).view(np.float32).reshape(600,1,3,1024,2) for s in starts]);dt=np.stack([ati[s:s+600].astype(np.float32,copy=True) for s in starts]);dt[:,-1]=0
                x=torch.from_numpy(x).cuda().double();dt=torch.from_numpy(dt).cuda().double()
                a,b=pre(x,dt);ra,rb=pre(x.flip(1),dt.index_select(1,reverse))
                tt.append(torch.stack([a.float().cpu(),ra.float().cpu()],1));ss.append(torch.stack([b.float().cpu(),rb.float().cpu()],1))
        rows=[dict(identifier=identifier,room=meta['room'],layout=meta['layout'],start=i*600,stop=(i+1)*600) for i in range(20)]
        torch.save(dict(t=torch.cat(tt),s=torch.cat(ss),rows=rows,metadata=meta),dest);save(meta_path,meta);records.append(meta)
        del csi,ati;gc.collect();print('U_RAW_CACHED',len(records),identifier,active,flush=True)
    assert len(records)==150;save(WORK/'u_raw_manifest.json',records)

def prefix(model,t,s,index,sign,si):
    views=[(t[:,0],s[:,0]),(t[:,1],s[:,1]),(t[:,0].index_select(1,index)*sign,s[:,0].index_select(1,si)),(t[:,1].index_select(1,index)*sign,s[:,1].index_select(1,si))]
    phase=[];time=[];phase_out=[];time_out=[];summaries=[];captured={}
    def phase_hook(module,args):captured['phase']=args[0]
    def time_hook(module,args):captured['time']=args[0]
    hp=model.phase.project.register_forward_pre_hook(phase_hook);ht=model.temporal.blocks[-1].register_forward_pre_hook(time_hook)
    for tt,ss in views:
        pp=model.phase(tt[:,768:960].reshape(-1,6,32,150));tm=model.temporal(tt[:,384:768].reshape(-1,3,128,150),tt[:,768:960].reshape(-1,6,32,150),tt[:,960:])
        phase.append(captured['phase'].cpu());time.append(captured['time'].cpu());phase_out.append(pp.cpu());time_out.append(tm.cpu());summaries.append(ss.cpu())
    hp.remove();ht.remove()
    return dict(phase=torch.stack(phase,1),time=torch.stack(time,1),phase_out=torch.stack(phase_out,1),time_out=torch.stack(time_out,1),summaries=torch.stack(summaries,1))

def activation_cache(room):
    out=WORK/'prefixes'/room;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():assert read(out/'complete.json')['checkpoint_sha256']==sha(checkpoint(room));return
    model,bias=build(checkpoint(room),full=room=='actual');model=model.cuda().eval()
    index,sign,si=[v.cuda() for v in permutation_indices((2,1,0))]
    if not (out/'source.pt').exists():
        d=torch.load(CACHE,weights_only=True);rev=torch.load(OLD/'reverse_features.pt',weights_only=True);assert d['rows']==rev['rows']
        chunks=[]
        with torch.inference_mode():
            for start in range(0,960,16):
                t=torch.stack([d['t'][start:start+16],rev['t'][start:start+16]],1).cuda();s=torch.stack([d['s'][start:start+16],rev['s'][start:start+16]],1).cuda()
                chunks.append(prefix(model,t,s,index,sign,si))
        source={k:torch.cat([c[k] for c in chunks]) for k in chunks[0]};old=torch.load(old_latents(room),weights_only=True)
        source.update(labels=d['y'],rows=d['rows'],latents=old['latents'],logits=old['logits'])
        assert float((source['phase_out']-old['latents'][:,:,2]).abs().max())<1e-4
        assert float((source['time_out']-old['latents'][:,:,3]).abs().max())<1e-4
        torch.save(source,out/'source.pt');del d,rev,old,source,chunks;gc.collect();print('SOURCE_PREFIX',room,flush=True)
    manifest=read(WORK/'u_raw_manifest.json');n=3000
    phase=np.lib.format.open_memmap(out/'u_phase.npy',mode='w+',dtype=np.float32,shape=(n,4,2304))
    time=np.lib.format.open_memmap(out/'u_time.npy',mode='w+',dtype=np.float32,shape=(n,4,75,256));rows=[];start=0
    with torch.inference_mode():
        for r,record in enumerate(manifest):
            cached=torch.load(WORK/'u_features'/f"{record['identifier']}.pt",weights_only=True)
            for offset in range(0,20,10):
                values=prefix(model,cached['t'][offset:offset+10].cuda(),cached['s'][offset:offset+10].cuda(),index,sign,si)
                phase[start:start+10]=values['phase'].numpy();time[start:start+10]=values['time'].numpy();start+=10
            rows.extend(cached['rows'])
            if r%15==14:print('U_PREFIX',room,r+1,flush=True)
    assert start==n;phase.flush();time.flush();del phase,time,model;gc.collect();torch.cuda.empty_cache()
    save(out/'u_rows.json',rows);save(out/'complete.json',dict(checkpoint_sha256=sha(checkpoint(room)),source_windows=960,u_windows=n,views=4,prefix_cache_float32=True,no_labels_for_U=True))

def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;WORK.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('--room',default='all');a=p.parse_args()
    raw_cache()
    for room in ROUTES if a.room=='all' else (a.room,):activation_cache(room)
if __name__=='__main__':main()
