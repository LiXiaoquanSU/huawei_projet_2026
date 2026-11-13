"""Extract actual ONNX preprocessing and cache all960 official labeled windows."""
import gc,time,onnx,blosc2
from onnx import helper,TensorProto,numpy_helper
from n53_common import *

def main():
    torch.set_num_threads(4);WORK.mkdir(parents=True,exist_ok=True);assert sha(SOURCE)==SOURCE_SHA
    save(WORK/'plan.json',dict(baseline_platform=.54,last_update_platform=.53,
      new_directions=['separate presence/01 transfer from high count identity','compare three scored fallback opinions via fixed categorical/ordinal pools'],
      no_new_bias_fitting=True,no_hidden_labels=True,selection='Record whole-room/whole-record and source tradeoffs. No platform improvement claims from local results. Fixed few policies, no threshold sweep.'))
    cal=onnx.load(ROOT/'models/teammate054_20261025/gcal_calibrated_fallback_web053.onnx')
    save(WORK/'gcal_bias.json',{t.name.removeprefix('__gcal_bias_'):numpy_helper.to_array(t).tolist() for t in cal.graph.initializer if t.name.startswith('__gcal_bias_')});del cal
    outputs=[f'__rxg_012___tta_v{t}_{suffix}' for t in (0,1) for suffix in ('cat_8','cat_6')]
    temp=WORK/'preprocessing.onnx'
    if not temp.exists():
        model=onnx.load(SOURCE)
        for name in outputs:model.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,963,150] if name.endswith('cat_8') else [1,17]))
        # The original graph lacks internal value_info; make endpoints explicit.
        staged=WORK/'with_pre_outputs.onnx';onnx.save(model,staged);del model
        onnx.utils.extract_model(str(staged),str(temp),['frames','intervals'],outputs,check_model=True)
    session=make_session(temp);rows=torch.load(old.CACHE,weights_only=True)['rows'];N=len(rows)
    complete=WORK/'features_complete.json'
    if not complete.exists():
        tt=np.lib.format.open_memmap(WORK/'t.npy',mode='w+',dtype=np.float32,shape=(N,2,963,150));ss=np.lib.format.open_memmap(WORK/'s.npy',mode='w+',dtype=np.float32,shape=(N,2,17));started=time.perf_counter()
        for identifier in sorted({r['identifier'] for r in rows}):
            path=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(path/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(path/f'ATI_{identifier}.b2')).reshape(-1)
            for i,row in enumerate(rows):
                if row['identifier']!=identifier:continue
                a,z=row['start'],row['stop'];feed=dict(frames=np.ascontiguousarray(csi[a:z]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[a:z].astype(np.float32,copy=True));feed['intervals'][-1]=0
                out=session.run(None,feed);tt[i]=np.stack([out[0][0],out[2][0]]);ss[i]=np.stack([out[1][0],out[3][0]])
            print('N53_FEATURES',identifier,round(time.perf_counter()-started,1),flush=True);del csi,ati;gc.collect()
        tt.flush();ss.flush();del tt,ss
        save(WORK/'rows.json',rows);save(complete,dict(windows=N,preprocessor_sha256=sha(temp),source_sha256=SOURCE_SHA,mode='actual054FLOATONNX,identityandreversefeatures,source labels untouched'))
    # Validate receiver feature permutation against actual12-view preprocessor.
    if not (WORK/'permutation_check.json').exists():
        allnames=[f'__rxg_{p}___tta_v{t}_{suffix}' for p in ('012','021','102','120','201','210') for t in (0,1) for suffix in ('cat_8','cat_6')]
        model=onnx.load(SOURCE)
        for name in allnames:model.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,963,150] if name.endswith('cat_8') else [1,17]))
        staged=WORK/'all_pre_outputs.onnx';onnx.save(model,staged);del model
        allpath=WORK/'all_preprocessing.onnx';onnx.utils.extract_model(str(staged),str(allpath),['frames','intervals'],allnames,check_model=True);full=make_session(allpath)
        ts=np.load(WORK/'t.npy',mmap_mode='r');summ=np.load(WORK/'s.npy',mmap_mode='r');cases=[]
        for i in [0,20,100,240,500,599,800,959]:
            row=rows[i];root=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(root/f"CSI_{row['identifier']}.b2"));ati=blosc2.load_array(str(root/f"ATI_{row['identifier']}.b2")).reshape(-1);a,z=row['start'],row['stop'];feed=dict(frames=np.ascontiguousarray(csi[a:z]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[a:z].astype(np.float32,copy=True));feed['intervals'][-1]=0;out=full.run(None,feed)
            errors=[]
            for k,permutation in enumerate(PERMS):
                idx,sign,si=old.permutation_indices(permutation);t=torch.from_numpy(np.array(ts[i],copy=True)).index_select(1,idx)*sign;s=torch.from_numpy(np.array(summ[i],copy=True)).index_select(1,si)
                errors.append(dict(permutation=permutation,feature_error=max(float(np.abs(t[j].numpy()-out[k*4+j*2][0]).max()) for j in (0,1)),summary_error=max(float(np.abs(s[j].numpy()-out[k*4+j*2+1][0]).max()) for j in (0,1))))
            cases.append(dict(index=i,errors=errors));del csi,ati;gc.collect()
        maxerr=max(e['feature_error'] for c in cases for e in c['errors']);save(WORK/'permutation_check.json',dict(cases=cases,max_feature_error=maxerr));assert maxerr<1e-4,maxerr
    print('N53_PREPARED',flush=True)
if __name__=='__main__':main()
