"""Locate raw-input ORT/Torch disagreement without relaxing tolerances."""
from export import *
def main():
    torch.set_num_threads(4);d=torch.load(WORK/'actual.pt',weights_only=True);row=d['rows'][599];root=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(root/f"CSI_{row['identifier']}.b2"));ati=blosc2.load_array(str(root/f"ATI_{row['identifier']}.b2")).reshape(-1);start=row['start']
    feed=dict(frames=np.ascontiguousarray(csi[start:start+600]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[start:start+600].astype(np.float32,copy=True));feed['intervals'][-1]=0
    m=onnx.load(EXPORT/'model.onnx');names=[f'__rxg_{p}___tta_v{t}_logits' for p in ('012','021','102','120','201','210') for t in (0,1)]
    for n in names:m.graph.output.append(helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,[5]))
    for name,shape in [('slice_15',[1,3,128,150]),('slice_16',[1,3,128,150]),('view_25',[1,6,32,150])]:m.graph.output.append(helper.make_tensor_value_info('__rxg_012___tta_v0_'+name,onnx.TensorProto.FLOAT,shape))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;s=ort.InferenceSession(m.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);out=s.run(None,feed);del s,m
    model=RawReference().eval();frames=torch.from_numpy(feed['frames']);intervals=torch.from_numpy(feed['intervals']);views=ReceiverViews();errors=[]
    with torch.inference_mode():
        x=torch.stack([frames,frames.flip(0)]);dt=torch.stack([intervals,intervals.index_select(0,model.reverse)]);t,s=model.model.preprocess(x.double(),dt.double());tt,ss=views(t.float(),s.float());feature_raw=model.model.forward_features(tt,ss)+model.bias
        f32,_=model.model.preprocess(x.float(),dt.float())
        for dtype,ft in [('double',t.float()),('float',f32)]:
            vals=[ft[0,:384].reshape(1,3,128,150),ft[0,384:768].reshape(1,3,128,150),ft[0,768:960].reshape(1,6,32,150)]
            print('PREPROCESS',dtype,[float(np.abs(a-b.numpy()).max()) for a,b in zip(out[13:],vals)],flush=True)
        for k,p in enumerate(PERMS):
            xx=x[:,:,:,list(p)].contiguous();rt,rs=model.model.preprocess(xx.double(),dt.double());direct=model.model.forward_features(rt.float(),rs.float())+model.bias
            for direction in (0,1):errors.append(dict(permutation=p,time=direction,feature_permutation_error=float((tt[k*2+direction]-rt.float()[direction]).abs().max()),summary_error=float((ss[k*2+direction]-rs.float()[direction]).abs().max()),onnx_vs_feature_logit_error=float(np.abs(out[1+k*2+direction]-feature_raw[k*2+direction].numpy()).max()),onnx_vs_direct_logit_error=float(np.abs(out[1+k*2+direction]-direct[direction].numpy()).max())))
    print(errors,flush=True);save(WORK/'numerical_diagnosis.json',dict(row=row,cases=errors))
if __name__=='__main__':main()
