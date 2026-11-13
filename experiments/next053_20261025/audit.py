"""All960rawwindows in actual batchedONNX, with same-pass054control."""
import gc,blosc2,onnx
from onnx import helper,TensorProto
from n53_common import *
from export_batch import FeatureHead,NAMES
def main():
    selection=read(WORK/'selection.json');folder=WORK/'export';torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    m=onnx.load(folder/'model.onnx');prep=onnx.load(WORK/'preprocessing.onnx');assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(prep.graph.node,m.graph.node));del prep
    m.graph.output.append(helper.make_tensor_value_info('__n53_control',TensorProto.FLOAT,[5]))
    for name in NAMES:m.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,963,150] if name.endswith('cat_8') else [1,17]))
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;session=ort.InferenceSession(m.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);del m
    head=FeatureHead(selection['method']).cuda().eval();rows=read(WORK/'rows.json');records=[];references=[];maxp=0.;maxl=0.;borderline=0
    with torch.inference_mode():
        for identifier in sorted({r['identifier'] for r in rows}):
            root=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(root/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(root/f'ATI_{identifier}.b2')).reshape(-1)
            for i,row in enumerate(rows):
                if row['identifier']!=identifier:continue
                a,z=row['start'],row['stop'];feed=dict(frames=np.ascontiguousarray(csi[a:z]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[a:z].astype(np.float32,copy=True));feed['intervals'][-1]=0
                out,base,*features=session.run(None,feed);ref=head(*[torch.from_numpy(v).cuda() for v in features]).cpu().numpy()
                if selection['method']=='transfer_low_guard' and base.argmax()>=2:assert np.array_equal(out,base),'Protected highcount output changed'
                def prob(q):p=np.exp(q-q.max());return p/p.sum()
                pe=float(np.abs(prob(out)-prob(ref)).max());le=float(np.abs(out-ref).max());assert pe<.002 and le<.02,(i,pe,le)
                margin=float(np.sort(prob(ref))[-1]-np.sort(prob(ref))[-2]);same=out.argmax()==ref.argmax();assert same or margin<.002
                borderline+=int(not same);maxp=max(maxp,pe);maxl=max(maxl,le)
                truth=int(row['headcount']);pred=int(out.argmax());bp=int(base.argmax());records.append(dict(index=i,identifier=identifier,start=a,truth=truth,prediction=pred,base=bp,logits=out.tolist(),base_logits=base.tolist()))
                kind='corrected' if pred==truth and bp!=truth else 'broken' if pred!=truth and bp==truth else 'unchanged'
                if not any(r.get('kind')==kind for r in references):
                    path=folder/f'fixture_{kind}.npz';np.savez_compressed(path,**feed);references.append(dict(input=str(path),logits=ref.tolist(),kind=kind))
            save(folder/'audit_progress.json',dict(candidate_sha256=sha(folder/'model.onnx'),records=records));print('N53_RAW',identifier,len(records),flush=True);del csi,ati;gc.collect()
        for path in (ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero'):
            if path=='zero':feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
            else:
                with np.load(path) as f:feed={k:f[k].copy() for k in ('frames','intervals')}
            out,base,*features=session.run(None,feed);ref=head(*[torch.from_numpy(v).cuda() for v in features]).cpu().numpy();assert np.abs(out-ref).max()<.02;references.append(dict(input=str(path),logits=ref.tolist()))
    cm=np.zeros((5,5),np.int64);bcm=cm.copy()
    for r in records:cm[r['truth'],r['prediction']]+=1;bcm[r['truth'],r['base']]+=1
    validation=dict(raw_windows=len(records),correct=int(np.trace(cm)),base_correct=int(np.trace(bcm)),accuracy=float(np.trace(cm)/len(records)),
      corrected=sum(r['prediction']==r['truth'] and r['base']!=r['truth'] for r in records),broken=sum(r['prediction']!=r['truth'] and r['base']==r['truth'] for r in records),
      confusion_matrix=cm.tolist(),baseline_confusion_matrix=bcm.tolist(),independent_encoder_probability_max_error=maxp,independent_encoder_logit_max_error=maxl,borderline_reference_argmax_disagreements=borderline,
      preprocessor_exact_original_nodes=True,scope='All960actualrawONNX. Original054-style control in same batched graph. Torch reference uses same actual preprocessingoutputs, not an independently approximated preprocessing.',source_labels_trained=True,platform_score=None)
    save(folder/'references.json',references);save(folder/'all_predictions.json',records);save(folder/'validation.json',validation);print('N53_VALIDATED',validation,flush=True)
if __name__=='__main__':main()
