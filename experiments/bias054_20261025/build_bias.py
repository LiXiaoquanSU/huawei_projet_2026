"""Append one Add to exact scored054 artifact, audit960original raw windows."""
import gc,time,blosc2,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from evaluate_bias import *

def engine(model):
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    return ort.InferenceSession(model,sess_options=opts,providers=['CPUExecutionProvider'])
def prob(z):p=np.exp(z-z.max());return p/p.sum()

def main():
    torch.set_num_threads(4);s=read(BWORK/'selection.json');assert s['eligible'];source=BASES[s['base']]['path'];assert sha(source)==BASES[s['base']]['sha256']
    folder=BWORK/'export';folder.mkdir(exist_ok=True);delta=np.asarray(s['delta'],np.float32);original=onnx.load(source);m=onnx.ModelProto();m.CopyFrom(original)
    producers=[n for n in m.graph.node if 'logits' in n.output];assert len(producers)==1 and not any('logits' in n.input for n in m.graph.node)
    producer=producers[0];producer.output[list(producer.output).index('logits')]='__bias054_control'
    m.graph.initializer.append(numpy_helper.from_array(delta,'__bias054_delta'));m.graph.node.append(helper.make_node('Add',['__bias054_control','__bias054_delta'],['logits'],name='__bias054_final_add'))
    for a,b in zip(original.graph.node,m.graph.node):
        n=onnx.NodeProto();n.CopyFrom(b)
        for i,name in enumerate(n.output):
            if name=='__bias054_control':n.output[i]='logits'
        assert a.SerializeToString()==n.SerializeToString()
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(original.graph.initializer,m.graph.initializer));assert original.graph.input==m.graph.input and original.graph.output==m.graph.output
    props={p.key:p.value for p in original.metadata_props};props.update(base_sha256=sha(source),base_platform_accuracy='0.54 userreported',platform_accuracy='not evaluated',method='OnlyonefinalAdd,delta'+json.dumps(s['delta']),weights='alloriginalnodes/initializerspreserved;internalbiasandroutingunchanged',caveat='Biasselectedonrepeateddevelopmentrooms,notblindtest;sourcehastrainedlabels.')
    helper.set_model_props(m,props);onnx.checker.check_model(m);onnx.save(m,folder/'model.onnx');proof=dict(base_sha256=sha(source),candidate_sha256=sha(folder/'model.onnx'),base=s['base'],delta=s['delta'],added_nodes=1,added_float32_values=5,original_nodes_initializers_inputs_outputs_preserved_except_output_rename=True,bytes=(folder/'model.onnx').stat().st_size);save(folder/'structure_proof.json',proof)
    del original;gc.collect();m.graph.output.append(helper.make_tensor_value_info('__bias054_control',TensorProto.FLOAT,[5]));session=engine(m.SerializeToString());del m;gc.collect()
    cached,y,rows=data('actual',s['base']);cached=cached.numpy();records=[];refs=[];maxp=0.;maxl=0.;borderline=0
    for identifier in sorted({r['identifier'] for r in rows}):
        root=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(root/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(root/f'ATI_{identifier}.b2')).reshape(-1)
        for i,row in enumerate(rows):
            if row['identifier']!=identifier:continue
            a,z=row['start'],row['stop'];feed=dict(frames=np.ascontiguousarray(csi[a:z]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[a:z].astype(np.float32,copy=True));feed['intervals'][-1]=0
            out,base=session.run(None,feed);assert np.array_equal(out,base+delta),'OnlyfinalAddmustbeexact'
            pe=float(np.abs(prob(base)-prob(cached[i])).max());le=float(np.abs(base-cached[i]).max());assert pe<.002 and le<.02,(i,pe,le);margin=float(np.sort(prob(cached[i]))[-1]-np.sort(prob(cached[i]))[-2]);same=base.argmax()==cached[i].argmax();assert same or margin<.002;borderline+=not same;maxp=max(maxp,pe);maxl=max(maxl,le)
            truth=int(y[i]);pred=int(out.argmax());bp=int(base.argmax());records.append(dict(index=i,identifier=identifier,start=a,truth=truth,prediction=pred,base=bp,logits=out.tolist(),base_logits=base.tolist()))
            kind='corrected' if pred==truth and bp!=truth else 'broken' if pred!=truth and bp==truth else 'unchanged'
            if not any(r.get('kind')==kind for r in refs):
                path=folder/f'fixture_{kind}.npz';np.savez_compressed(path,**feed);refs.append(dict(input=str(path),logits=(base+delta).tolist(),kind=kind))
        save(folder/'audit_progress.json',dict(candidate_sha256=proof['candidate_sha256'],records=records));print('BIAS_RAW',identifier,len(records),flush=True);del csi,ati;gc.collect()
    for path in (ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero'):
        if path=='zero':feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
        else:
            with np.load(path) as f:feed={k:f[k].copy() for k in ('frames','intervals')}
        out,base=session.run(None,feed);assert np.isfinite(out).all() and np.array_equal(out,base+delta);refs.append(dict(input=str(path),logits=(base+delta).tolist()))
    cm=np.zeros((5,5),np.int64);bcm=cm.copy()
    for r in records:cm[r['truth'],r['prediction']]+=1;bcm[r['truth'],r['base']]+=1
    v=dict(raw_windows=len(records),correct=int(cm.trace()),base_correct=int(bcm.trace()),accuracy=float(cm.trace()/len(records)),corrected=sum(r['prediction']==r['truth'] and r['base']!=r['truth'] for r in records),broken=sum(r['prediction']!=r['truth'] and r['base']==r['truth'] for r in records),confusion_matrix=cm.tolist(),baseline_confusion_matrix=bcm.tolist(),all_outputs_exact_base_plus_delta=True,cached_network_probability_max_error=maxp,cached_network_logit_max_error=maxl,borderline_argmax_disagreements=int(borderline),source_labels_trained=True,platform_score=None)
    save(folder/'validation.json',v);save(folder/'all_predictions.json',records);save(folder/'references.json',refs);print('BIAS_VALIDATED',v,flush=True)
if __name__=='__main__':main()
