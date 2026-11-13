"""Append explicit covariance-moment evidence to the exact scored049 graph."""
import copy,gc
import blosc2,onnx,onnxruntime as ort
from onnx import helper,TensorProto
from run import *
SOURCE=ROOT/'models/final_bias_20261024/01_final_bias_one035.onnx'
SOURCE_SHA='d083e3bb36f522e29b45acb88e41e5577013f9140b2f59dd2f9741c476f839df'
EXPORT=WORK/'export'
BASE='__correlation049_base';FEATURES='__rxf_v0___tta_v0_cat_8'

class Fusion(nn.Module):
    def __init__(self):
        super().__init__();self.descriptor=Descriptor();heads=[]
        for name in ('actual_no_clock_final.pt','no_clock_actual_normalized_seed2026_final.pt'):
            cp=torch.load(WORK/name,weights_only=True);assert cp['chosen_step']==50 and len(cp['fit_indices'])==960
            state=cp['state'];h=CountHead(state['mean'],state['std']);h.load_state_dict(state);heads.append(h)
        self.heads=nn.ModuleList(heads)
    def forward(self,base,t):
        f=self.descriptor(t,t.new_zeros((1,17)),False,False)
        p=(self.heads[0](f).exp()+self.heads[1](f).exp())*.5
        candidate=(.5*base.softmax(-1)+.5*p[0]).clamp_min(1e-8).log()
        result=torch.where(base.argmax(-1)<=1,base,candidate)
        return result,p[0],f[0]

def sha(p):
    import hashlib
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def assemble():
    torch.set_num_threads(4);EXPORT.mkdir(parents=True,exist_ok=True);assert sha(SOURCE)==SOURCE_SHA
    head=Fusion().eval();m=onnx.load(SOURCE);source=copy.deepcopy(m);assert m.graph.node[-1].output[0]=='logits';m.graph.node[-1].output[0]=BASE
    assert FEATURES in {v for n in m.graph.node for v in n.output}
    torch.onnx.export(head,(torch.zeros(5),torch.zeros(1,963,150)),str(EXPORT/'head.onnx'),input_names=[BASE,FEATURES],output_names=['__corr_result','__corr_probability','__corr_descriptor'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(EXPORT/'head.onnx');mapping={x.name:'__corr_aux_'+x.name for x in small.graph.initializer}
    for n in small.graph.node:
        for v in n.output:mapping[v]='logits' if v=='__corr_result' else '__corr_aux_'+v
    for x in small.graph.initializer:x.name=mapping[x.name]
    for n in small.graph.node:
        n.name='__corr_aux_'+n.name
        for i,v in enumerate(n.input):n.input[i]=mapping.get(v,v)
        for i,v in enumerate(n.output):n.output[i]=mapping.get(v,v)
    m.graph.node.extend(small.graph.node);m.graph.initializer.extend(small.graph.initializer)
    for old,new in zip(source.graph.node,m.graph.node):
        check=copy.deepcopy(new)
        if BASE in check.output:check.output[0]='logits'
        assert old.SerializeToString()==check.SerializeToString()
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(source.graph.initializer,m.graph.initializer))
    assert m.graph.input==source.graph.input and m.graph.output==source.graph.output
    props={v.key:v.value for v in source.metadata_props}
    props.update(platform_accuracy='not evaluated',base_sha256=SOURCE_SHA,
      transformation='Retain exact scored049 output if base argmax0/1. Else equal probability mixture with two-seed explicit covariance-moment + ACF evidence branch.',
      training='Two114->64->32->5 softmax heads, AdamW400-step ablation, final checkpoint50steps fixed from proxy calibrations, seeds42/2026. Only48 original labeled recordings; no external,U or pseudo labels. Backbone and biases unchanged.',
      caveat='Normalized second-order signal evidence is not literal rank=count. Source trained accuracy not independent. New branch reads neither absolute amplitude summary nor ATI summary. Platform improvement unknown.')
    helper.set_model_props(m,props);onnx.checker.check_model(m);onnx.save(m,EXPORT/'model.onnx')
    save(EXPORT/'structure_proof.json',dict(base_sha256=SOURCE_SHA,candidate_sha256=sha(EXPORT/'model.onnx'),original_nodes=len(source.graph.node),added_nodes=len(small.graph.node),all_original_nodes_and_initializers_preserved=True,
      base_predicted01_outputs_exactly_preserved=True,new_classifier_parameters=sum(p.numel() for h in head.heads for p in h.parameters()),extra_backbone_inferences=0,
      heads={name:sha(WORK/name) for name in ('actual_no_clock_final.pt','no_clock_actual_normalized_seed2026_final.pt')}))
    del m,source,small;gc.collect()
    return head

def audit(head):
    inst=onnx.load(EXPORT/'model.onnx')
    for name,shape in [(BASE,[5]),(FEATURES,[1,963,150])]:inst.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shape))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(inst.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);del inst
    cache=torch.load(WORK/'features.pt',weights_only=True);rows=cache['rows'];predictions=[];references=[];errors=[];base_errors=[];protected=0
    with np.load(Path('D:/huawei2026-experiments/final_bias_20261024/cached_exact048.npz')) as v:exactbase=v['logits']+np.array([0,.35,0,0,0],np.float32)
    lookup={(r['identifier'],r['start']):i for i,r in enumerate(rows)}
    with torch.no_grad():
        for identifier in sorted({r['identifier'] for r in rows}):
            rawdir=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(rawdir/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(rawdir/f'ATI_{identifier}.b2')).reshape(-1)
            for r in [r for r in rows if r['identifier']==identifier]:
                start,stop=r['start'],r['stop'];inputs=dict(frames=np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[start:stop].astype(np.float32,copy=True));inputs['intervals'][-1]=0
                out,base,features=session.run(None,inputs);expected,aux,_=head(torch.from_numpy(base),torch.from_numpy(features))
                error=float(np.abs(out-expected.numpy()).max());assert error<2e-4,error;errors.append(error)
                base_error=float(np.abs(base-exactbase[lookup[identifier,start]]).max());assert base_error<2e-5,base_error;base_errors.append(base_error)
                if base.argmax()<=1:assert np.array_equal(base,out);protected+=1
                p=dict(identifier=identifier,start=start,room=r['room'],truth=r['headcount'],base=int(base.argmax()),prediction=int(out.argmax()),logits=out.tolist(),aux_prediction=int(aux.argmax()))
                predictions.append(p)
                kind=('corrected' if out.argmax()==r['headcount'] else 'broken') if out.argmax()!=base.argmax() else ('protected' if base.argmax()<=1 else 'unchanged')
                if not any(ref.get('kind')==kind for ref in references):
                    path=EXPORT/f'fixture_{kind}.npz';np.savez_compressed(path,**inputs);references.append(dict(input=str(path),logits=expected.numpy().tolist(),kind=kind))
            print('CORRELATION_RAW_RECORD',identifier,flush=True);del csi,ati;gc.collect()
        for path in [ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero']:
            if path=='zero':inputs=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
            else:
                with np.load(path) as v:inputs={k:v[k].copy() for k in ('frames','intervals')}
            out,base,features=session.run(None,inputs);expected,_,_=head(torch.from_numpy(base),torch.from_numpy(features));assert np.abs(out-expected.numpy()).max()<2e-4
            references.append(dict(input=str(path),logits=expected.numpy().tolist()))
    del session;gc.collect();cm=np.zeros((5,5),np.int64)
    for r in predictions:cm[r['truth'],r['prediction']]+=1
    result=dict(raw_windows=len(predictions),total=960,correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/960),corrected=sum(r['prediction']==r['truth'] and r['base']!=r['truth'] for r in predictions),
      broken=sum(r['prediction']!=r['truth'] and r['base']==r['truth'] for r in predictions),confusion_matrix=cm.tolist(),
      raw_protected_windows=protected,independent_torch_auxiliary_logit_max_error=max(errors),original049_logit_max_error=max(base_errors),source_diagnostic_only=True,platform_score=None)
    save(EXPORT/'validation.json',result);save(EXPORT/'all_predictions.json',predictions);save(EXPORT/'references.json',references)
    print('CORRELATION_EXPORTED',result,flush=True)

if __name__=='__main__':audit(assemble())
