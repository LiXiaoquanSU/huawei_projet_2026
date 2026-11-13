"""Replace encoder weights directly in054; graph and routing stay unchanged."""
import gc,time,blosc2,onnx,onnxruntime as ort
from onnx import numpy_helper,helper
from g60_common import *
from quantized import updated_state
from common import ReceiverViews
EXPORT=WORK/'export'

class LegacyONNXLog1p(torch.overrides.TorchFunctionMode):
    """Match the scored graph's exported Add(1,x)->Log, not native log1p."""
    def __torch_function__(self,func,types,args=(),kwargs=None):
        if func is torch.log1p:return torch.log(args[0]+1)
        return func(*args,**(kwargs or {}))

class RawReference(nn.Module):
    def __init__(self):
        super().__init__();self.model,b=build(checkpoint('actual'),full=True);state,_=updated_state('actual');self.model.load_state_dict(state,strict=False)
        self.register_buffer('bias',b);self.views=ReceiverViews()
        self.register_buffer('reverse',torch.cat([torch.arange(598,-1,-1),torch.tensor([599])]))
    def forward(self,frames,intervals):
        x=torch.stack([frames,frames.flip(0)]);dt=torch.stack([intervals,intervals.index_select(0,self.reverse)])
        # The scored044/054 graph preprocesses in FLOAT, not DOUBLE. Historical
        # feature caches used DOUBLE and can differ on low-variance windows.
        with LegacyONNXLog1p():t,s=self.model.preprocess(x.float(),dt.float())
        t,s=self.views(t.float(),s.float());raw=self.model.forward_features(t,s).reshape(1,6,2,5)
        return policy(raw,self.bias)[0]

def assemble():
    EXPORT.mkdir(parents=True,exist_ok=True);assert sha(SOURCE)==SOURCE_SHA
    m=onnx.load(SOURCE);original=onnx.ModelProto();original.CopyFrom(m);state,provenance=updated_state('actual')
    arrays={t.name:t for t in m.graph.initializer};casts={n.output[0]:n.input[0] for n in m.graph.node if n.op_type=='Cast' and n.input[0] in arrays};changed=[];targets=set()
    for name,value in state.items():
        key='model.'+name;target=key if key in arrays else key+'__fp16' if key+'__fp16' in arrays else casts.get(key);assert target is not None,name;assert target not in targets;targets.add(target)
        old=numpy_helper.to_array(arrays[target]);new=value.numpy().astype(old.dtype);assert new.shape==old.shape,(name,new.shape,old.shape)
        old_digest=hashlib.sha256(old.tobytes()).hexdigest();arrays[target].CopyFrom(numpy_helper.from_array(new.copy(),name=target));changed.append(dict(parameter=name,initializer=target,shape=list(new.shape),dtype=str(new.dtype),old_tensor_sha256=old_digest,new_tensor_sha256=hashlib.sha256(new.tobytes()).hexdigest()))
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(original.graph.node,m.graph.node))
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(original.graph.initializer,m.graph.initializer) if a.name not in targets)
    props={p.key:p.value for p in m.metadata_props if not p.key.startswith('claude_') and p.key!='base_submission'};props.update(base_model_sha256=SOURCE_SHA,base_accuracy='0.54 (user reported)',platform_accuracy='not evaluated',
      transformation='Keep05412views/time reversal/conditionalrouting/final1person+0.35. Replace phase.project, temporal.blocks.2, temporal.project with75%original+25%previouspairedseed42weights, rounded to original storage.',
      learned_weights='Selected encoder initializers updated; all other initializers and all graph nodes unchanged.',new_training='Reuse existing paired encoder supervised training; new reliability gate experiments rejected.',
      bias='OriginalUdirect and final+0.35 unchanged; no new output bias',target_platform_accuracy='0.60 goal, not prediction',caveat='Cached source diagnostics and reused room development controls are not hidden platform accuracy.')
    helper.set_model_props(m,props);onnx.checker.check_model(m);onnx.save(m,EXPORT/'model.onnx')
    proof=dict(base_sha256=SOURCE_SHA,candidate_sha256=sha(EXPORT/'model.onnx'),original_graph_nodes=len(m.graph.node),all_nodes_unchanged=True,all_unselected_initializers_byte_identical=True,changed_initializers=changed,updated_encoder_parameters=1512192,provenance=provenance)
    save(EXPORT/'structure_proof.json',proof);print('G60_DIRECT_EXPORTED',len(changed),(EXPORT/'model.onnx').stat().st_size,flush=True)

def audit():
    torch.set_num_threads(4);d=torch.load(WORK/'actual.pt',weights_only=True);rows=d['rows']
    with np.load(WORK/'actual_paired12_42_storage.npz') as f:expected=f['logits'].copy();base=f['base'].copy()
    # Cover everychanged prediction and all probability margins<.02, plus endpoints of every recording.
    probs=torch.from_numpy(expected).softmax(-1).numpy();margin=np.sort(probs,axis=-1)[:,-1]-np.sort(probs,axis=-1)[:,-2]
    chosen={i for i,r in enumerate(rows) if int(r['start']) in (0,11400) or expected[i].argmax()!=base[i].argmax() or margin[i]<.02}
    graph=onnx.load(EXPORT/'model.onnx')
    for permutation in ('012','021','102','120','201','210'):
        for direction in (0,1):
            for suffix,shape in [('cat_8',[1,963,150]),('cat_6',[1,17])]:
                graph.graph.output.append(helper.make_tensor_value_info(f'__rxg_{permutation}___tta_v{direction}_'+suffix,onnx.TensorProto.FLOAT,shape))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;session=ort.InferenceSession(graph.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);del graph
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    reference=RawReference().cuda().eval();records=[];references=[];maxlogit=0.;maxprob=0.;cacheargdiff=0
    def prob(z):p=np.exp(z-z.max());return p/p.sum()
    with torch.inference_mode():
        for identifier in sorted({rows[i]['identifier'] for i in chosen}):
            path=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(path/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(path/f'ATI_{identifier}.b2')).reshape(-1)
            for i in sorted(chosen):
                row=rows[i]
                if row['identifier']!=identifier:continue
                start,stop=row['start'],row['stop'];feed=dict(frames=np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[start:stop].astype(np.float32,copy=True));feed['intervals'][-1]=0
                output,*observed=session.run(None,feed)
                # Audit updated networks against the SAME actual graph features.
                # Independent raw preprocessing differs in historical caches;
                # never hide that issue by loosening the numerical thresholds.
                tt=torch.from_numpy(np.concatenate(observed[::2],0)).cuda();ss=torch.from_numpy(np.concatenate(observed[1::2],0)).cuda()
                raw=reference.model.forward_features(tt,ss).reshape(1,6,2,5);ref=policy(raw,reference.bias)[0].cpu().numpy()
                le=float(np.abs(output-ref).max());pe=float(np.abs(prob(output)-prob(ref)).max());assert le<.02 and pe<.002,(i,le,pe)
                assert output.argmax()==ref.argmax() or np.sort(prob(ref))[-1]-np.sort(prob(ref))[-2]<.002
                maxlogit=max(maxlogit,le);maxprob=max(maxprob,pe);cacheargdiff+=int(output.argmax()!=expected[i].argmax())
                records.append(dict(index=i,identifier=identifier,start=start,truth=row['headcount'],prediction=int(output.argmax()),cached_prediction=int(expected[i].argmax()),logit_max_error=le,probability_max_error=pe))
                kind='corrected' if output.argmax()==row['headcount'] and base[i].argmax()!=row['headcount'] else 'broken' if output.argmax()!=row['headcount'] and base[i].argmax()==row['headcount'] else 'unchanged'
                if not any(x.get('kind')==kind for x in references):
                    file=EXPORT/f'fixture_{kind}.npz';np.savez_compressed(file,**feed);references.append(dict(input=str(file),logits=ref.tolist(),kind=kind))
            print('G60_RAW_AUDIT',identifier,len(records),'of',len(chosen),flush=True);del csi,ati;gc.collect()
        for path in (ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero'):
            if path=='zero':feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
            else:
                with np.load(path) as f:feed={k:f[k].copy() for k in ('frames','intervals')}
            out,*observed=session.run(None,feed);tt=torch.from_numpy(np.concatenate(observed[::2],0)).cuda();ss=torch.from_numpy(np.concatenate(observed[1::2],0)).cuda();ref=policy(reference.model.forward_features(tt,ss).reshape(1,6,2,5),reference.bias)[0].cpu().numpy();assert np.abs(prob(out)-prob(ref)).max()<.002
            references.append(dict(input=str(path),logits=ref.tolist()))
    save(EXPORT/'references.json',references);save(EXPORT/'raw_audit_records.json',records)
    save(EXPORT/'validation.json',dict(cached_source=read(WORK/'quantized_summary.json')['source'],raw_audited_windows=len(records),raw_total_windows=960,
      coverage='Every changed cached prediction, cached probability margin<.02, and first/last window of each recording. This is a subset audit, not all960rawONNX.',
      numerical_reference_scope='Independent PyTorch updated encoders/readouts consume the same24 actual ORT preprocessed tensors. This verifies updated network/route math, not independent raw preprocessing equivalence.',
      preprocessing_caveat='Historical double/native-log1p feature cache differs from actual054float32 exported preprocessing. Case599maxstatic/dynamic/phase inputdifferences~.001/.004/.0017. Log1p expansion alone did not fully explain it. Source/cache accuracy cannot be treated as full rawONNX accuracy.',
      independent_gpu_torch_logit_max_error=maxlogit,independent_gpu_torch_probability_max_error=maxprob,raw_vs_cached_argmax_disagreements=cacheargdiff,platform_score=None))
    print('G60_AUDIT_DONE',len(records),cacheargdiff,maxlogit,maxprob,flush=True)
if __name__=='__main__':assemble();audit()
