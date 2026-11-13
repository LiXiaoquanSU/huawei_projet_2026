"""Reuse exact049 frozen-prefix tensors; append updated phase/time suffixes."""
import gc,blosc2,onnx,onnxruntime as ort
from onnx import helper,TensorProto
from ssl_common import *
SOURCE=ROOT/'models/final_bias_20261024/01_final_bias_one035.onnx'
SOURCE_SHA='d083e3bb36f522e29b45acb88e41e5577013f9140b2f59dd2f9741c476f839df'
BASE='__u_ssl_original049_logits'

class InferenceHead(nn.Module):
    def __init__(self,selection,identity=False):
        super().__init__();model,b=build(checkpoint('actual'),full=True);self.register_buffer('bias',b)
        self.selection=selection;models=[]
        seeds=(42,2026) if selection['variant']=='ensemble' else (int(selection['variant'].removeprefix('seed')),)
        for seed in seeds:
            fit=WORK/'fits'/f"actual_{selection['method']}_s{seed}"
            complete=read(fit/'complete.json');cp=torch.load(fit/f"step{complete['chosen_step']}.pt",weights_only=True)
            student=Student(model).eval();amount=0. if identity else selection['amount']
            state={k:cp['initial'][k]*(1-amount)+v*amount for k,v in cp['state'].items()}
            student.load_state_dict(state,strict=False);models.append(student)
        self.students=nn.ModuleList(models)
    def forward(self,base,*views):
        allraw=[];d=self.students[0].decoder
        for student in self.students:
            raw=[]
            for view in range(4):
                phase,time,a,dy,context,strength,strength_encoded=views[view*7:view*7+7]
                p,t=student.encode(phase,time);evidence=[a,dy,p,t];content=torch.cat(evidence,1)
                gate=.5*d.gate(torch.cat([content,context],1)).softmax(-1)+.125
                experts=torch.stack([h(x) for h,x in zip(d.experts,evidence)],1)
                mixture=(experts.softmax(-1)*gate[:,:,None]).sum(1)
                z=d.shared(content)+strength_encoded
                direct=.5*d.classifier(z).softmax(-1)+.5*mixture
                ordinal=d.ordinal(z);presence=d.presence(torch.cat([dy,p,t,strength],1))
                ordered,_=shared.ordered_probabilities(torch.cat([presence,ordinal[:,1:]],1))
                raw.append(((direct+ordered)*.5).clamp_min(1e-8).log())
            allraw.append(torch.stack(raw,1))
        raw=torch.stack([z.exp() for z in allraw]).mean(0).clamp_min(1e-8).log()
        score=policy(raw,self.bias).squeeze(0)
        if self.selection['deployment']=='rescue':
            votes=(raw+self.bias+raw.new_tensor([0,.35,0,0,0])).argmax(-1)
            margin=raw[:,:,1]-raw[:,:,0]
            use=(base.argmax(-1)==0)&(score.argmax(-1)==1)&((votes==1).sum(-1)[0]>=3)&(margin[0,0]>=np.log(4))&(margin[0,1]>=np.log(4))
            out=torch.where(use,score,base)
        else:out=score+base*0.
        return out,score,raw.squeeze(0)

def inputs():
    names=[BASE];examples=[torch.zeros(5)];shapes=[[5]]
    for rx,time in ((0,0),(0,1),(1,0),(1,1)):
        p=f'__rxf_v{rx}___tta_v{time}_'
        for suffix,shape in [('cat_11',[1,2304]),('add_42',[1,75,256]),('silu_8',[1,256]),('silu_17',[1,256]),('silu_28',[1,64]),('cat_18',[1,11]),('linear_27',[1,256])]:
            names.append(p+suffix);examples.append(torch.zeros(shape));shapes.append(shape)
    return names,examples,shapes

def assemble(selection):
    assert sha(SOURCE)==SOURCE_SHA;torch.set_num_threads(4)
    export=WORK/'export';export.mkdir(parents=True,exist_ok=True)
    head=InferenceHead(selection).eval();m=onnx.load(SOURCE);original=copy.deepcopy(m);m.graph.node[-1].output[0]=BASE
    names,examples,shapes=inputs();assert set(names)<={v for n in m.graph.node for v in n.output}
    torch.onnx.export(head,tuple(examples),str(export/'suffix.onnx'),input_names=names,
      output_names=['__u_ssl_result','__u_ssl_student_logits','__u_ssl_raw'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(export/'suffix.onnx');mapping={v.name:'__u_ssl_aux_'+v.name for v in small.graph.initializer}
    for n in small.graph.node:
        for v in n.output:mapping[v]='logits' if v=='__u_ssl_result' else '__u_ssl_aux_'+v
    for v in small.graph.initializer:v.name=mapping[v.name]
    for n in small.graph.node:
        assert not any(a.type==onnx.AttributeProto.GRAPH for a in n.attribute),'Need recursive remapping for control flow'
        n.name='__u_ssl_aux_'+n.name
        for i,v in enumerate(n.input):n.input[i]=mapping.get(v,v)
        for i,v in enumerate(n.output):n.output[i]=mapping.get(v,v)
    m.graph.node.extend(small.graph.node);m.graph.initializer.extend(small.graph.initializer)
    for a,b in zip(original.graph.node,m.graph.node):
        check=copy.deepcopy(b)
        if BASE in check.output:check.output[0]='logits'
        assert a.SerializeToString()==check.SerializeToString()
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(original.graph.initializer,m.graph.initializer))
    props={v.key:v.value for v in original.metadata_props}
    props.update(platform_accuracy='not evaluated',base_sha256=SOURCE_SHA,
      transformation=f"Updated phase.project, temporal.blocks.2 and temporal.project; method{selection['method']}, weightinterpolation{selection['amount']}, {selection['variant']}, deployment{selection['deployment']}. Fixed original bias/final+0.35.",
      training='Original encoders early layers and readouts frozen eval. True official source labels; source/U self-supervision only as selected. No pseudo labels/external data. Train-only projector removed at export.',
      caveat='Historical source trained diagnostics and reused independent-room development controls, not hidden score. No guarantee to reach0.52.')
    helper.set_model_props(m,props);onnx.checker.check_model(m);onnx.save(m,export/'model.onnx')
    save(export/'structure_proof.json',dict(candidate_sha256=sha(export/'model.onnx'),base_sha256=SOURCE_SHA,
      original_nodes=len(original.graph.node),added_nodes=len(small.graph.node),all_original_body_and_initializers_preserved=True,
      original_frozen_prefixes_reused=True,new_backbone_suffixes_only=True,updated_encoder_parameters_per_seed=1512192,
      training_projector_not_exported=not any('projector.' in v.name for v in small.graph.initializer),selection=selection))
    del m,original,small;gc.collect();return head,export

def audit(head,export):
    m=onnx.load(export/'model.onnx');names,_,shapes=inputs()
    for name,shape in zip(names,shapes):m.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shape))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(m.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);del m
    identity=InferenceHead(head.selection,identity=True).eval()
    d=torch.load(WORK/'prefixes/actual/source.pt',weights_only=True);rows=d['rows'];lookup={(r['identifier'],r['start']):i for i,r in enumerate(rows)}
    with np.load(Path('D:/huawei2026-experiments/final_bias_20261024/cached_exact048.npz')) as values:base_reference=values['logits']+np.array([0,.35,0,0,0],np.float32)
    records=[];references=[];maxerr=0.;baseerr=0.;identityerr=0.;identity_count=0
    def probability(z):p=np.exp(z-z.max());return p/p.sum()
    with torch.no_grad():
        for identifier in sorted({r['identifier'] for r in rows}):
            path=Path('D:/huawei2026-data/labeled');csi=blosc2.load_array(str(path/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(path/f'ATI_{identifier}.b2')).reshape(-1)
            for row in [r for r in rows if r['identifier']==identifier]:
                start,stop=row['start'],row['stop'];feed=dict(frames=np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[start:stop].astype(np.float32,copy=True));feed['intervals'][-1]=0
                out,*observed=session.run(None,feed);args=[torch.from_numpy(v) for v in observed];reference,student,raw=head(*args)
                err=float(np.abs(out-reference.numpy()).max());assert err<3e-4,err;maxerr=max(maxerr,err)
                base=observed[0];be=float(np.abs(base-base_reference[lookup[identifier,start]]).max());assert be<2e-5,be;baseerr=max(baseerr,be)
                if start==0:
                    ident=identity(*args)[0].numpy();ie=float(np.abs(probability(ident)-probability(base)).max());assert ie<2e-5,ie;identityerr=max(identityerr,ie);identity_count+=1
                pred=int(out.argmax());truth=row['headcount'];bp=int(base.argmax())
                records.append(dict(identifier=identifier,start=start,truth=truth,room=row['room'],base=bp,prediction=pred,logits=out.tolist()))
                kind=('corrected' if pred==truth else 'broken') if pred!=bp else 'unchanged'
                if not any(r.get('kind')==kind for r in references):
                    path=export/f'fixture_{kind}.npz';np.savez_compressed(path,**feed);references.append(dict(input=str(path),logits=reference.numpy().tolist(),kind=kind))
            print('SSL_RAW_AUDIT',identifier,flush=True);del csi,ati;gc.collect()
        for path in [ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero']:
            if path=='zero':feed=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
            else:
                with np.load(path) as values:feed={k:values[k].copy() for k in ('frames','intervals')}
            out,*observed=session.run(None,feed);ref=head(*[torch.from_numpy(v) for v in observed])[0].numpy();assert np.abs(out-ref).max()<3e-4
            references.append(dict(input=str(path),logits=ref.tolist()))
    del session;gc.collect();cm=np.zeros((5,5),np.int64)
    for r in records:cm[r['truth'],r['prediction']]+=1
    validation=dict(raw_windows=960,correct=int(np.trace(cm)),total=960,accuracy=float(np.trace(cm)/960),
      corrected=sum(r['prediction']==r['truth'] and r['base']!=r['truth'] for r in records),broken=sum(r['prediction']!=r['truth'] and r['base']==r['truth'] for r in records),
      confusion_matrix=cm.tolist(),independent_torch_suffix_logit_max_error=maxerr,original049_logits_max_error=baseerr,zero_update_identity_checks=identity_count,zero_update_probability_max_error=identityerr,source_diagnostic_not_independent=True,platform_score=None)
    save(export/'validation.json',validation);save(export/'all_predictions.json',records);save(export/'references.json',references);print('SSL_EXPORTED',validation,flush=True)

if __name__=='__main__':
    selected=read(WORK/'selection.json');head,export=assemble(selected);audit(head,export)
