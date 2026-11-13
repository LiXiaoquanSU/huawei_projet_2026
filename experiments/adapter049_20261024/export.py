"""Reuse actual scored .49 encoders; append adapter and shared frozen readout."""
import argparse,gc,json,sys,copy
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,TensorProto
import torch
from torch import nn
from torch.nn import functional as F
import onnxruntime as ort

from run import ROOT,WORK,AdapterHead,build,policy,save,read,sha,ordered_probabilities
SOURCE=ROOT/'models/final_bias_20261024/01_final_bias_one035.onnx'
SOURCE_SHA='d083e3bb36f522e29b45acb88e41e5577013f9140b2f59dd2f9741c476f839df'
BASE='__adapter049_original_logits'

class ExportHead(nn.Module):
    def __init__(self,head,bias,scale,mix):
        super().__init__();self.head=head;self.register_buffer('bias',bias);self.scale=scale;self.mix=mix
    def forward(self,base,*args):
        raw=[]
        h=self.head
        for v in range(4):
            a,d,p,t,context,strength,strength_encoded=args[v*7:v*7+7]
            t=t+self.scale*h.up(F.silu(h.down(t)))
            evidence=[a,d,p,t];content=torch.cat(evidence,1)
            gate=.5*h.gate(torch.cat([content,context],1)).softmax(-1)+.125
            experts=torch.stack([module(x) for module,x in zip(h.experts,evidence)],1)
            mixture=(experts.softmax(-1)*gate[:,:,None]).sum(1)
            z=h.shared(content)+strength_encoded
            direct=.5*h.classifier(z).softmax(-1)+.5*mixture
            ordinal=h.ordinal(z);presence=h.presence(torch.cat([d,p,t,strength],1))
            ordered,_=ordered_probabilities(torch.cat([presence,ordinal[:,1:]],1))
            raw.append(((direct+ordered)*.5).clamp_min(1e-8).log())
        adapted=policy(torch.stack(raw,1),self.bias).squeeze(0)
        # Fixed probability interpolation limits drift, not another class bias.
        out=((1-self.mix)*base.softmax(0)+self.mix*adapted.softmax(0)).clamp_min(1e-8).log()
        return out,adapted

def names_and_examples():
    names=[BASE];examples=[torch.zeros(5)]
    for rx,time in ((0,0),(0,1),(1,0),(1,1)):
        prefix=f'__rxf_v{rx}___tta_v{time}_'
        names.extend(prefix+k for k in ('silu_8','silu_17','silu_26','silu_27','silu_28','cat_18','linear_27'))
        examples.extend(torch.zeros(1,n) for n in (256,256,256,256,64,11,256))
    return names,examples

def assemble(cp_path,mix):
    torch.set_num_threads(4)
    cp=torch.load(cp_path,weights_only=True);model,b=build(ROOT/'models/best_037/final.pt',full=True)
    head=AdapterHead(model).eval();head.load_state_dict(cp['adapter'],strict=False)
    wrapper=ExportHead(head,b,cp['scale'],mix).eval()
    export=WORK/'exports'/f"{cp['method']}_s{cp['scale']}_mix{mix}"
    export.mkdir(parents=True,exist_ok=True)
    assert sha(SOURCE)==SOURCE_SHA
    source=onnx.load(SOURCE);m=copy.deepcopy(source);assert m.graph.node[-1].output[0]=='logits'
    m.graph.node[-1].output[0]=BASE
    names,examples=names_and_examples();assert set(names)<={v for n in m.graph.node for v in n.output}
    torch.onnx.export(wrapper,tuple(examples),str(export/'head.onnx'),input_names=names,
        output_names=['__adapter_result','__adapter_adapted_logits'],opset_version=18,dynamo=False,external_data=False)
    small=onnx.load(export/'head.onnx')
    mapping={x.name:'__adapter_aux_'+x.name for x in small.graph.initializer}
    for n in small.graph.node:
        for v in n.output:mapping[v]='logits' if v=='__adapter_result' else '__adapter_aux_'+v
    for x in small.graph.initializer:x.name=mapping[x.name]
    for n in small.graph.node:
        n.name='__adapter_aux_'+n.name
        for i,v in enumerate(n.input):n.input[i]=mapping.get(v,v)
        for i,v in enumerate(n.output):n.output[i]=mapping.get(v,v)
    m.graph.node.extend(small.graph.node);m.graph.initializer.extend(small.graph.initializer)
    for a,z in zip(source.graph.node,m.graph.node):
        check=copy.deepcopy(z)
        if BASE in check.output:check.output[0]='logits'
        assert a.SerializeToString()==check.SerializeToString()
    assert all(a.SerializeToString()==z.SerializeToString() for a,z in zip(source.graph.initializer,m.graph.initializer))
    props={v.key:v.value for v in source.metadata_props}
    props.update(platform_accuracy='not evaluated',base_sha256=SOURCE_SHA,
      transformation=f"Temporal residual adapter256-32-256, method{cp['method']}, step{cp['step']}, residualscale{cp['scale']}, probabilitymix{mix}; fixed .44/.49 biases unchanged",
      training='Only16672 adapter parameters trained. Original encoders/readouts frozen eval. All48 official labeled source records. No external or hidden labels.',
      caveat='Source diagnostic accuracy is not independent test accuracy. Hidden platform improvement unknown.')
    helper.set_model_props(m,props);onnx.checker.check_model(m);onnx.save(m,export/'model.onnx')
    save(export/'structure_proof.json',dict(base_sha256=SOURCE_SHA,candidate_sha256=sha(export/'model.onnx'),original_nodes=len(source.graph.node),added_nodes=len(small.graph.node),original_nodes_and_weights_preserved=True,extra_encoder_inferences=0,trained_parameters=16672,checkpoint_sha256=sha(cp_path),mix=mix,scale=cp['scale']))
    del m,source,small,model;gc.collect()
    # Independent Torch recomputation on actual original ONNX intermediate values.
    inst=onnx.load(export/'model.onnx')
    for name,n in zip(names,[5]+[x for _ in range(4) for x in (256,256,256,256,64,11,256)]):
        inst.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[5] if name==BASE else [1,n]))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(inst.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider']);del inst
    fixtures=[ROOT/'checks/smoke_input.npz',ROOT/'checks/case_gap.npz','zero']
    # Whole-record sampling: one original raw window per available label/room record.
    import blosc2
    cache=torch.load(Path('D:/huawei2026-experiments/evidence049_research_20261024/frozen_latents.pt'),weights_only=True)
    rows=cache['rows'];references=[];parity=[];preds=[];baseline_errors=[]
    with np.load(Path('D:/huawei2026-experiments/final_bias_20261024/cached_exact048.npz')) as d:
        exactbase=d['logits']+np.array([0,.35,0,0,0],np.float32)
    lookup={(r['identifier'],r['start']):i for i,r in enumerate(rows)}
    identity=copy.deepcopy(wrapper)
    nn.init.zeros_(identity.head.up.weight);nn.init.zeros_(identity.head.up.bias)
    with torch.no_grad():
        for identifier in sorted({r['identifier'] for r in rows}):
            rawdir=Path('D:/huawei2026-data/labeled')
            csi=blosc2.load_array(str(rawdir/f'CSI_{identifier}.b2'));ati=blosc2.load_array(str(rawdir/f'ATI_{identifier}.b2')).reshape(-1)
            for r in [x for x in rows if x['identifier']==identifier]:
                start,stop=r['start'],r['stop']
                inputs=dict(frames=np.ascontiguousarray(csi[start:stop]).view(np.float32).reshape(600,1,3,1024,2),intervals=ati[start:stop].astype(np.float32,copy=True));inputs['intervals'][-1]=0
                out,*inter=session.run(None,inputs)
                expected=wrapper(*[torch.from_numpy(v) for v in inter])[0].numpy()
                error=float(np.abs(out-expected).max());assert error<2e-4,error
                baseline=inter[0]
                base_error=float(np.abs(baseline-exactbase[lookup[identifier,start]]).max());assert base_error<2e-5,base_error
                baseline_errors.append(base_error)
                if start==0:
                    id_out=identity(*[torch.from_numpy(v) for v in inter])[0].numpy()
                    prob=lambda x:np.exp(x-x.max())/np.exp(x-x.max()).sum()
                    assert np.abs(prob(id_out)-prob(baseline)).max()<2e-5
                preds.append(dict(identifier=identifier,start=start,truth=r['headcount'],room=r['room'],base=int(baseline.argmax()),prediction=int(out.argmax()),logits=out.tolist()))
                parity.append(error)
                # Representative active cases only; check_candidate repeats each20x.
                kind=('corrected' if out.argmax()==r['headcount'] else 'broken') if out.argmax()!=baseline.argmax() else 'unchanged'
                if not any(ref.get('kind')==kind for ref in references):
                    fixture=export/f'fixture_{kind}.npz';np.savez_compressed(fixture,**inputs)
                    references.append(dict(input=str(fixture),logits=expected.tolist(),kind=kind))
            print('RAW_RECORD',identifier,'complete',flush=True)
            del csi,ati;gc.collect()
        for path in fixtures:
            if path=='zero':inputs=dict(frames=np.zeros((600,1,3,1024,2),np.float32),intervals=np.zeros(600,np.float32))
            else:
                with np.load(path) as d:inputs={k:d[k].copy() for k in ('frames','intervals')}
            out,*inter=session.run(None,inputs);expected=wrapper(*[torch.from_numpy(v) for v in inter])[0].numpy()
            assert np.abs(out-expected).max()<2e-4
            references.append(dict(input=str(path),logits=expected.tolist()))
    del session;gc.collect()
    cm=np.zeros((5,5),np.int64)
    for r in preds:cm[r['truth'],r['prediction']]+=1
    result=dict(raw_windows=len(preds),correct=int(np.trace(cm)),total=960,accuracy=float(np.trace(cm)/960),
      corrected=sum(r['prediction']==r['truth'] and r['base']!=r['truth'] for r in preds),
      broken=sum(r['prediction']!=r['truth'] and r['base']==r['truth'] for r in preds),confusion_matrix=cm.tolist(),
      independent_torch_readout_logit_max_error=max(parity),independent_test=False,platform_score=None)
    result['frozen_actual049_baseline_logit_max_error']=max(baseline_errors)
    result['zero_adapter_actual_raw_record_identity_checks']=48
    save(export/'validation.json',result);save(export/'all_predictions.json',preds);save(export/'references.json',references)
    print('EXPORTED',export,result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--mix',type=float,default=.5);a=p.parse_args();assemble(a.checkpoint,a.mix)
