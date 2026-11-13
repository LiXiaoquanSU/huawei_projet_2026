"""Embed the trained 26-parameter head using existing .48 intermediate tensors."""
import gc
import importlib.util
from pathlib import Path
import numpy as np
import onnx
from onnx import helper
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('learned_high_shared_audit',ROOT/'experiments/pair_evidence_20261024/build.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
WORK=Path('D:/huawei2026-experiments/high_count_20261024')
EXPORT=WORK/'learned_export'
BASE='__learned_high_base'
HEAD='__learned_high_head_logits'
PROB='__learned_high_head_probability'


class Head(nn.Module):
    def __init__(self,cp):
        super().__init__();self.register_buffer('weights',cp['weights']);self.register_buffer('bias',cp['bias'])
    def forward(self,base,*views):
        a=[];b=[]
        for k in range(4):
            for p in (views[k*3],views[k*3+1],views[k*3+2][:,0]):
                p=p.clamp_min(1e-6)
                a.append(((p[:,3]+p[:,4]).log()-p[:,2].log()).clamp(-6,6))
                b.append((p[:,4].log()-p[:,3].log()).clamp(-6,6))
        x=torch.stack([torch.stack(a,1),torch.stack(b,1)],1)
        z=(x*self.weights[None]).sum(-1)+self.bias
        q,r=z.sigmoid().unbind(-1)
        p=torch.stack([1-q,q*(1-r),q*r],1)
        high_logits=torch.cat([torch.full((1,2),-100.,dtype=base.dtype),p.clamp_min(1e-8).log()],1).squeeze(0)
        high_probability=torch.cat([torch.zeros((1,2),dtype=base.dtype),p],1).squeeze(0)
        winner=high_probability.argmax(0);base_class=base.argmax(0)
        use=(base_class>=2)&(winner<base_class)&(high_probability.max()>=.6)
        return torch.where(use,high_logits,base),high_logits,high_probability


def policy(base,head,prob):
    k=int(head.argmax());trigger=int(base.argmax())>=2 and k<int(base.argmax()) and float(prob.max())>=float(np.float32(.6))
    return (head if trigger else base),trigger


def assemble():
    torch.set_num_threads(4)
    assert audit.sha(audit.SOURCE)==audit.SOURCE_HASH
    cp=torch.load(WORK/'full_data/pair0/final.pt',map_location='cpu',weights_only=True)
    assert len(cp['fit_identifiers'])==19
    head=Head(cp).eval()
    source=onnx.load(audit.SOURCE);model=onnx.ModelProto();model.CopyFrom(source)
    model.graph.node[-1].output[0]=BASE
    names=[BASE];examples=[torch.zeros(5)]
    for rx,time in ((0,0),(0,1),(1,0),(1,1)):
        prefix=f'__rxf_v{rx}___tta_v{time}_'
        names.extend([prefix+'add_48',prefix+'cat_21',prefix+'softmax_4'])
        examples.extend([torch.ones(1,5)/5,torch.ones(1,5)/5,torch.ones(1,4,5)/5])
    available={v for n in model.graph.node for v in n.output}
    assert set(names)<=available
    EXPORT.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(head,tuple(examples),str(EXPORT/'head.onnx'),input_names=names,
        output_names=['__learned_high_result',HEAD,PROB],opset_version=18,dynamo=False,
        external_data=False,do_constant_folding=True)
    small=onnx.load(EXPORT/'head.onnx')
    assert {v.name for v in small.graph.input}==set(names)
    mapping={}
    for init in small.graph.initializer:mapping[init.name]='__learned_high_aux_'+init.name
    for node in small.graph.node:
        for name in node.output:
            if name not in (HEAD,PROB):mapping[name]='logits' if name=='__learned_high_result' else '__learned_high_aux_'+name
    for init in small.graph.initializer:init.name=mapping[init.name]
    for node in small.graph.node:
        node.name='__learned_high_aux_'+node.name
        for i,v in enumerate(node.input):node.input[i]=mapping.get(v,v)
        for i,v in enumerate(node.output):node.output[i]=mapping.get(v,v)
    model.graph.initializer.extend(small.graph.initializer);model.graph.node.extend(small.graph.node)
    for old,new in zip(source.graph.node,model.graph.node):
        n=onnx.NodeProto();n.CopyFrom(new)
        if BASE in n.output:n.output[0]='logits'
        assert old.SerializeToString()==n.SerializeToString()
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(source.graph.initializer,model.graph.initializer))
    assert model.graph.input==source.graph.input and model.graph.output==source.graph.output
    props={v.key:v.value for v in source.metadata_props}
    props.update(platform_accuracy='not evaluated',base_sha256=audit.SOURCE_HASH,
        transformation='26-parameter conditional high-count head on existing 4-view direct/ordinal/static expert evidence. Only lower base3/4 when head confidence>=.6; retain all base0/1/2 outputs.',
        training='Frozen original048 encoder; train26 positive-readout parameters600steps on all19 true2/3/4 source records. No external data or hidden labels.',
        caveat='Exploratory source-data fit. High-count labels only in two rooms; independent proxy does not show transfer benefit. Hidden score unknown.')
    helper.set_model_props(model,props);onnx.checker.check_model(model)
    onnx.save_model(model,EXPORT/'model.onnx',save_as_external_data=False)
    audit.save(EXPORT/'structure_proof.json',dict(base_sha256=audit.SOURCE_HASH,candidate_sha256=audit.sha(EXPORT/'model.onnx'),
        original_nodes=len(source.graph.node),added_nodes=len(small.graph.node),original_body_and_weights_preserved=True,
        predicted012_outputs_structurally_unchanged=True,trained_new_parameters=26,source_records=19,
        no_extra_backbone_inference=True,checkpoint_sha256=audit.sha(WORK/'full_data/pair0/final.pt'),
        reused_tensors=names[1:]))
    del model,source,small,head,cp;gc.collect()


if __name__=='__main__':
    assemble()
    audit.WORK=WORK;audit.EXPORT=EXPORT;audit.BASE=BASE;audit.RAW=[HEAD,PROB]
    audit.ELIGIBLE=lambda prediction:prediction in (3,4)
    audit.policy=policy
    audit.audit()
