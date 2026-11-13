"""Reuse the raw-input audit harness; append high-count-only consensus rule."""
import importlib.util
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('shared_guard_audit',ROOT/'experiments/pair_evidence_20261024/build.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
WORK=Path('D:/huawei2026-experiments/high_count_20261024')
EXPORT=WORK/'export'
PREFIX='__high_count_consensus_'
BASE=PREFIX+'base'
RAW=audit.RAW


def policy(base,raw0,raw1):
    k=int(raw0.argmax())
    trigger=k in (2,3) and int(raw1.argmax())==k and int(base.argmax())>k
    return (raw0 if trigger else base),trigger


def assemble():
    assert audit.sha(audit.SOURCE)==audit.SOURCE_HASH
    source=onnx.load(audit.SOURCE);model=onnx.ModelProto();model.CopyFrom(source)
    assert model.graph.node[-1].output[0]=='logits' and not any('logits' in n.input for n in model.graph.node)
    model.graph.node[-1].output[0]=BASE
    for name,value in [('two',2),('three',3)]:model.graph.initializer.append(numpy_helper.from_array(np.array(value,np.int64),PREFIX+name))
    nodes=[helper.make_node('ArgMax',[BASE],[PREFIX+'base_class'],axis=0,keepdims=0),
        helper.make_node('ArgMax',[RAW[0]],[PREFIX+'original_class'],axis=0,keepdims=0),
        helper.make_node('ArgMax',[RAW[1]],[PREFIX+'reverse_class'],axis=0,keepdims=0),
        helper.make_node('Equal',[PREFIX+'original_class',PREFIX+'reverse_class'],[PREFIX+'agree']),
        helper.make_node('Equal',[PREFIX+'original_class',PREFIX+'two'],[PREFIX+'two_supported']),
        helper.make_node('Equal',[PREFIX+'original_class',PREFIX+'three'],[PREFIX+'three_supported']),
        helper.make_node('Or',[PREFIX+'two_supported',PREFIX+'three_supported'],[PREFIX+'high_supported']),
        helper.make_node('Greater',[PREFIX+'base_class',PREFIX+'original_class'],[PREFIX+'lower']),
        helper.make_node('And',[PREFIX+'agree',PREFIX+'high_supported'],[PREFIX+'consensus']),
        helper.make_node('And',[PREFIX+'consensus',PREFIX+'lower'],[PREFIX+'trigger']),
        helper.make_node('Where',[PREFIX+'trigger',RAW[0],BASE],['logits'])]
    model.graph.node.extend(nodes)
    for old,new in zip(source.graph.node,model.graph.node):
        v=onnx.NodeProto();v.CopyFrom(new)
        if BASE in v.output:v.output[0]='logits'
        assert old.SerializeToString()==v.SerializeToString()
    assert all(a.SerializeToString()==b.SerializeToString() for a,b in zip(source.graph.initializer,model.graph.initializer))
    assert model.graph.input==source.graph.input and model.graph.output==source.graph.output
    props={v.key:v.value for v in source.metadata_props}
    props.update(platform_accuracy='not evaluated',base_sha256=audit.SOURCE_HASH,
        transformation='If original/time-reversed PRE-BIAS argmax agree at2 or3 AND below base048 argmax, return original pre-bias scores; otherwise exact048.',
        training='No new learned weights or bias fit. Two integer routing constants.',
        caveat='Exploratory source-data diagnosis; high-count labels only in a1/c1. Hidden score unknown.')
    helper.set_model_props(model,props);onnx.checker.check_model(model)
    EXPORT.mkdir(parents=True,exist_ok=True);onnx.save_model(model,EXPORT/'model.onnx',save_as_external_data=False)
    audit.save(EXPORT/'structure_proof.json',dict(base_sha256=audit.SOURCE_HASH,candidate_sha256=audit.sha(EXPORT/'model.onnx'),
        original_nodes=len(source.graph.node),added_nodes=len(nodes),original_initializers=len(source.graph.initializer),
        original_weights_and_body_byte_preserved=True,predicted012_outputs_structurally_unchanged=True,
        two_added_int64_constants=True,no_extra_backbone_inference=True))


if __name__=='__main__':
    assemble()
    audit.WORK=WORK;audit.EXPORT=EXPORT;audit.BASE=BASE;audit.PREFIX=PREFIX
    audit.ELIGIBLE=lambda prediction:prediction in (3,4)
    audit.policy=policy
    audit.audit()
